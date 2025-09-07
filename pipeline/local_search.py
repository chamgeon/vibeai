from pathlib import Path
import json
from typing import List
import numpy as np
import faiss

from openai import OpenAI



client = OpenAI()
OPENAI_EMBED_MODEL = "text-embedding-3-small"

def embed_batch(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def _read_meta_jsonl(path: Path):
    """Return a list of metadata dicts in the same order vectors were added to FAISS."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines]


def load_faiss_only(load_dir: Path):
    """
    Load only:
      - meta.jsonl   (payload for each vector, by position)
      - index.faiss  (the FAISS index)
    """
    meta_path = load_dir / "meta.jsonl"
    index_path = load_dir / "index.faiss"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}. Needed to map results to records.")
    if not index_path.exists():
        raise FileNotFoundError(f"Missing {index_path}. Build/save a FAISS index first.")

    metas = _read_meta_jsonl(meta_path)
    index = faiss.read_index(str(index_path))

    # Quick sanity check: when index has ntotal != len(metas), warn hard
    if getattr(index, "ntotal", None) is not None and index.ntotal != len(metas):
        raise RuntimeError(
            f"FAISS index.ntotal ({index.ntotal}) != number of meta records ({len(metas)}). "
            "Make sure meta.jsonl was created in the same order as vectors were added."
        )

    return metas, index


def search_faiss_only(query: str, load_dir: Path, top_k: int = 5):
    """
    Search using FAISS only (plus meta.jsonl for payloads).
    """
    metas, index = load_faiss_only(load_dir)

    # Embed + normalize query (cosine via Inner Product on unit vectors)
    q = np.asarray([embed_batch([query])[0]], dtype="float32")
    faiss.normalize_L2(q)  # safe even if already normalized

    D, I = index.search(q, top_k)  # I are positional indices if index isn't an IDMap

    hits = []
    for rank, pos in enumerate(I[0]):
        if pos < 0:  # FAISS returns -1 when not enough results
            continue
        rec = metas[int(pos)].copy()
        rec["_score"] = float(D[0][rank])  # cosine similarity if vectors were normalized at build time
        rec["_rank"] = rank + 1
        rec["_pos"] = int(pos)            # position within the index
        hits.append(rec)

    return hits