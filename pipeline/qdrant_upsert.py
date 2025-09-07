from __future__ import annotations
import os, json, math, time
from pathlib import Path
from dotenv import load_dotenv
from typing import Iterable, List, Dict, Any
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    OptimizersConfigDiff,
    HnswConfigDiff,
    PayloadSchemaType,
    TextIndexParams,
)

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# ---------- config ----------
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COLLECTION = "music_chunks"                   # change if you like
BATCH_SIZE = 300                             # upsert batch size
DISTANCE = Distance.COSINE                    # cosine is standard for OpenAI embeds

# ---------- helpers ----------
def load_artifacts(artifacts_dir: Path):
    m = json.loads((artifacts_dir / "manifest.json").read_text(encoding="utf-8"))
    dim = int(m["dim"])
    ids = np.load(artifacts_dir / "ids.npy")           # uint64
    vecs = np.load(artifacts_dir / "vectors.npy")      # float32 [N, dim]
    metas = [json.loads(l) for l in (artifacts_dir / "meta.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(ids) == len(vecs) == len(metas), "Row count mismatch across artifacts."
    return dim, ids, vecs, metas

def chunks(n: int, size: int):
    for i in range(0, n, size):
        yield slice(i, min(i + size, n))

def ensure_collection(client: QdrantClient, dim: int):
    # create or recreate a collection with HNSW; for tiny corpora, defaults are fine.
    existing = None
    try:
        existing = client.get_collection(COLLECTION)
    except Exception:
        pass

    if existing is None:
        client.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=DISTANCE),
            hnsw_config=HnswConfigDiff(m=32, ef_construct=128),   # safe defaults
            optimizers_config=OptimizersConfigDiff(memmap_threshold=20000),
        )
        # Optional: declare payload schema + indexes for fast filtering / full-text
        client.create_payload_index(COLLECTION, field_name="artist", field_schema=PayloadSchemaType.KEYWORD)
        client.create_payload_index(COLLECTION, field_name="song_name", field_schema=PayloadSchemaType.KEYWORD)
        # Full-text index on content (stored in payload) if you want hybrid queries later
        try:
            client.create_payload_index(
                COLLECTION,
                field_name="content",
                field_schema=PayloadSchemaType.TEXT,
                field_index_params=TextIndexParams(tokenizer="en", min_token_len=2),
            )
        except Exception:
            # text index requires enterprise/feature availability; ignore if not supported
            pass
    else:
        # sanity-check dimension
        col = client.get_collection(COLLECTION)
        got = col.config.params.vectors.size  # type: ignore
        if got != dim:
            raise ValueError(f"Collection exists with dim={got}, but artifacts are dim={dim}.")

def to_points(ids: np.ndarray, vecs: np.ndarray, metas: list[dict]) -> List[PointStruct]:
    # Put the chunk text into payload as 'content' (already in meta.jsonl per your build script)
    return [
        PointStruct(
            id=int(ids[i]),
            vector=vecs[i].tolist(),
            payload=metas[i],
        )
        for i in range(len(ids))
    ]

def upsert_all(client: QdrantClient, ids: np.ndarray, vecs: np.ndarray, metas: list[dict]):
    n = len(ids)
    for sl in chunks(n, BATCH_SIZE):
        pts = to_points(ids[sl], vecs[sl], metas[sl])
        client.upsert(collection_name=COLLECTION, points=pts)
    # You can also tune write consistency:
    # client.upsert(..., wait=True)  # block until persisted

def verify_sample_search(client: QdrantClient, query_vec: list[float], top_k=5):
    hits = client.search(
        collection_name=COLLECTION,
        query_vector=query_vec,
        limit=top_k,
        with_payload=True,
    )
    return [{"score": h.score, "artist": h.payload.get("artist"), "song": h.payload.get("song_name"),
             "preview": (h.payload.get("content") or "")[:120]} for h in hits]

# ---------- main ----------
if __name__ == "__main__":
    root = Path(r"C:\Projects\VibeAI\pipeline\music-rag")              # dataset root
    artifacts = root / "artifacts"        # where build_local_embeddings.py wrote files
    dim, ids, vecs, metas = load_artifacts(artifacts)

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30.0)
    ensure_collection(client, dim)
    upsert_all(client, ids, vecs, metas)

    print(f"Uploaded {len(ids)} points to '{COLLECTION}'.")

    # Quick smoke test (embed a query elsewhere; here we reuse any one vector)
    test_vec = vecs[0].tolist()
    sample = verify_sample_search(client, test_vec, top_k=3)
    for i, r in enumerate(sample, 1):
        print(f"{i}. {r['artist']} - {r['song']}  (score={r['score']:.4f})  '{r['preview']}'")