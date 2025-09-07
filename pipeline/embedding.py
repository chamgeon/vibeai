import os, json, hashlib, zlib
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple
import numpy as np

from openai import OpenAI

try:
    import faiss
except Exception:
    faiss = None

client = OpenAI()
OPENAI_EMBED_MODEL = "text-embedding-3-small"   # 1536-d
EMB_DIM = 1536
BATCH_SIZE = 64

def iter_chunks_files(dataset_root: Path) -> Iterable[Path]:
    # music-rag/songs/*/digestion/chunks.jsonl
    yield from dataset_root.glob("songs/*/digestion/chunks.jsonl")

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

def deterministic_id(payload: Dict[str, Any], content: str) -> int:
    # stable 64-bit int id from a small canonical string
    src = f"{payload.get('song_name','')}|{content}"
    h = hashlib.sha1(src.encode("utf-8")).digest()[:8]
    return int.from_bytes(h, "big", signed=False)

def embed_batch(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def build_local_embeddings(
    dataset_root: Path,
    out_dir: Path | None = None,
    build_faiss: bool = True,
) -> dict:
    """
    Walk all chunks.jsonl, embed, and write artifacts to out_dir (default: music-rag/artifacts).
    Returns a tiny summary dict.
    """
    if out_dir is None:
        out_dir = dataset_root / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect corpus
    rows: List[Tuple[int, str, Dict[str, Any]]] = []
    for p in iter_chunks_files(dataset_root):
        for obj in read_jsonl(p):
            content = obj["content"]
            meta = obj.get("metadata", {})
            pid = deterministic_id(meta, content)
            rows.append((pid, content, meta))

    if not rows:
        print("No chunks found. Did you run the chunking step?")
        return {"count": 0}

    # Sort by id for determinism
    rows.sort(key=lambda r: r[0])

    # Embed
    ids = np.array([r[0] for r in rows], dtype=np.uint64)
    texts = [r[1] for r in rows]
    metas = [r[2] | {"content": r[1]} for r in rows]  # keep content in payload file

    vecs = np.zeros((len(rows), EMB_DIM), dtype="float32")
    for i in range(0, len(rows), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        embs = embed_batch(batch)
        vecs[i:i+len(embs)] = np.asarray(embs, dtype="float32")
        print(f"embedding finished {i+BATCH_SIZE}/{len(rows)}")

    # Save artifacts
    vectors_path = out_dir / "vectors.npy"
    ids_path     = out_dir / "ids.npy"
    meta_path    = out_dir / "meta.jsonl"
    manifest_path= out_dir / "manifest.json"

    np.save(vectors_path, vecs)
    np.save(ids_path, ids)
    with meta_path.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Optional FAISS index (cosine via IP on normalized vectors)
    index_path = None
    if build_faiss:
        if faiss is None:
            print("FAISS not installed; skipping index. pip install faiss-cpu")
        else:
            x = vecs.copy()
            faiss.normalize_L2(x)
            index = faiss.IndexFlatIP(EMB_DIM)
            index.add(x)
            index_path = out_dir / "index.faiss"
            faiss.write_index(index, str(index_path))

    # Write manifest
    def _crc32(p: Path) -> str:
        return f"{zlib.crc32(p.read_bytes()) & 0xffffffff:08x}"

    manifest = {
        "model": OPENAI_EMBED_MODEL,
        "dim": EMB_DIM,
        "count": int(vecs.shape[0]),
        "files": {
            "vectors.npy": {"bytes": vectors_path.stat().st_size, "crc32": _crc32(vectors_path)},
            "ids.npy":     {"bytes": ids_path.stat().st_size,     "crc32": _crc32(ids_path)},
            "meta.jsonl":  {"bytes": meta_path.stat().st_size,    "crc32": _crc32(meta_path)},
        },
    }
    if index_path and index_path.exists():
        manifest["files"]["index.faiss"] = {
            "bytes": index_path.stat().st_size, "crc32": _crc32(index_path)
        }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"✅ Wrote {manifest['count']} embeddings to {out_dir}")
    return {"out_dir": str(out_dir), **manifest}


# -------- CLI-ish usage --------
if __name__ == "__main__":
    root = Path(r"C:\Projects\VibeAI\pipeline\music-rag")
    build_local_embeddings(dataset_root=root, out_dir=root / "artifacts", build_faiss=True)