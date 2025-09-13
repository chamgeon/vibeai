from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
import os
import json
from typing import List
import time

from app.playlist_gpt import vibe_extraction_schema, call_gpt_and_verify
from app.utils import resize_image_by_longest_side
from app.prompts import VIBE_EXTRACTION_PROMPT

env_path = Path(r"C:\Projects\VibeAI\pipeline\.env")
load_dotenv(dotenv_path=env_path)

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COLLECTION = "music_chunks"
OPENAI_EMBED_MODEL = "text-embedding-3-small" 

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30.0)
openai_client = OpenAI()

def search_qdrant(qdrant_client, query_vec: list[float], top_k=5, artist: str | None=None):
    f = None
    if artist:
        f = Filter(must=[FieldCondition(key="artist", match=MatchValue(value=artist))])
    hits = qdrant_client.search(
        collection_name="music_chunks",
        query_vector=query_vec,
        query_filter=f,
        with_payload=True,
        limit=top_k,
    )
    return [{"score": h.score, **h.payload} for h in hits]


def embed_batch(openai_client, texts: List[str]) -> List[List[float]]:
    resp = openai_client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def retrieval_test_single_query(query:str):

    query_list = [query]

    embedding_start = time.time()
    embedding = embed_batch(openai_client, query_list)[0]
    print(f"embedding spent {time.time() - embedding_start} secs")
    sample = search_qdrant(qdrant_client, embedding, 20)

    for i, r in enumerate(sample, 1):
        print(f"{i}. {r['artist']} - {r['song_name']}  (score={r['score']:.4f})  '{r['content'][:150]}'")


def extract_vibe(photo_dir: Path, vibe_dir: Path):
    """
    extract vibes from photos in photo_dir directory, and save the result in vibe_dir directory.
    The result files is json file with schema:
    {
        original_source: path of the input photo file,
        description: description of the photo (gpt output),
        imagination: imagination of the situation (gpt output),
        vibes: [
            {
            label: vibe label of the photo (gpt output),
            explanation: explanation for the vibe (gpt output)
            },
            ...
        ]
    }
    """

    vibe_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png"}
    saved_paths = []
    errors = []

    # Deterministic order
    for p in sorted(photo_dir.iterdir()):
        if not (p.is_file() and p.suffix.lower() in exts):
            continue

        try:
            with p.open("rb") as f:
                # Resize image (expects file-like input; returns file-like output)
                resized_file = resize_image_by_longest_side(f, 512)

                # Give the resized file a name attribute (helps MIME guessing downstream)
                if not hasattr(resized_file, "name"):
                    try:
                        resized_file.name = p.name  # e.g., "photo.jpg"
                    except Exception:
                        pass  # non-fatal

                # Call GPT with prompt + schema + image
                result = call_gpt_and_verify(
                    VIBE_EXTRACTION_PROMPT,
                    vibe_extraction_schema,
                    file_obj=resized_file
                )


            # Ensure original_source is present
            result.setdefault("original_source", str(p.resolve()))

            # Write JSON next to vibes directory with same stem
            out_path = vibe_dir / f"{p.stem}.json"
            with out_path.open("w", encoding="utf-8") as out:
                json.dump(result, out, ensure_ascii=False, indent=2)

            saved_paths.append(str(out_path))

        except Exception as e:
            errors.append((str(p), repr(e)))
        

    return {"saved": saved_paths, "errors": errors}






if __name__ == "__main__":
    query = "The casual positions and the clear, sunny day create an atmosphere that feels light-hearted and calm, where both humans and animals coexist in playfulness and mutual respect."
    retrieval_test_single_query(query)




