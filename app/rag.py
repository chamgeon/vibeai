import os
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, NamedVector
from typing import List, Dict


QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COLLECTION = "music_chunks"
OPENAI_EMBED_MODEL = "text-embedding-3-small"


def process_vibe_extraction(vibe_extraction: Dict) -> List[str]:
    """
    Process vibe extraction dictionary into a batch to feed into embedding.
    """

    if not isinstance(vibe_extraction, dict):
        raise ValueError("Input must be a dictionary")
    
    description = vibe_extraction.get("description", "")
    imagination = vibe_extraction.get("imagination", "")
    flattened_vibes = [
        f"{v['label']} - {v['explanation']}"
        for v in vibe_extraction.get("vibes", [])
        if isinstance(v, dict) and "label" in v and "explanation" in v
    ]

    return [description,imagination] + flattened_vibes


def embed_batch(openai_client, texts: List[str]) -> List[List[float]]:
    resp = openai_client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def search_qdrant_batch(qdrant_client, collection: str, query_vecs: list[float], top_k=5, artist: str | None=None):
    """
    Retrieve similarity search results for a batch of query vectors.

    Args:
        qdrant_client: Qdrant client instance
        collection: name of qdrant collection
        query_vecs (list[list[float]]): Batch of query vectors
        top_k (int): Number of top results per query
        artist (str | None): Optional filter by artist

    Returns:
        list[list[dict]]: For each query vector, a list of result dicts
    """
    f = None
    if artist:
        f = Filter(must=[FieldCondition(key="artist", match=MatchValue(value=artist))])

    results = qdrant_client.search_batch(
        collection_name=collection,
        requests=[
            {
                "vector": vec,
                "limit": top_k,
                "with_payload": True,
                "filter": f,
            }
            for vec in query_vecs
        ]
    )

    # Format into a list of lists of dicts
    return [
        [{"score": h.score, **h.payload} for h in hits]
        for hits in results
    ]


def retrieval_vibe(vibe_extraction, openai_client, qdrant_client, collection, top_k):
    """
    Retrives similarity search result from vibe extraction.
    """
    batch_to_embed = process_vibe_extraction(vibe_extraction)
    embeddings_batch = embed_batch(openai_client, batch_to_embed)
    search_result = search_qdrant_batch(qdrant_client,collection,embeddings_batch,top_k)

    return search_result