from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document
import tiktoken
import json
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

def chunk_markdown(
    md_text: str,
    chunk_size: int = 900,
    chunk_overlap: int = 120,
):
    # 1) Split by markdown headers to preserve section semantics
    headers = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers,
        strip_headers=False,   # keep the header text in the chunks
    )
    sections = header_splitter.split_text(md_text)  # -> list of Documents

    # 2) Recursive, token-aware splitting inside each section
    #    Prefer to break on code fences & headers first, then paragraphs/sentences.
    enc = tiktoken.get_encoding("cl100k_base")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda s: len(enc.encode(s)),
        separators=[
            "\n```",        # don’t slice through fenced code blocks
            "\n###### ",
            "\n##### ",
            "\n#### ",
            "\n### ",
            "\n## ",
            "\n# ",
            "\n\n",         # paragraphs
            "\n",           # lines
            " ",            # words
            "",             # characters (last resort)
        ],
    )

    # Create final chunks, preserving header metadata
    chunks = []
    for doc in sections:
        split_docs = text_splitter.create_documents(
            texts=[doc.page_content],
            metadatas=[doc.metadata],
        )
        chunks.extend(split_docs)

    return chunks  # list[Document] with .page_content and .metadata

def save_chunks(chunks: List[Document], path: Path):
    # save chunks as a line in a single jsonl file, as a form of dictionary

    with path.open("w", encoding="utf-8") as f:
        for doc in chunks:
            record = {
                "content": doc.page_content,
                "metadata": doc.metadata or {},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_meta(song_dir: Path) -> Dict[str, Any]:
    """
    Load ../meta.json for a given digestion folder or song root.
    Returns {} on error/missing file.
    """
    meta_path = (song_dir / "meta.json") if song_dir.is_dir() else (song_dir.parent / "meta.json")
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _augment_metadata_from_meta(
    chunks: List[Document],
    md_path: Path
) -> None:
    """
    Add metadata from meta.json (+ path hints) onto each Document:
      - song_name, artist
      - youtube_video_ids (if present)
    """
    digestion_dir = md_path.parent                          # .../<Artist - Song>/digestion/
    song_dir = digestion_dir.parent                         # .../<Artist - Song>/
    meta = _load_meta(song_dir)

    song_name = meta.get("song_name")
    artist = meta.get("artist")
    yt_ids = meta.get("sources", {}).get("youtube", None)


    for d in chunks:
        meta_out = dict(d.metadata or {})
        if song_name is not None:
            meta_out.setdefault("song_name", song_name)
        if artist is not None:
            meta_out.setdefault("artist", artist)
        if yt_ids:
            meta_out.setdefault("youtube_video_ids", yt_ids)
        d.metadata = meta_out


def build_markdown_chunks_for_dataset(
    dataset_root: Path,
    replace_existing: bool = False,
    chunk_size: int = 900,
    chunk_overlap: int = 120,
) -> None:
    """
    Walk music-rag dataset, find each 'songs/*/digestion/summary.md',
    chunk it, and save 'chunks.jsonl' in the same 'digestion' folder.

    Args:
        dataset_root: path to 'music-rag/' root folder.
        replace_existing: if False, skip when chunks.jsonl already exists.
        chunk_size, chunk_overlap: splitter parameters.
    """
    summaries = list(dataset_root.glob("songs/*/digestion/summary.md"))
    if not summaries:
        print("No summary.md files found under songs/*/digestion/")
        return

    for md_path in summaries:
        out_path = md_path.with_name("chunks.jsonl")
        if out_path.exists() and not replace_existing:
            print(f"SKIP (exists): {out_path}")
            continue

        try:
            text = md_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"ERROR reading {md_path}: {e}")
            continue

        chunks = chunk_markdown(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        _augment_metadata_from_meta(chunks, md_path)

        try:
            save_chunks(chunks, out_path)
        except Exception as e:
            print(f"ERROR writing {out_path}: {e}")
            continue

        print(f"WROTE {out_path} ({len(chunks)} chunks)")




if __name__ == "__main__":
    root = Path(r"C:\Projects\VibeAI\pipeline\music-rag")
    build_markdown_chunks_for_dataset(
        dataset_root=root,
        replace_existing=False,   # set True to overwrite existing chunks.jsonl
        chunk_size=900,
        chunk_overlap=120,
    )