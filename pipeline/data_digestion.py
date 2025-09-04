from prompts import YOUTUBE_COMMENTS_DIGESTION_PROMPT
from scrapers import youtube_comment_scrape
from schema import Comment
from openai import OpenAI
import os, json, hashlib
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import time

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY']
)

def call_gpt(prompt, model, temperature = 0.9):

    if model == "gpt-5":
        response = client.responses.create(
            model=model,
            input=prompt,
            reasoning={ "effort": "low"},
            text = {"verbosity": "medium"}
        )

        return response.output_text
    
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            temperature=temperature,
            max_tokens=1000
        )

        return response.choices[0].message.content

def _safe_name(s: str) -> str:
    # simple filesystem-safe slug (keep spaces & dashes readable)
    return "".join(ch for ch in s if ch.isalnum() or ch in " -_().").strip()

def _song_dir(base_dir: str, song: str, artist: str) -> str:
    return os.path.join(base_dir, "songs", f"{_safe_name(artist)} - {_safe_name(song)}")



# ----------------- saving functions ---------------- #



def save_youtube_comments_jsonl(
    base_dir: str,
    song: str,
    artist: str,
    comments: List["Comment"]
) -> Tuple[Dict[str, int], List[str]]:
    """
    Writes one JSONL file per video under:
      songs/<Artist - Song>/sources/youtube_<videoId>.comments.jsonl
    Returns: (counts_by_video_id, file_paths)
    """
    grouped: Dict[str, List["Comment"]] = defaultdict(list)
    for c in comments:
        grouped[c.video_id].append(c)

    dest_dir = os.path.join(_song_dir(base_dir, song, artist), "sources")
    os.makedirs(dest_dir, exist_ok=True)

    counts, paths = {}, []
    for vid, items in grouped.items():
        path = os.path.join(dest_dir, f"youtube_{vid}.comments.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for c in items:
                # Pydantic: model_dump() in v2, dict() in v1 – use getattr for compatibility
                payload = c.model_dump() if hasattr(c, "model_dump") else c.dict()
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        counts[vid] = len(items)
        paths.append(path)
    return counts, paths

def save_digestion(
    base_dir: str,
    song: str,
    artist: str,
    model: str,
    digestion_markdown: str,
    video_ids: List[str],
    total_comment_count: int,
    prompt_used: str
) -> Tuple[str, str]:
    """
    Writes:
      songs/<Artist - Song>/digestion/summary.md
      songs/<Artist - Song>/digestion/summary.json
    Also (optionally) updates songs/<Artist - Song>/meta.json with sources + timestamps.
    Returns: (summary_md_path, summary_json_path)
    """
    ddir = os.path.join(_song_dir(base_dir, song, artist), "digestion")
    os.makedirs(ddir, exist_ok=True)

    # markdown
    md_path = os.path.join(ddir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(digestion_markdown.strip() + "\n")

    # metadata json
    created_at = datetime.utcnow().isoformat() + "Z"
    prompt_fingerprint = hashlib.sha256(prompt_used.encode("utf-8")).hexdigest()

    j = {
        "song_name": song,
        "artist": artist,
        "model": model,
        "created_at": created_at,
        "comment_count": total_comment_count,
        "sources": {"youtube": video_ids},
        "prompt_fingerprint": f"sha256:{prompt_fingerprint}"
        # You can add: "schema_version": "digestion-v1"
    }
    json_path = os.path.join(ddir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(j, f, ensure_ascii=False, indent=2)

    # update meta.json (nice to have)
    _update_meta(base_dir, song, artist, video_ids, created_at)

    return md_path, json_path

def _update_meta(base_dir: str, song: str, artist: str, video_ids: List[str], ts_iso: str) -> None:
    sdir = _song_dir(base_dir, song, artist)
    os.makedirs(sdir, exist_ok=True)
    meta_path = os.path.join(sdir, "meta.json")

    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    meta.update({
        "song_name": song,
        "artist": artist,
        "sources": {
            **(meta.get("sources") or {}),
            "youtube": sorted(set((meta.get("sources") or {}).get("youtube", []) + video_ids)),
        },
        "last_updated": ts_iso,
    })
    if "created_at" not in meta:
        meta["created_at"] = ts_iso

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)



# ------------- digestion -------------------



def youtube_comment_digest(
    song: str,
    artist: str,
    model: str = "gpt-5",
    max_comments: int = 25,
    save_dir: Optional[str] = None,      # e.g., "/Users/you/music-rag"
    save_files: bool = True
) -> str:
    """
    Scrape comments, build prompt, call GPT, and (optionally) save:
      - per-video raw comments (JSONL)
      - digestion markdown + json

    Returns:
        digestion markdown (str)
    """
    comments = youtube_comment_scrape(song_name=song, artist_name=artist, max_comments=max_comments)
    if not comments:
        return "No YouTube comments found."

    # Join raw comments for the prompt
    sep = "\n\n-----\n\n"
    raw_comments = sep.join(c.text for c in comments if c.text)

    prompt = YOUTUBE_COMMENTS_DIGESTION_PROMPT.replace("[COMMENTS]", raw_comments)
    digestion_md = call_gpt(prompt, model)

    if save_files and save_dir:
        # 1) save raw comments per video
        counts_by_vid, _paths = save_youtube_comments_jsonl(save_dir, song, artist, comments)

        # 2) save digestion (md + json) w/ provenance
        video_ids = list(counts_by_vid.keys())
        total_count = sum(counts_by_vid.values())
        save_digestion(
            base_dir=save_dir,
            song=song,
            artist=artist,
            model=model,
            digestion_markdown=digestion_md,
            video_ids=video_ids,
            total_comment_count=total_count,
            prompt_used=prompt,
        )

    return digestion_md




if __name__ == "__main__":
    base_dir = "C:\Projects\VibeAI\pipeline\music-rag"
    song = "Better left unsaid"
    artist = "stoop kids"
    md = youtube_comment_digest(song, artist, model="gpt-5", max_comments=25,
                                save_dir=base_dir, save_files=True)