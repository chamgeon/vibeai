from pathlib import Path
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def _safe_name(s: str) -> str:
    return "".join(ch for ch in s if ch.isalnum() or ch in " -_().").strip()

def _song_dir(base_dir: Path, song: str, artist: str) -> Path:
    return base_dir / "songs" / f"{_safe_name(artist)} - {_safe_name(song)}"

# --- CONFIG ---
BASE_DIR = Path(r"C:\Projects\VibeAI\pipeline\music-rag")
INPUT_JSONL = Path(r"C:\Projects\VibeAI\pipeline\music-rag\playlists\wineanddine.jsonl")  # each line: {"song": "...", "artist": "..."}
MODEL = "gpt-5"
MAX_COMMENTS = 25
SAVE_FILES = True
CONCURRENCY = 3
SLEEP_BETWEEN_TASKS = 0
OUTPUT_JSONL = BASE_DIR / "processed_songs.jsonl"
# ---------------

def already_done(song: str, artist: str) -> bool:
    song_dir = _song_dir(BASE_DIR, song, artist)
    return song_dir.exists()

def process_row(song: str, artist: str) -> dict:
    from data_digestion import youtube_comment_digest
    try:
        if not already_done(song, artist):
            _ = youtube_comment_digest(
                song,
                artist,
                model=MODEL,
                max_comments=MAX_COMMENTS,
                save_dir=str(BASE_DIR),
                save_files=SAVE_FILES,
            )
            if SLEEP_BETWEEN_TASKS:
                time.sleep(SLEEP_BETWEEN_TASKS)
        return {"song": song, "artist": artist}
    except Exception as e:
        print(f"[error] {artist} - {song}: {e}")
        return {"song": song, "artist": artist}

def read_rows(jsonl_path: Path):
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                song = (obj.get("song") or "").strip()
                artist = (obj.get("artist") or "").strip()
                if song and artist:
                    yield song, artist
            except json.JSONDecodeError:
                print(f"[warn] Skipping invalid line: {line}")

def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = [ex.submit(process_row, song, artist) for song, artist in read_rows(INPUT_JSONL)]
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            print(f"[done] {res['artist']} - {res['song']}")

    # Write final JSONL with only song/artist
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(results)} items to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()