from __future__ import annotations
import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Iterator, Tuple, List, Optional
from spotipy.exceptions import SpotifyException

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

ID_RE = re.compile(r"(playlist[:/])?([0-9A-Za-z]{22})")

def extract_playlist_id(s: str) -> str:
    """
    Accepts playlist URL/URI/ID and returns the bare 22-char playlist ID.
    Examples:
      - https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M
      - spotify:playlist:37i9dQZF1DXcBWIGoYBM5M
      - 37i9dQZF1DXcBWIGoYBM5M
    """
    s = s.strip()
    m = ID_RE.search(s)
    if m:
        return m.group(2)
    # fallback: if it's already 22 chars, assume ID
    if len(s) == 22 and s.isalnum():
        return s
    raise ValueError(f"Could not parse playlist ID from: {s}")

def get_sp_client(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None
) -> spotipy.Spotify:
    """
    Returns an authenticated Spotipy client using Client Credentials flow.
    Suitable for reading public playlists (no user login/scopes).
    """
    client_id = client_id or os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = client_secret or os.getenv("SPOTIPY_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise EnvironmentError(
            "Missing SPOTIPY_CLIENT_ID / SPOTIPY_CLIENT_SECRET. "
            "Set env vars or pass to get_sp_client()."
        )
    auth_mgr = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    return spotipy.Spotify(auth_manager=auth_mgr, requests_timeout=30, retries=5)

def iter_playlist_tracks(sp, playlist_id):
    # First check basic metadata (may 404 for Spotify-owned playlists)
    try:
        meta = sp.playlist(playlist_id)  # may 404 for editorial/algorithmic
    except SpotifyException as e:
        if e.http_status == 404:
            raise RuntimeError(
                "Playlist not accessible via API (often Spotify-owned/editorial)."
            ) from e
        raise

    owner = (meta.get("owner") or {}).get("id")
    if owner == "spotify":
        raise RuntimeError(
            "Spotify-owned/editorial playlist: tracks endpoint often returns 404."
        )

    limit, offset = 100, 0
    while True:
        page = sp.playlist_items(
            playlist_id, limit=limit, offset=offset,
            additional_types=("track",)
        )
        items = page.get("items", [])
        if not items:
            break
        for it in items:
            yield it
        offset += len(items)
        if not page.get("next"):
            break

def extract_song_artist(item: dict) -> Optional[Tuple[str, str]]:
    """
    Converts a playlist item into (song, artist). Returns None for non-track items.
    - Skips 'local' or missing track entries
    - Joins multiple artists with ', '
    """
    track = item.get("track")
    if not track:
        return None
    if track.get("is_local"):
        return None
    if track.get("type") != "track":
        return None
    name = (track.get("name") or "").strip()
    artists = track.get("artists") or []
    artist_names = ", ".join(a.get("name", "").strip() for a in artists if a.get("name"))
    if not name or not artist_names:
        return None
    return name, artist_names

def playlist_to_jsonl(
    playlist: str,
    out_path: str | Path,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    dedupe: bool = True
) -> List[Tuple[str, str]]:
    """
    Fetch a public Spotify playlist and write JSONL with {"song","artist"} per line.

    Args:
      playlist: URL/URI/ID for the playlist
      out_path: target .jsonl path
      client_id/client_secret: optional overrides for Spotify creds
      dedupe: if True, remove duplicates by (song, artist)

    Returns:
      The list of (song, artist) pairs (post-deduping if enabled)
    """
    playlist_id = extract_playlist_id(playlist)
    sp = get_sp_client(client_id, client_secret)

    pairs: List[Tuple[str, str]] = []
    seen = set()

    for item in iter_playlist_tracks(sp, playlist_id):
        pair = extract_song_artist(item)
        if not pair:
            continue
        if dedupe:
            if pair in seen:
                continue
            seen.add(pair)
        pairs.append(pair)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for song, artist in pairs:
            f.write(json.dumps({"song": song, "artist": artist}, ensure_ascii=False) + "\n")

    return pairs

if __name__ == "__main__":
    
    playlist = "https://open.spotify.com/playlist/5J7NmmIVCDxMSNDElDonXt"
    out = Path(r"C:\Projects\VibeAI\pipeline\music-rag\playlists\songstotestspeakers.jsonl")
    pairs = playlist_to_jsonl(playlist, out)
    print(f"Wrote {len(pairs)} lines to {out}")