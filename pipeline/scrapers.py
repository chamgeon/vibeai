from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError
from typing import List, Dict, Optional, Tuple
from schema import Comment
from openai import OpenAI
import asyncio
import httpx
from tavily import AsyncTavilyClient
import trafilatura




def youtube_search(query, max_results = 5):
    """
    Search youtube using yt-dlp, with max results number = max_results
    """
    ydl_opts = {
        "quiet": True,
        "extract_flat": True,  # don't download, just metadata
        "skip_download": True,
    }
    search_term = f"ytsearch{max_results}:{query}"
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(search_term, download=False)
    
    if not info.get("entries"):
        return None

    return [
        {
            "id": e.get("id"),
            "title": e.get("title"),
            "duration": e.get("duration"),
            "uploader": e.get("uploader"),
            "url": f"https://www.youtube.com/watch?v={e.get('id')}"
        }
        for e in info.get("entries", [])
    ]


def youtube_get_root_comments(song, artist, video_id, max_comments = 25):
    """
    Get video id and return root comments, and return None if there's no comments.
    """

    comment_opts = {
        "quiet": True,
        "skip_download": True,
        "simulate": True,
        "check_formats": False,
        "ignore_no_formats_error": True,
        "getcomments": True,
        "extractor_args": {
            "youtube": {
                "comment_sort": ["top"],
                "max_comments": [str(max_comments),str(max_comments),"0","0"]
            }
        }
    }

    try:
        with YoutubeDL(comment_opts) as ydl:
            video_info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    except DownloadError as e:
        # comments off, private, or no valid formats
        print(f"[warn] Could not fetch comments for {video_id}: {e}")
        return None
    
    if not video_info:
        return None
    if video_info.get("comment_count", 0) == 0:
        return None
    
    comments = video_info.get("comments",[])
    if not comments:
        return None
    
    comments_list = []
    for c in comments:
        comment_info = {"song":song, "artist":artist, "video_id":video_id, "text":c["text"]}
        comments_list.append(Comment(**comment_info))

    return comments_list


def youtube_comment_scrape(song_name: str, artist_name: str, max_queries: int = 5, max_comments: int = 25) -> List[Comment]:
    
    """
    Get youtube comments of the music (or music video) provided in args.
    args:
        song_name (str): song name to scrape comments from
        artist_name (str): artist for the target song
        max_quesries (int): max queries for searching videos
        max_comments (int): max comment number to fetch

    returns:
        comments (list): list of comments
    """

    query = f"{song_name} by {artist_name}"
    search_results = youtube_search(query, max_queries)
    if not search_results:
        return "no youtube search results"
    
    pending_comments = []

    for item in search_results:
        video_id = item['id']
        comments = youtube_get_root_comments(song_name, artist_name, video_id, max_comments)

        if not comments:
            continue

        if len(comments) >= max_comments:
            return comments[:max_comments]
        
        pending_comments.extend(comments)
        if len(pending_comments) >= max_comments:
            return pending_comments[:max_comments]
        
    return pending_comments



###############################################
# --------- web search functions ------------ #
###############################################





def call_gpt(client: OpenAI, prompt, model = "gpt-5"):

    response = client.responses.create(
            model=model,
            input=prompt,
            reasoning={ "effort": "low"},
            text = {"verbosity": "medium"}
        )

    return response.output_text


async def tavily_search(client: AsyncTavilyClient, song: str, artist: str, max_results: int = 3) -> Dict:
    """
    Search about a song using Tavily. It generates subqueries and return urls of search results.
    
    Input:
        client: tavily client
        song: song name
        artist: artist name
    Output:
        Dictionary with fields "song", "artist", and "urls"
    """
    subqueries = [
        f"{song} by {artist} review",
        f"{song} by {artist} musical style",
        f"{song} by {artist} themes"
    ]

    tasks = [client.search(query=q, search_depth="advanced", max_results=max_results, exclude_domains=["youtube.com", "www.youtube.com", "m.youtube.com"]) for q in subqueries]
    search_results = await asyncio.gather(*tasks)
    
    urls = set()
    for res in search_results:
        if not isinstance(res, dict):
            continue
        for r in res.get("results", []):
            if isinstance(r, dict) and "url" in r:
                urls.add(r["url"])

    return {
        "song": song,
        "artist": artist,
        "urls": list(urls)
    }

async def tavily_search_batch(client: AsyncTavilyClient, pairs: List[Tuple]):
    tasks = [tavily_search(client, song, artist) for song, artist in pairs]
    results = await asyncio.gather(*tasks)   # all songs in parallel
    return results


async def fetch_html(url: str, client: httpx.AsyncClient, timeout: float = 20.0) -> Optional[str]:
    try:
        r = await client.get(url, timeout=timeout, follow_redirects=True, headers={
            "User-Agent": "Mozilla/5.0 (compatible; RAG/1.0)"
        })
        r.raise_for_status()
        return r.text
    except Exception:
        return None
    

def extract_main_text(html: str):
    # Options: include metadata, comments, best-effort recall
    return trafilatura.bare_extraction(
        html,
        include_comments=True,
        include_tables=False,
        favor_recall=True,      # try harder to get content if structure is odd
        with_metadata=True,     # returns JSON-ish dict when as_dict=True (below)
    )


async def extract_urls(urls):
    out = []
    async with httpx.AsyncClient() as client:
        htmls = await asyncio.gather(*(fetch_html(u, client) for u in urls))
    for url, html in zip(urls, htmls):
        if not html:
            out.append({"url": url, "ok": False, "reason": "fetch-failed"})
            continue
        data = extract_main_text(html).as_dict()
        if not data or not (data.get("raw_text") or "").strip():
            out.append({"url": url, "ok": False, "reason": "no-main-content"})
            continue
        # Light cleanup & minimal schema for RAG
        raw_text = " ".join(data["raw_text"].split())
        out.append({
            "url": url,
            "ok": True,
            "title": data.get("title"),
            "author": data.get("author"),
            "date": data.get("date"),
            "text": raw_text
        })
    return out


def web_search_scrape(client: AsyncTavilyClient, song: str, artist: str, pages_per_query: int = 3) -> List[Dict]:
    """
    Perform web search for a song-artist pair. Return scraped data into list of dictionary

    Input:
        client: Async Tavily Client
        song: target song name
        artist: target artist name
        pages_per_query: web search result number for each query, the output result will be *3, because there are 3 queries (review, music style, theme)
    
    Output:
        list of dict that has web search result
    """

    tavily_result = asyncio.run(tavily_search(client, song, artist, pages_per_query))
    urls = tavily_result["urls"]
    
    res = asyncio.run(extract_urls(urls))
    return res
    




if __name__ == "__main__":
    from dotenv import load_dotenv
    from pathlib import Path
    import os
    import time

    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)

    
    TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
    tavily = AsyncTavilyClient(api_key=TAVILY_API_KEY)

    song = "Goons"
    artist = "Hanumankind"

    tavily_result = asyncio.run(tavily_search(tavily, song, artist))
    urls = tavily_result["urls"]
    
    res = asyncio.run(extract_urls(urls))

    for i in range(len(res)):
        print(res[i])
        print("\n\n===================\n\n")
    
    


