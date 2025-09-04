from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError
from typing import List, Dict, Optional
from schema import Comment



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
        "getcomments": True,
        "extract_flat": True,
        "writesubtitles": False,
        "writeinfojson": False,
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
