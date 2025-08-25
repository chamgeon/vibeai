from flask import redirect, jsonify
from datetime import datetime, timedelta, timezone
import time
import os
import pytz
from youtubesearchpython import VideosSearch

from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.cloud import monitoring_v3
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError

PROJECT_ID = os.environ.get("GCP_PROJECT")

def last_pt_midnight_utc_now():
    PT = pytz.timezone("America/Los_Angeles")
    now_pt = datetime.now(PT)
    start_pt = now_pt.replace(hour=0, minute=0, second=0, microsecond=0)
    return start_pt.astimezone(pytz.utc), now_pt.astimezone(pytz.utc)

def get_api_quota_usage():

    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{PROJECT_ID}"

    # Time window: last PT midnight → now (UTC)
    start_utc, end_utc = last_pt_midnight_utc_now()
    interval = monitoring_v3.TimeInterval(
        start_time={"seconds": int(start_utc.timestamp())},
        end_time={"seconds": int(end_utc.timestamp())},
    )

    # Metric: per-API request count for consumed APIs
    # Group by API method so we can translate to quota units.
    filter_ = (
        'metric.type="serviceruntime.googleapis.com/api/request_count" '
        'AND resource.type="consumed_api" '
        'AND resource.label."service"="youtube.googleapis.com"'
    )


    # Aggregate to a single sum in the window, grouped by method
    aggregation = monitoring_v3.Aggregation(
        alignment_period={"seconds": 60},  # align per minute
        per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_DELTA,  # sum deltas
        cross_series_reducer=monitoring_v3.Aggregation.Reducer.REDUCE_SUM,
        group_by_fields=['resource.label."method"']
    )

    request = monitoring_v3.ListTimeSeriesRequest(
        name=project_name,
        filter=filter_,
        interval=interval,
        view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
        aggregation=aggregation,
    )

    results = client.list_time_series(request=request)

    # Minimal quota cost map (add the methods you actually use)
    QUOTA_COST = {
        "youtube.api.v3.V3DataVideoService.List": 1,
        "youtube.api.v3.V3DataPlaylistItemService.Insert": 50,
        "youtube.api.v3.V3DataPlaylistService.Insert": 50,
    }

    per_method_calls = {}
    for ts in results:
        method = ts.resource.labels.get("method", "unknown")
        # Sum points in this series
        calls = sum(p.value.int64_value for p in (pt for pt in ts.points))
        per_method_calls[method] = per_method_calls.get(method, 0) + calls

    total_units = 0
    per_method_units = {}
    for method, calls in per_method_calls.items():
        cost = QUOTA_COST.get(method, 1)  # default to 1 if unknown
        units = calls * cost
        per_method_units[method] = {"calls": calls, "units": units, "cost_per_call": cost}
        total_units += units

    return total_units

def get_daily_limit() -> float:
    return float(os.getenv("YT_DAILY_QUOTA") or 0)


def youtube_credentials_to_dict(creds):
    return {
        'token': creds.token,
        'refresh_token': creds.refresh_token,
        'token_uri': creds.token_uri,
        'client_id': creds.client_id,
        'client_secret': creds.client_secret,
        'scopes': creds.scopes
    }

def load_and_refresh_credentials(session, pl_id):
    """
    load and refresh youtube credentials. session is Flask session, pl_id it playlist id
    """

    # reached only when youtube_credentials is in session
    creds = Credentials(**session["youtube_credentials"])
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except RefreshError:
            session.pop("youtube_credentials", None)
            return redirect(f"/youtube-login?pl_id={pl_id}")
        session["youtube_credentials"] = youtube_credentials_to_dict(creds)
    return creds


def get_video_id(track, creds):
    """
    Searches YouTube for a song and returns the first video ID with category 'Music'.

    Args:
        track (dict): A dictionary with 'song' and 'artist'.
        creds (google.oauth2.credentials.Credentials): Authenticated credentials.

    Returns:
        str or None: YouTube video ID or None if not found.
    """
    query = f"{track['song']} by {track['artist']}"
    videos_search = VideosSearch(query, limit=5)  # Get top 5 candidates
    search_results = videos_search.result().get("result", [])

    youtube = build("youtube", "v3", credentials=creds)

    for item in search_results:
        video_id = item["id"]
        try:
            request = youtube.videos().list(
                part="snippet",
                id=video_id
            )
            video_info = execute_with_retry(request)

            items = video_info.get("items", [])
            if not items:
                continue

            category_id = items[0]["snippet"].get("categoryId")
            if category_id == "10":  # 10 = Music
                return video_id
        except Exception as e:
            print(f"Error checking video {video_id}: {e}")

    return None


def create_youtube_playlist(creds, playlist_data):
    """
    Creates a Youtube playlist based on a list of dictionaries containing 'song' and 'artist' keys.
    
    Args:
        creds (Credentials): youtube credentials class
        playlist_data (dict): playlist data that has name, description, and tracks

    Returns:
        str: Youtube playlist URL
    """
    pl_name, pl_description, tracks = playlist_data['name'], playlist_data['description'], playlist_data['tracks']
    youtube = build("youtube", "v3", credentials=creds)

    request = youtube.playlists().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": f"{pl_name}",
                "description": f"{pl_description}"
            },
            "status": {
                "privacyStatus": "public"
            }
        }
    )
    response = request.execute()
    playlist_id = response['id']

    for track in tracks:
        video_id = get_video_id(track, creds)

        if video_id:
            request = youtube.playlistItems().insert(
                part="snippet",
                body={
                    "snippet": {
                        "playlistId": playlist_id,
                        "resourceId": {
                            "kind": "youtube#video",
                            "videoId": video_id
                        }
                    }
                }
            )
            execute_with_retry(request)

        
        else:
            print(f"cannot find youtube video for track: {track['song']}, artist: {track['artist']}")
    
    playlist_url = f"https://www.youtube.com/playlist?list={playlist_id}"

    return playlist_url


def execute_with_retry(request, max_retries=3):
    for i in range(max_retries):
        try:
            return request.execute()
        except HttpError as e:
            if e.resp.status == 409 and "SERVICE_UNAVAILABLE" in str(e):
                wait = 2 ** i
                print(f"[WARN] YouTube api execution failed. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise