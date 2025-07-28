from flask import redirect
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from youtubesearchpython import VideosSearch




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
            video_info = youtube.videos().list(
                part="snippet",
                id=video_id
            ).execute()

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
            response_2 = youtube.playlistItems().insert(
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
            ).execute()
        
        else:
            print(f"cannot find youtube video for track: {track['song']}, artist: {track['artist']}")
    
    playlist_url = f"https://www.youtube.com/playlist?list={playlist_id}"

    return playlist_url