from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
from spotipy.cache_handler import FlaskSessionCacheHandler
import os


def create_sp_oauth(flask_session = None):
    if flask_session:
        return SpotifyOAuth(scope="playlist-modify-public ugc-image-upload",
                        cache_handler=FlaskSessionCacheHandler(flask_session))
    else:
        return SpotifyOAuth(scope="playlist-modify-public ugc-image-upload")

def create_sp_oauth_clientcredentials():
    return SpotifyClientCredentials()

def fetch_spotify_data(sp:Spotify, tracks):
    """
    Fetch spotify data from dictionary. It calls spotify api to fetch song name, artist, cover art, url.

    Args:
        sp (Spotify): agent class in spotipy
        tracks (list): List of dicts with field 'song', 'artist', and 'vibe'
    
    Returns:
        modified tracks list with fields 'song', 'artist', 'uri', 'cover_url', and 'vibe'
    """

    enriched_tracks = []

    for track in tracks:
        query = f"{track['song']} {track['artist']}"
        try:
            result = sp.search(q=query, type='track', limit=1)
            items = result['tracks']['items']
            if items:
                t = items[0]
                enriched_tracks.append({
                    "song": t['name'],
                    "artist": ", ".join([a['name'] for a in t['artists']]),
                    "uri": t['uri'],
                    "cover_url": t['album']['images'][-1]['url'] if t['album']['images'] else None,
                    "vibe": track['vibe']
                })

        except Exception as e:
            print(f"Error fetching track '{track['song']}' by '{track['artist']}': {e}")

    return enriched_tracks


def create_spotify_playlist(sp:Spotify, playlist_data, cover_image = None):
    """
    Creates a Spotify playlist based on a list of dictionaries containing 'song' and 'artist' keys.
    
    Args:
        sp (Spotify): agent class in spotipy
        playlist_data (dict): playlist data that has name, description, and tracks
        cover_image: base64 image for the playlist cover

    Returns:
        str: Spotify playlist URL
    """

    pl_name, pl_description, tracks = playlist_data['name'], playlist_data['description'], playlist_data['tracks']

    user_id = sp.me()['id']
    playlist = sp.user_playlist_create(user=user_id, name=pl_name, description=pl_description, public=True)
    playlist_id = playlist['id']

    uris = [track['uri'] for track in tracks]
    sp.playlist_add_items(playlist_id, uris)

    # cover image
    if cover_image:
        sp.playlist_upload_cover_image(playlist_id, cover_image)

    return playlist['external_urls']['spotify']