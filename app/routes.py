from . import app
from .playlist_gpt import vibe_extraction_schema, playlist_schema, build_playlist_generation_prompt, call_gpt_and_verify
from .prompts import VIBE_EXTRACTION_PROMPT, PLAYLIST_GENERATION_PROMPT
from .spotify import create_sp_oauth, create_sp_oauth_clientcredentials, fetch_spotify_data, create_spotify_playlist
from .youtube import youtube_credentials_to_dict, load_and_refresh_credentials, create_youtube_playlist
from .utils import upload_to_s3, generate_presigned_url, resize_image_by_longest_side, download_image
from .models import UserInteraction
from . import db
import uuid, os, json
from flask import request, render_template, redirect, session, Blueprint, jsonify, abort
from spotipy import Spotify
from google_auth_oauthlib.flow import Flow
import traceback
import logging


routes = Blueprint('routes', __name__)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# youtube credentials
with open("google_client_secret.txt", "r") as f:
    youtube_client_config = json.load(f)
if os.getenv("FLASK_DEBUG") == "1":
    youtube_redirect_uri = "http://127.0.0.1:8888/youtube-oauth2-callback"
else:
    youtube_redirect_uri = "https://vibeai-poz2.onrender.com/youtube-oauth2-callback"
youtube_scopes = ["https://www.googleapis.com/auth/youtube.force-ssl"]


@routes.route("/")
def index():
    return render_template("index.html")


@routes.route("/playlist", methods=["GET", "POST"])
def make_playlist():
    
    if request.method == 'POST':
        file = request.files.get("image")
        if not file:
            return render_template("playlist.html", error="No image uploaded")
        
        try:
            # vibe extraction
            resized_file = resize_image_by_longest_side(file,512)
            vibe_extraction = call_gpt_and_verify(VIBE_EXTRACTION_PROMPT, vibe_extraction_schema, file_obj=resized_file)

            # playlist generation
            playlist_prompt = build_playlist_generation_prompt(vibe_extraction, PLAYLIST_GENERATION_PROMPT)
            playlist_data = call_gpt_and_verify(playlist_prompt, playlist_schema)
            
            auth_manager = create_sp_oauth_clientcredentials()
            sp = Spotify(auth_manager=auth_manager)
            fetched_tracks = fetch_spotify_data(sp, playlist_data['tracks'])
            playlist_data['tracks'] = fetched_tracks

            # upload to db
            resized_file.seek(0)
            filename = upload_to_s3(resized_file)
            
            session["last_playlist"] = {
                "vibe_extraction": vibe_extraction,
                "playlist_data": playlist_data,
                "filename": filename
            }
            return redirect("/playlist")
        
        except Exception as e:
            logger.error("Exception during playlist generation: {e}\n" + traceback.format_exc())
            return render_template("playlist.html", error="something went wrong")
    
    if request.method == 'GET':
        last = session.get("last_playlist", None)
        if last:
            image_url = generate_presigned_url(last["filename"])
            return render_template("playlist.html", vibe_extraction = last['vibe_extraction'], playlist=last["playlist_data"], image_url=image_url)
        return render_template("playlist.html")


@routes.route("/save-tracks", methods = ['POST'])
def save_tracks():
    data = request.get_json()

    # 1. Validate session data
    last = session.get("last_playlist", None)
    if not last:
        logger.warning("No playlist in session")
        return jsonify({"success": False, "error": "No playlist in session"}), 400

    # 2. Update tracks from frontend
    if not data or "tracks" not in data:
        logger.warning("Missing tracks")
        return jsonify({"success": False, "error": "Missing tracks"}), 400
    last["playlist_data"]["tracks"] = data["tracks"]
    session.modified = True

    # 3. upload to db
    try:
        if "user_id" not in session:
            session["user_id"] = str(uuid.uuid4())
        last["playlist_id"] = str(uuid.uuid4())
        session.modified = True

        log = UserInteraction(
            session_id = session["user_id"],
            playlist_id = last["playlist_id"],
            image_filename = last.get('filename'),
            playlist_json = json.dumps(last.get('playlist_data'))
        )
        db.session.add(log)
        db.session.commit()
    except:
        logger.warning("Failed to db upload")
        return jsonify({"success": False, "error": "Failed in db upload"}), 400

    return jsonify({"success": True, "pl_id": last["playlist_id"]})


@routes.route("/spotify-login")
def spotify_login():
    last = session.pop("last_playlist", None)
    if not last:
        pl_id = request.args.get("pl_id")
        if not pl_id:
            return "no playlist in session"
        
    pl_id = request.args.get("pl_id") or last.get("playlist_id")
    if not pl_id:
        return "no playlist id in session"
    
    sp_oauth = create_sp_oauth(session)
    token_info = sp_oauth.validate_token(sp_oauth.cache_handler.get_cached_token())

    if token_info is not None:
        session["token_info"] = token_info
        interaction = UserInteraction.query.filter_by(playlist_id=pl_id).first()
        interaction.spotify_token_info = json.dumps(token_info)
        db.session.commit()
        return redirect(f"/spotify-finalize-playlist?pl_id={pl_id}")
    
    else:
        spotify_auth_url = sp_oauth.get_authorize_url(state=pl_id)
        return redirect(spotify_auth_url)


@routes.route("/callback")
def spotify_callback():
    code = request.args.get('code')
    pl_id = request.args.get('state')

    sp_oauth = create_sp_oauth(session)
    token_info = sp_oauth.get_access_token(code=code)

    if token_info:
        session["token_info"] = token_info
        interaction = UserInteraction.query.filter_by(playlist_id=pl_id).first()
        interaction.spotify_token_info = json.dumps(token_info)
        db.session.commit()
    
    else:
        return "something went wrong during the spotify authentification"

    return redirect(f"/spotify-finalize-playlist?pl_id={pl_id}")


@routes.route("/spotify-finalize-playlist", methods=["GET"])
def spotify_finalize_playlist():
    pl_id = request.args.get("pl_id")
    if not pl_id:
        return render_template("finalize_sp.html", error = "Missing pl_id parameter")

    # 1. Get playlist data from db
    interaction = UserInteraction.query.filter_by(playlist_id=pl_id).first()
    if not interaction:
        return render_template("finalize_sp.html", error = "Playlist not found")
    
    if interaction.playlist_url is not None:   # refresh or re-visit logic
        return render_template("finalize_sp.html", pl_url = interaction.playlist_url)

    # 2. Spotify authentification
    sp_oauth = create_sp_oauth()
    token_info = sp_oauth.validate_token(json.loads(interaction.spotify_token_info))
    if not token_info:
        return render_template("finalize_sp.html", error = "Spotify token not found")
    interaction.spotify_token_info = json.dumps(token_info)    # in case token refreshed
    db.session.commit()

    # 3. Create playlist
    sp = Spotify(auth=token_info["access_token"])
    image_url = generate_presigned_url(interaction.image_filename)
    base64_image = download_image(image_url)
    playlist_data = json.loads(interaction.playlist_json)
    playlist_url = create_spotify_playlist(sp, playlist_data, base64_image)

    interaction.playlist_url = playlist_url
    db.session.commit()

    return render_template("finalize_sp.html", pl_url = playlist_url)


@routes.route("/youtube-login")
def youtube_login():
    last = session.pop("last_playlist", None)
    if not last:
        pl_id = request.args.get("pl_id")
        if not pl_id:
            return "no playlist in session"
        
    pl_id = request.args.get("pl_id") or last.get("playlist_id")
    if not pl_id:
        return "no playlist id in session"
    
    yt_credentials = session.get("youtube_credentials")
    interaction = UserInteraction.query.filter_by(playlist_id=pl_id).first()

    if yt_credentials is not None:
        load_and_refresh_credentials(session, pl_id)
        interaction.youtube_credentials = json.dumps(session["youtube_credentials"])
        db.session.commit()
        return redirect(f"/youtube-finalize-playlist?pl_id={pl_id}")
    
    else:
        oauth_state = str(uuid.uuid4())
        interaction.oauth_state = oauth_state
        db.session.commit()
        flow = Flow.from_client_config(
            youtube_client_config,
            scopes=youtube_scopes,
            redirect_uri=youtube_redirect_uri,
            state=oauth_state
        )

        youtube_auth_url, _ = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'
        )

        return redirect(youtube_auth_url)


@routes.route("/youtube-oauth2-callback")
def youtube_callback():
    oauth_state = request.args.get('state')
    if not oauth_state:
        return "Missing state", 400
    
    interaction = UserInteraction.query.filter_by(oauth_state=oauth_state).first()
    if interaction is None:
        return "Internal service error: failed to fetch playlist data"

    try:
        flow = Flow.from_client_config(
            youtube_client_config,
            scopes=youtube_scopes,
            state=oauth_state,
            redirect_uri=youtube_redirect_uri
        )

        flow.fetch_token(authorization_response=request.url)
        credentials = flow.credentials

    except Exception as e:
        return "Failed to exchange code for tokens", 400
    
    session["youtube_credentials"] = youtube_credentials_to_dict(credentials)
    interaction.youtube_credentials = json.dumps(session["youtube_credentials"])
    db.session.commit()
    return redirect(f"/youtube-finalize-playlist?pl_id={interaction.playlist_id}")



@routes.route("/youtube-finalize-playlist")
def youtube_finalize_playlist():
    pl_id = request.args.get("pl_id")
    if not pl_id:
        return render_template("finalize_yt.html", error = "Missing pl_id parameter")

    # 1. Get playlist data from db
    interaction = UserInteraction.query.filter_by(playlist_id=pl_id).first()
    if not interaction:
        return render_template("finalize_yt.html", error = "Playlist not found")
    
    if interaction.playlist_url is not None:   # refresh or re-visit logic
        return render_template("finalize_yt.html", pl_url = interaction.playlist_url)
    
    # 2. Youtube authentification
    yt_credentials = session.get("youtube_credentials")
    if not yt_credentials:
        try:
            session["youtube_credentials"] = interaction.youtube_credentials
        except:
            render_template("finalize_yt.html", error = "Youtube credentials not found")
    
    creds = load_and_refresh_credentials(session, pl_id)
    interaction.youtube_credentials = json.dumps(session["youtube_credentials"])
    db.session.commit()

    # 3. Create playlist
    playlist_data = json.loads(interaction.playlist_json)
    playlist_url = create_youtube_playlist(creds, playlist_data)
    interaction.playlist_url = playlist_url
    db.session.commit()
    
    return render_template("finalize_yt.html", pl_url = playlist_url)



@routes.route('/logout')
def logout():
    token_info = session.get("token_info")
    session.clear()
    if token_info:
        session["token_info"] = token_info
    return redirect('/playlist')


@routes.route('/privacy-policy')
def privacy():
    return render_template("privacy_policy.html")

@routes.route('/terms-of-service')
def termsofservice():
    return render_template('terms_of_service.html')