from . import app
from .playlist_gpt import vibe_extraction_schema, playlist_schema, build_playlist_generation_prompt, call_gpt_and_verify
from .prompts import VIBE_EXTRACTION_PROMPT, PLAYLIST_GENERATION_PROMPT
from .spotify import create_sp_oauth, create_sp_oauth_clientcredentials, fetch_spotify_data, create_spotify_playlist
from .utils import upload_to_s3, generate_presigned_url, resize_image_by_longest_side, download_image
from .models import UserInteraction
from . import db
import uuid, os, json
from flask import request, render_template, redirect, session, Blueprint, jsonify, abort
from spotipy import Spotify
import traceback
import logging


routes = Blueprint('routes', __name__)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 


@routes.route("/")
def index():
    return render_template("index.html")


@routes.route("/spotify-login")
def spotify_login():
    sp_oauth = create_sp_oauth(session)
    token_info = sp_oauth.validate_token(sp_oauth.cache_handler.get_cached_token())

    if token_info is not None:
        return redirect("/spotify-finalize-playlist")
    else:
        auth_url = sp_oauth.get_authorize_url()
        return redirect(auth_url)


@routes.route("/callback")
def callback():
    code = request.args.get('code')
    sp_oauth = create_sp_oauth(session)
    token_info = sp_oauth.get_access_token(code=code)

    if token_info:
        session["token_info"] = token_info

    return redirect("/spotify-finalize-playlist")


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

            if "user_id" not in session:
                session["user_id"] = str(uuid.uuid4())
            log = UserInteraction(
                session_id = session["user_id"],
                image_filename = filename,
                playlist_json = json.dumps(playlist_data)
            )
            db.session.add(log)
            db.session.commit()
            
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

    return jsonify({"success": True})


@routes.route("/spotify-finalize-playlist", methods=["GET"])
def spotify_finalize_playlist():

    # 1. Validate session data
    last = session.get("last_playlist", None)
    if not last:
        last_playlist_url = session.get("last_playlist_url", None)
        if not last_playlist_url:
            return render_template("finalize_sp.html", error = "No playlist in session. Please try again.")
        return render_template("finalize_sp.html", pl_url = last_playlist_url)

    # 2. Spotify authentification
    sp_oauth = create_sp_oauth(session)
    token_info = sp_oauth.validate_token(sp_oauth.cache_handler.get_cached_token())
    if not token_info:
        return redirect("/spotify-login")
    session["token_info"] = token_info

    # 3. Create playlist
    last = session.pop("last_playlist", None)
    sp = Spotify(auth=token_info["access_token"])
    image_url = generate_presigned_url(last["filename"])
    base64_image = download_image(image_url)
    playlist_url = create_spotify_playlist(sp, last["playlist_data"], base64_image)

    session["last_playlist_url"] = playlist_url
    session.modified = True

    return render_template("finalize_sp.html", pl_url = playlist_url)


@routes.route('/logout')
def logout():
    token_info = session.get("token_info")
    session.clear()
    if token_info:
        session["token_info"] = token_info
    return redirect('/playlist')