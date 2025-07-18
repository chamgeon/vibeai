from . import app
from .playlist_gpt import vibe_extraction_schema, playlist_schema, build_playlist_generation_prompt, call_gpt_and_verify
from .prompts import VIBE_EXTRACTION_PROMPT, PLAYLIST_GENERATION_PROMPT
from .spotify import create_sp_oauth, fetch_spotify_data, create_spotify_playlist
from .utils import upload_to_s3, generate_presigned_url, resize_image_by_longest_side, download_image
from .models import UserInteraction
from . import db
import uuid, os, json
from flask import request, render_template, redirect, session, Blueprint, jsonify
from spotipy import Spotify


routes = Blueprint('routes', __name__)

@routes.route("/")
def index():
    return render_template("index.html")


@routes.route("/login")
def login():
    sp_oauth = create_sp_oauth(session)
    token_info = sp_oauth.validate_token(sp_oauth.cache_handler.get_cached_token())

    if token_info is not None:
        return redirect("/playlist")
    else:
        auth_url = sp_oauth.get_authorize_url()
        return redirect(auth_url)


@routes.route("/callback")
def callback():
    code = request.args.get('code')
    sp_oauth = create_sp_oauth(session)
    sp_oauth.get_access_token(code=code)

    return redirect('/playlist')


@routes.route("/playlist", methods=["GET", "POST"])
def make_playlist():

    # Check Spotify token
    sp_oauth = create_sp_oauth(session)
    token_info = sp_oauth.validate_token(sp_oauth.cache_handler.get_cached_token())
    if not token_info:
        return redirect("/login")
    session["token_info"] = token_info
    
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
            
            sp = Spotify(auth=token_info["access_token"])
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
            return render_template("playlist.html", error=str(e))
    
    if request.method == 'GET':
        last = session.get("last_playlist", None)
        if last:
            image_url = generate_presigned_url(last["filename"])
            return render_template("playlist.html", vibe_extraction = last['vibe_extraction'], playlist=last["playlist_data"], image_url=image_url)
        return render_template("playlist.html")


@routes.route("/finalize_playlist", methods=["POST"])
def finalize_playlist():
    # 1. Auth check
    sp_oauth = create_sp_oauth(session)
    token_info = sp_oauth.validate_token(sp_oauth.cache_handler.get_cached_token())
    if not token_info:
        return jsonify({"error": "Unauthorized"}), 401
    session["token_info"] = token_info

    # 2. Validate session data
    last = session.pop("last_playlist", None)
    if not last:
        return jsonify({"error": "No playlist in session"}), 400

    # 3. Update tracks from frontend
    data = request.get_json()
    if not data or "tracks" not in data:
        return jsonify({"error": "Missing tracks"}), 400
    last["playlist_data"]["tracks"] = data["tracks"]
    session.modified = True

    # 4. Create Spotify playlist
    sp = Spotify(auth=token_info["access_token"])
    image_url = generate_presigned_url(last["filename"])
    base64_image = download_image(image_url)
    playlist_url = create_spotify_playlist(sp, last["playlist_data"], base64_image)

    return jsonify({"playlist_url": playlist_url})