from . import db
from datetime import datetime, timezone

class UserInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(64), nullable=False)
    image_filename = db.Column(db.String(200), nullable=False)
    caption = db.Column(db.Text)
    playlist_json = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now(timezone.utc))