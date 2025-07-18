import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_session import Session

if os.getenv("FLASK_DEBUG") == "1":
    from dotenv import load_dotenv
    load_dotenv()

base_dir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__,
            template_folder=os.path.join(base_dir, "templates"),
            static_folder=os.path.join(base_dir, "static"))

app.secret_key = os.getenv("SECRET_KEY")

if os.getenv("FLASK_DEBUG") == "1":
    app.config["SESSION_COOKIE_SECURE"] = False
else:
    app.config["SESSION_COOKIE_SECURE"] = True

db_path = os.path.join(app.instance_path, "site.db")
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{db_path}"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config['SESSION_TYPE'] = 'filesystem'

db=SQLAlchemy(app)
with app.app_context():
    db.create_all()

Session(app)

from .routes import routes
app.register_blueprint(routes)