from flask import Flask
from flask_jwt_extended import JWTManager
from .config import Config
from .models import db, User
from .auth import auth_blueprint
from .detection import detection_blueprint
from dotenv import load_dotenv
import os
load_dotenv()

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your_fallback_secret_key')
app.config.from_object(Config)

jwt = JWTManager(app)

db.init_app(app)
app.register_blueprint(auth_blueprint)
app.register_blueprint(detection_blueprint)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000)
