from flask import Flask
from flask_jwt_extended import JWTManager
from .config import Config
from .models import db, User
from .auth import auth_blueprint
from .detection import detection_blueprint
from .security import update_secret_key
from apscheduler.schedulers.background import BackgroundScheduler
import secrets
import atexit

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = secrets.token_urlsafe(16)
app.config.from_object(Config)

jwt = JWTManager(app)

db.init_app(app)
app.register_blueprint(auth_blueprint)
app.register_blueprint(detection_blueprint)

scheduler = BackgroundScheduler()
scheduler.add_job(func=lambda: update_secret_key(app), trigger="interval", days=30)
scheduler.start()

# Shut down the scheduler when exiting the app
atexit.register(lambda: scheduler.shutdown())

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000)
