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

app = Flask(__name__)  # Initialise the Flask application

# Securely generate a random JWT secret key using secrets library
app.config['JWT_SECRET_KEY'] = secrets.token_urlsafe(16)

# Load configurations from Config class
app.config.from_object(Config)

# Initialise JWTManager with the Flask app
jwt = JWTManager(app)

# Initialise the database with the Flask app
db.init_app(app)

# Register authentication-related routes
app.register_blueprint(auth_blueprint)

# Register object detection-related routes
app.register_blueprint(detection_blueprint)

# Set up a background scheduler
scheduler = BackgroundScheduler()

# Schedule a job to update the JWT secret key every 30 days
scheduler.add_job(func=lambda: update_secret_key(app), trigger="interval", days=30)

# Start the scheduler
scheduler.start()

# Ensure the scheduler is shut down gracefully upon exiting the application
atexit.register(lambda: scheduler.shutdown())

if __name__ == '__main__':
    # Ensure the database tables are created at launch
    with app.app_context():
        db.create_all()
    
    # Run the Flask application on all available IPs at port 5000
    app.run(host='0.0.0.0', port=5000)