from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from sahi import AutoDetectionModel

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class DetectionModelManager:
    def __init__(self):
        self.models = self.load_models()

    def load_models(self):
        models = {
            # 'yolov8n': AutoDetectionModel.from_pretrained("yolov8", model_path='models/best_yolov8n.pt'),
            # 'yolov8s': AutoDetectionModel.from_pretrained("yolov8", model_path='models/best_yolov8s.pt'),
            # 'yolov8m': AutoDetectionModel.from_pretrained("yolov8", model_path='models/pt/best_yolov8m.pt', device="cuda:0"),
            'yolov8l': AutoDetectionModel.from_pretrained("yolov8", model_path='models/pt/best_yolov8l.pt', device="cuda:0"),
            'yolov8x': AutoDetectionModel.from_pretrained("yolov8", model_path='models/pt/best_yolov8x.pt', device="cuda:0")
        }
        return models

    def get_model(self, model_key):
        return self.models.get(model_key)