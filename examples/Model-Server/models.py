from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from sahi import AutoDetectionModel
from typing import Dict, Optional

db = SQLAlchemy()

class User(db.Model):
    """
    User model for storing user related data.

    Attributes:
        id (int): The unique identifier for the user, serves as the primary key.
        username (str): The user's username, must be unique.
        password_hash (str): The hashed version of the user's password.
    """
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)

    def set_password(self, password: str) -> None:
        """
        Generates a hash for the given password and stores it.

        Args:
            password (str): The plain text password.
        """
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """
        Checks the given password against the stored hash.

        Args:
            password (str): The plain text password to verify.

        Returns:
            bool: True if the password matches, otherwise False.
        """
        return check_password_hash(self.password_hash, password)

class DetectionModelManager:
    """
    Manages the loading and accessing of object detection models.

    Attributes:
        models (Dict[str, AutoDetectionModel]): Dictionary of loaded models.
    """
    def __init__(self):
        self.models: Dict[str, AutoDetectionModel] = self.load_models()

    def load_models(self) -> Dict[str, AutoDetectionModel]:
        """
        Loads and returns a dictionary of SAHI's AutoDetectionModels.

        Returns:
            A dictionary of model names associated with their respective AutoDetectionModel instances.
        """
        models = {
            # Uncomment below lines to enable more models as needed.
            # 'yolov8n': AutoDetectionModel.from_pretrained("yolov8", model_path='models/best_yolov8n.pt'),
            # 'yolov8s': AutoDetectionModel.from_pretrained("yolov8", model_path='models/best_yolov8s.pt'),
            # 'yolov8m': AutoDetectionModel.from_pretrained("yolov8", model_path='models/pt/best_yolov8m.pt', device="cuda:0"),
            'yolov8l': AutoDetectionModel.from_pretrained("yolov8", model_path='models/pt/best_yolov8l.pt', device="cuda:0"),
            'yolov8x': AutoDetectionModel.from_pretrained("yolov8", model_path='models/pt/best_yolov8x.pt', device="cuda:0")
        }
        return models

    def get_model(self, model_key: str) -> Optional[AutoDetectionModel]:
        """
        Retrieves a model by its key.

        Args:
            model_key (str): The key associated with the model to retrieve.

        Returns:
            The AutoDetectionModel if found, otherwise None.
        """
        return self.models.get(model_key)