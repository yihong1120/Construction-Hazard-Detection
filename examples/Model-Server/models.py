from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from sahi import AutoDetectionModel
import threading
import time
from pathlib import Path
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
        last_modified_times (Dict[str, float]): Dictionary of last modified times of model files.
    """
    def __init__(self):
        self.base_model_path = Path('models/pt/')
        self.model_names = ['yolov8x', 'yolov8l', 'yolov8m', 'yolov8s', 'yolov8n']
        self.models: Dict[str, AutoDetectionModel] = self.load_all_models()
        self.last_modified_times = self.get_last_modified_times()
        self.model_reload_thread = threading.Thread(target=self.reload_models_every_hour)
        self.model_reload_thread.start()

    def load_single_model(self, model_name: str) -> AutoDetectionModel:
        """
        Loads and returns a SAHI's AutoDetectionModel.

        Returns:
            A AutoDetectionModel instance.
        """
        return AutoDetectionModel.from_pretrained("yolov8", model_path=str(self.base_model_path / f'best_{model_name}.pt'), device="cuda:0")

    def load_all_models(self) -> Dict[str, AutoDetectionModel]:
        """
        Loads and returns a dictionary of SAHI's AutoDetectionModels.

        Returns:
            A dictionary of model names associated with their respective AutoDetectionModel instances.
        """
        models = {name: self.load_single_model(name) for name in self.model_names}
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

    def get_last_modified_time(self, model_name: str) -> float:
        """
        Retrieves the last modified time of the model file.

        Returns:
            The last modified time of the model file.
        """
        return (self.base_model_path / f'best_{model_name}.pt').stat().st_mtime

    def get_last_modified_times(self) -> Dict[str, float]:
        """
        Retrieves the last modified times of the model files.

        Returns:
            A dictionary of model names associated with their respective last modified times.
        """
        last_modified_times = {name: self.get_last_modified_time(name) for name in self.model_names}
        return last_modified_times

    def reload_models_every_hour(self):
        while True:
            time.sleep(3600)  # Wait for one hour
            current_last_modified_times = self.get_last_modified_times()
            for model_name in self.model_names:
                if current_last_modified_times[model_name] != self.last_modified_times.get(model_name):
                    self.models[model_name] = self.load_single_model(model_name)
                    self.last_modified_times[model_name] = current_last_modified_times[model_name]
