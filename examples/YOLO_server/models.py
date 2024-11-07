from __future__ import annotations

import threading
from collections.abc import Generator
from datetime import datetime
from datetime import timezone
from pathlib import Path

from sahi.predict import AutoDetectionModel
from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import create_engine
from sqlalchemy import DateTime
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from werkzeug.security import check_password_hash
from werkzeug.security import generate_password_hash

from .config import Settings

# Load settings
settings = Settings()

# Set up SQLAlchemy engine and session
engine = create_engine(settings.sqlalchemy_database_uri)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base for SQLAlchemy ORM models
Base: type = declarative_base()


def get_db() -> Generator[Session]:
    """
    Provides a database session generator, ensuring that the session is
    closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class User(Base):
    """
    Represents a user entity in the database with password hashing for
    security.
    """

    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(80), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    # admim, model_manage, user, guest
    role = Column(String(20), default='user', nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(
        DateTime, default=datetime.now(
            timezone.utc,
        ), nullable=False,
    )
    updated_at = Column(
        DateTime, default=datetime.now(timezone.utc),
        onupdate=datetime.now(timezone.utc),
        nullable=False,
    )

    def set_password(self, password: str) -> None:
        """
        Hashes the given password and stores it as the password hash.

        Args:
            password (str): The plaintext password to be hashed.
        """
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """
        Checks if the provided password matches the stored hashed password.

        Args:
            password (str): The plaintext password to verify.

        Returns:
            bool: True if the password is correct, False otherwise.
        """
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        """
        Returns the user entity as a dictionary.

        Returns:
            dict: The user entity as a dictionary.
        """
        return {
            'id': self.id,
            'username': self.username,
            'role': self.role,
            'is_active': self.is_active,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        }


class ModelFileChangeHandler(FileSystemEventHandler):
    """
    Handles file system events for model files, triggering reloads on
    modification.
    """

    def __init__(self, model_manager: DetectionModelManager) -> None:
        """
        Initialises the file change handler with a model manager.

        Args:
            model_manager (DetectionModelManager): The manager responsible for
            loading models.
        """
        self.model_manager = model_manager

    def on_modified(self, event) -> None:
        """
        Handles the modification event, reloading models
        if relevant files are updated.

        Args:
            event: The file system event.
        """
        #
        if event.is_directory:
            return

        #
        if event.src_path.endswith('.pt'):
            model_name = Path(event.src_path).stem.split('best_')[-1]
            if model_name in self.model_manager.model_names:
                # Reload the model in the manager
                self.model_manager.models[model_name] = (
                    self.model_manager.load_single_model(model_name)
                )
                print(f"Model {model_name} reloaded due to file modification.")


class DetectionModelManager:
    """
    Manages the loading and access of object detection models
    with file system monitoring.
    """

    def __init__(self) -> None:
        """
        Initialises the model manager, loading models
        and setting up a file monitor.
        """
        self.base_model_path: Path = Path('models/pt/')
        self.model_names: list[str] = [
            'yolo11x',
            'yolo11l', 'yolo11m', 'yolo11s', 'yolo11n',
        ]
        # Load each model
        self.models: dict[str, AutoDetectionModel] = {
            name: self.load_single_model(name) for name in self.model_names
        }

        # Set up a watchdog observer for monitoring model file changes
        self.event_handler = ModelFileChangeHandler(self)
        self.observer = Observer()
        self.observer.schedule(
            self.event_handler, str(
                self.base_model_path,
            ), recursive=False,
        )
        # Run the observer in a separate thread
        self.observer_thread = threading.Thread(target=self.observer.start)
        self.observer_thread.start()

    def load_single_model(self, model_name: str) -> AutoDetectionModel:
        """
        Loads a specified model from a file and returns it
        as an AutoDetectionModel.

        Args:
            model_name (str): The name of the model to load.

        Returns:
            AutoDetectionModel: The loaded model ready for predictions.
        """
        return AutoDetectionModel.from_pretrained(
            'yolov8',
            model_path=str(self.base_model_path / f"best_{model_name}.pt"),
            device='cuda:0',
        )

    def get_model(self, model_key: str) -> AutoDetectionModel | None:
        """
        Retrieves a model by its key if it exists within the loaded models.

        Args:
            model_key (str): The key name of the model to retrieve.

        Returns:
            AutoDetectionModel | None: The requested model
            or None if it does not exist.
        """
        return self.models.get(model_key)

    def __del__(self) -> None:
        """
        Cleans up by stopping the file observer thread.
        """
        self.observer.stop()
        self.observer.join()
