from __future__ import annotations

import threading
from pathlib import Path

from sahi.predict import AutoDetectionModel
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


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
        # Ignore directories
        if event.is_directory:
            return

        # Reload model if it is a .pt file
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
            'yolo11',
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
        Cleans up by stopping the file observer thread if it exists.
        """
        if hasattr(self, 'observer') and self.observer is not None:
            self.observer.stop()
            self.observer.join()
