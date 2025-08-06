from __future__ import annotations

import threading
from pathlib import Path
from typing import Union

from watchdog.events import FileSystemEvent
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from examples.YOLO_server_api.backend.config import USE_TENSORRT

if USE_TENSORRT:
    from ultralytics import YOLO
else:
    from sahi.predict import AutoDetectionModel

# Type alias for model objects (varies based on USE_TENSORRT configuration)
ModelType = Union['YOLO', 'AutoDetectionModel']


class ModelFileChangeHandler(FileSystemEventHandler):
    """
    File system event handler for automatic model hot-reloading.
    """

    def __init__(self, model_manager: DetectionModelManager) -> None:
        """Initialise the file change handler.

        Args:
            model_manager: Reference to the DetectionModelManager instance that
                manages the models to be hot-reloaded.
        """
        super().__init__()
        self.model_manager = model_manager

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events for model hot-reloading.

        Triggered when a file in the monitored directory is modified. Checks if
        the modified file is a model file that should trigger a reload.

        Args:
            event: File system event containing details
                about the modified file.

        Note:
            Only processes files with the model extension (.engine or .pt)
            and matching model names from the predefined model list.
        """
        # Ignore directory modification events
        if event.is_directory:
            return

        # Check if the modified file has the correct model extension
        if event.src_path.endswith(self.model_manager.extension):
            # Extract model name from filename (removes 'best_' prefix)
            name = Path(event.src_path).stem.split('best_')[-1]

            # Only reload if it's a recognised model name
            if name in self.model_manager.model_names:
                self.model_manager.models[name] = (
                    self.model_manager.load_single_model(name)
                )
                print(f'🟢  Model {name} hot-reloaded.')


class DetectionModelManager:
    """
    Comprehensive YOLO model manager
    with hot-reloading and multi-variant support.

    Attributes:
        model_names: List of supported YOLO model variant names.
        base_model_path: Path to the directory containing model files.
        extension: File extension for model files (.engine or .pt).
        models: Dictionary mapping model names to loaded model instances.
        event_handler: File system event handler for hot-reloading.
        observer: File system observer for monitoring model file changes.
        observer_thread: Background thread running the file system observer.
    """

    def __init__(self) -> None:
        """Initialise the detection model manager.

        Sets up model loading, configures file system monitoring, and starts
        the hot-reloading background thread.

        Note:
            Automatically determine model format (TensorRT vs PyTorch) based on
            the USE_TENSORRT configuration flag and loads all available models.
        """
        # Define supported YOLO model variants (from smallest to largest)
        self.model_names: list[str] = [
            'yolo11x',  # Extra large model (highest accuracy, slowest)
            'yolo11l',  # Large model
            'yolo11m',  # Medium model
            'yolo11s',  # Small model
            'yolo11n',  # Nano model (fastest, lowest accuracy)
        ]

        # Configure model paths and extensions based on deployment type
        if USE_TENSORRT:
            self.base_model_path: Path = Path('models/int8_engine/')
            self.extension: str = '.engine'
        else:
            self.base_model_path: Path = Path('models/pt/')
            self.extension: str = '.pt'

        # Load all available models into memory
        self.models: dict[str, ModelType] = {
            name: self.load_single_model(name) for name in self.model_names
        }

        # Set up file system monitoring for hot-reloading
        self.event_handler: ModelFileChangeHandler = ModelFileChangeHandler(
            self,
        )
        self.observer = Observer()
        self.observer.schedule(
            self.event_handler,
            str(self.base_model_path),
            recursive=False,
        )

        # Start the file monitoring in a background daemon thread
        self.observer_thread: threading.Thread = threading.Thread(
            target=self.observer.start,
            daemon=True,
        )
        self.observer_thread.start()

    # Internal Methods
    def load_single_model(self, name: str) -> ModelType:
        """Load a single YOLO model by name.

        Creates and loads either a TensorRT optimised model
        or a standard PyTorch model based on the current configuration.

        Args:
            name: Model variant name (e.g., 'yolo11x', 'yolo11s').

        Returns:
            Loaded model instance (either YOLO or AutoDetectionModel).

        Raises:
            FileNotFoundError: If the specified model file doesn't exist.
            RuntimeError: If model loading fails due to incompatible formats.

        Note:
            - For TensorRT,
                expects model files to be in 'models/int8_engine/' directory.
            - For PyTorch,
                expects model files to be in 'models/pt/' directory.
            - Automatically handles both formats
                based on the USE_TENSORRT configuration.
        """
        if USE_TENSORRT:
            # Load TensorRT optimised model for maximum inference speed
            model_path = self.base_model_path / f'best_{name}.engine'
            return YOLO(str(model_path))

        # Load PyTorch model with SAHI for enhanced small object detection
        model_path = self.base_model_path / f'best_{name}.pt'
        return AutoDetectionModel.from_pretrained(
            'yolo11',
            model_path=str(model_path),
            device='cuda:0',  # Use first CUDA device for inference
        )

    # Public Interface
    def get_model(self, key: str) -> ModelType | None:
        """
        Retrieve a loaded model by its variant name.

        Args:
            key: Model variant name (e.g., 'yolo11x', 'yolo11s', 'yolo11n').

        Returns:
            The requested model instance if available, None if the model name
            is not recognised or the model failed to load.

        Example:
            ```python
            model_manager = DetectionModelManager()
            yolo_model = model_manager.get_model('yolo11x')
            if yolo_model:
                results = yolo_model.predict(image)
            ```

        Note:
            This method is thread-safe and can be called concurrently from
            multiple threads without synchronisation concerns.
        """
        return self.models.get(key)

    def get_available_models(self) -> list[str]:
        """Get list of successfully loaded model names.

        Returns:
            List of model variant names that are currently loaded and available
            for inference.

        Note:
            This list may be shorter than model_names if some models failed
            to load due to missing files or configuration issues.
        """
        return [
            name
            for name, model in self.models.items()
            if model is not None
        ]

    def is_model_loaded(self, name: str) -> bool:
        """Check if a specific model variant is loaded and ready for use.

        Args:
            name: Model variant name to check.

        Returns:
            True if the model is loaded and available, False otherwise.
        """
        return name in self.models and self.models[name] is not None

    def reload_model(self, name: str) -> bool:
        """
        Manually reload a specific model variant.

        Args:
            name: Model variant name to reload.

        Returns:
            True if the model was successfully reloaded, False if the model
            name is invalid or reloading failed.

        Example:
            ```python
            if model_manager.reload_model('yolo11x'):
                print("Model reloaded successfully")
            ```
        """
        if name not in self.model_names:
            return False

        try:
            self.models[name] = self.load_single_model(name)
            print(f'🔄  Model {name} manually reloaded.')
            return True
        except Exception as e:
            print(f'❌  Failed to reload model {name}: {e}')
            return False

    # Cleanup and Lifecycle Management
    def __del__(self) -> None:
        """Clean up resources when the model manager is destroyed.

        Ensures proper shutdown of the file system observer and background
        monitoring thread to prevent resource leaks.

        Note:
            This method is automatically called during garbage collection.
            It's safe to call multiple times and handles cases where the
            observer was never initialised.
        """
        # Safely stop the file system observer if it exists
        if hasattr(self, 'observer') and self.observer is not None:
            self.observer.stop()
            self.observer.join()
