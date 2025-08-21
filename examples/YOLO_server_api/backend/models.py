from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Union

from ultralytics import YOLO
from watchdog.events import FileSystemEvent
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from examples.YOLO_server_api.backend.config import EXPLICIT_CUDA_CLEANUP
from examples.YOLO_server_api.backend.config import LAZY_LOAD_MODELS
from examples.YOLO_server_api.backend.config import MAX_LOADED_MODELS
from examples.YOLO_server_api.backend.config import MODEL_VARIANTS
from examples.YOLO_server_api.backend.config import PRELOAD_SMALLEST
from examples.YOLO_server_api.backend.config import USE_SAHI
from examples.YOLO_server_api.backend.config import USE_TENSORRT
# YOLO is required in all modes (standard .pt and TensorRT .engine),
# import unconditionally to simplify control flow.

# For type hints only, avoid importing SAHI at runtime unless needed.
if TYPE_CHECKING:
    from sahi.predict import AutoDetectionModel

# Type alias used across the manager
ModelType = Union['YOLO', 'AutoDetectionModel']


class ModelFileChangeHandler(FileSystemEventHandler):
    """File change watcher used to support hot-reloading of models.

    Monitors the model directory for modified files matching the expected
    model file extension and triggers a reload of the corresponding model.
    """

    def __init__(self, model_manager: DetectionModelManager) -> None:
        """Initialise the file change handler.

        Args:
            model_manager: The parent detection model manager instance.
        """
        super().__init__()
        self.model_manager: DetectionModelManager = model_manager

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle a file modification event.

        Reloads the affected model if the modified file matches the
        manager's model file extension and the model name is recognised.

        Args:
            event: The file system event data.
        """
        if event.is_directory:
            return
        if not event.src_path.endswith(self.model_manager.extension):
            return
        name = Path(event.src_path).stem.split('best_')[-1]
        if name in self.model_manager.model_names:
            self.model_manager._safe_load(name)
            print(f'🟢  Model {name} hot-reloaded (watcher).')


class DetectionModelManager:
    """
    YOLO detection model manager with lazy load, LRU eviction, and hot reload.
    """

    def __init__(self) -> None:
        """Initialise the model manager, paths, caches, and file watcher."""
        # 1) Model names from configuration (order affects LRU priority)
        self.model_names: list[str] = MODEL_VARIANTS

        # 2) Base path and extension - SAHI forces .pt file usage
        if USE_SAHI or not USE_TENSORRT:
            self.base_model_path: Path = Path('models/pt')
            self.extension: str = '.pt'
        else:  # USE_TENSORRT=True and USE_SAHI=False
            self.base_model_path = Path('models/int8_engine')
            self.extension = '.engine'

        # 3) Model cache (None until loaded in lazy mode)
        self.models: dict[str, ModelType | None] = {
            name: None for name in self.model_names
        }
        self._lru_order: list[str] = []  # LRU usage order (oldest at index 0)

        # 4) Pre-load strategy
        if not LAZY_LOAD_MODELS:
            for name in self.model_names:
                self._safe_load(name)
        elif PRELOAD_SMALLEST and self.model_names:
            # Pre-load the smallest model (assumed to be the last in the list)
            self._safe_load(self.model_names[-1])

        # 5) File system monitoring (hot reload)
        self.event_handler = ModelFileChangeHandler(self)
        self.observer = Observer()
        self.observer.schedule(
            self.event_handler,
            str(self.base_model_path),
            recursive=False,
        )
        self.observer_thread = threading.Thread(
            target=self.observer.start,
            daemon=True,
        )
        self.observer_thread.start()

    # =============== Internal ==================
    def load_single_model(self, name: str) -> ModelType:
        """Load a single model according to the current configuration.

        Creates and returns either a TensorRT-optimised model, a SAHI
        ``AutoDetectionModel``, or a standard PyTorch YOLO model.

        Args:
            name: Model variant name (e.g., 'yolo11x', 'yolo11s').

        Returns:
            A loaded model instance (YOLO or AutoDetectionModel).

        Raises:
            FileNotFoundError: If the specified model file does not exist.
            RuntimeError: If a required dependency (e.g., SAHI) is missing.
        """
        if USE_SAHI:
            # SAHI mode: Use AutoDetectionModel with .pt files (slicing)
            try:
                from sahi.predict import AutoDetectionModel
            except Exception as exc:
                raise RuntimeError(
                    'SAHI mode is enabled but the sahi package is not '
                    'installed.',
                ) from exc
            model_path = self.base_model_path / f'best_{name}.pt'
            if not model_path.exists():
                raise FileNotFoundError(f'Model file not found: {model_path}')
            return AutoDetectionModel.from_pretrained(
                'yolo11', model_path=str(model_path), device='cuda:0',
            )

        if USE_TENSORRT:
            # TensorRT mode: Use .engine files with YOLO
            model_path = self.base_model_path / f'best_{name}.engine'
            if not model_path.exists():
                raise FileNotFoundError(f'Model file not found: {model_path}')
            return YOLO(str(model_path))

        # Standard mode: Use .pt files with YOLO
        model_path = self.base_model_path / f'best_{name}.pt'
        if not model_path.exists():
            raise FileNotFoundError(f'Model file not found: {model_path}')
        return YOLO(str(model_path))

    def _safe_load(self, name: str) -> None:
        """Load a model safely and update LRU bookkeeping.

        If the model is already cached, only its LRU position is updated.
        Otherwise, the model is loaded and inserted into the cache, with
        LRU limit enforcement applied afterwards.

        Args:
            name: Model variant name to load.
        """
        if name not in self.model_names:
            return
        if self.models.get(name) is not None:
            # Already loaded, update LRU order
            self._touch_lru(name)
            return
        try:
            self.models[name] = self.load_single_model(name)
            self._touch_lru(name)
            print(f'✅ Loaded model: {name}')
            self._enforce_lru_limit()
        except Exception as e:
            self.models[name] = None
            print(f'❌ Failed loading model {name}: {e}')

    def _touch_lru(self, name: str) -> None:
        """Mark a model as recently used by updating the LRU order.

        Args:
            name: Model variant name to mark as used.
        """
        if name in self._lru_order:
            self._lru_order.remove(name)
        self._lru_order.append(name)

    def _enforce_lru_limit(self) -> None:
        """
        Ensure loaded models do not exceed the configured limit.

        When lazy loading is enabled and the cache grows beyond
        ``MAX_LOADED_MODELS``, evict the least recently used model. Optionally
        perform explicit CUDA cache cleanup when using PyTorch models.
        """
        if not LAZY_LOAD_MODELS:
            return

        # Enforce LRU limit
        loaded_count = len([m for m in self.models.values() if m is not None])
        while loaded_count > MAX_LOADED_MODELS:
            # Remove the least recently used model
            if not self._lru_order:  # Safety check
                break
            evict_name = self._lru_order.pop(0)
            if self.models.get(evict_name) is not None:
                try:
                    # Release reference explicitly
                    self.models[evict_name] = None
                    print(f'🧹 Evicted model (LRU): {evict_name}')
                    # Perform CUDA cache cleanup only for PyTorch (.pt) mode
                    if not USE_TENSORRT and EXPLICIT_CUDA_CLEANUP:
                        try:
                            # Lazy import torch only when needed
                            import torch

                            torch.cuda.empty_cache()
                            print('💡 torch.cuda.empty_cache() called')
                        except Exception:
                            pass
                except Exception:
                    pass
            # Recalculate loaded count after eviction
            loaded_count = len(
                [m for m in self.models.values() if m is not None])

    # =============== Public API ==================
    def get_model(self, key: str) -> ModelType | None:
        """Retrieve a loaded model by its variant name.

        If the model is not yet loaded and lazy loading is enabled, this will
        load the model on demand.

        Args:
            key: Model variant name (e.g., 'yolo11x', 'yolo11s', 'yolo11n').

        Returns:
            The requested model instance if available; otherwise ``None``.
        """
        # Lazy load on demand
        if key not in self.models:
            return None
        if self.models[key] is None:
            self._safe_load(key)
        else:
            self._touch_lru(key)
        return self.models.get(key)

    def get_available_models(self) -> list[str]:
        """Return the list of currently loaded model names.

        Returns:
            A list of model variant names that are loaded and available for
            inference at the time of the call.
        """
        return [
            name for name, model in self.models.items() if model is not None
        ]

    def is_model_loaded(self, name: str) -> bool:
        """Check if a model variant is loaded and ready for inference.

        Args:
            name: Model variant name to check.

        Returns:
            ``True`` if the model is loaded; otherwise ``False``.
        """
        return name in self.models and self.models[name] is not None

    def reload_model(self, name: str) -> bool:
        """Manually reload a specific model variant.

        Args:
            name: Model variant name to reload.

        Returns:
            ``True`` if reloading succeeded; otherwise ``False``.
        """
        if name not in self.model_names:
            return False
        # Force reload regardless of cache state
        try:
            self.models[name] = self.load_single_model(name)
            self._touch_lru(name)
            self._enforce_lru_limit()
            print(f'🔄  Model {name} manually reloaded.')
            return True
        except Exception as e:
            print(f'❌  Failed to reload model {name}: {e}')
            return False

    # =============== Cleanup ==================
    def __del__(self) -> None:
        """Clean up resources when the model manager is destroyed.

        Ensures proper shutdown of the file system observer and background
        monitoring thread to prevent resource leaks. Safe to call multiple
        times and handles cases where the observer was never initialised.
        """
        # Safely stop the file system observer if it exists
        if hasattr(self, 'observer') and self.observer is not None:
            self.observer.stop()
            self.observer.join()
