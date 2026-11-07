from __future__ import annotations

import hashlib
import tempfile
import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from fastapi import HTTPException
lazy from sahi.predict import AutoDetectionModel
lazy from ultralytics import YOLO

from examples.YOLO_server_api.config import get_inference_device
from examples.YOLO_server_api.config import MAX_LOADED_MODELS
from examples.YOLO_server_api.config import USE_SAHI
from examples.YOLO_server_api.model_registry import artifact_path
from examples.YOLO_server_api.model_registry import ModelEntry


@dataclass
class ModelSlot:
    model: Any = None
    loading: bool = True
    error: BaseException | None = None
    users: int = 0
    inference_lock: Any = field(default_factory=threading.Lock)


class DetectionModelManager:

    def __init__(self, capacity: int = MAX_LOADED_MODELS) -> None:
        if capacity < 1:
            raise ValueError('Model capacity must be positive')
        self.capacity = capacity
        self._condition = threading.Condition()
        self._slots: OrderedDict[tuple[str, str, str], ModelSlot] = (
            OrderedDict()
        )
        self._closed = False

    @contextmanager
    def lease(
        self,
        entry: ModelEntry,
        cancelled: threading.Event | None = None,
    ):
        key = (entry.id, entry.version, entry.sha256)
        load = False
        with self._condition:
            while True:
                if self._closed or (
                    cancelled is not None and cancelled.is_set()
                ):
                    raise RuntimeError('Model request cancelled')
                slot = self._slots.get(key)
                if slot is not None:
                    while slot.loading:
                        self._condition.wait(0.05)
                        if self._closed or (
                            cancelled is not None and cancelled.is_set()
                        ):
                            raise RuntimeError('Model request cancelled')
                    if slot.error is not None:
                        raise slot.error
                    if self._slots.get(key) is not slot:
                        continue
                    slot.users += 1
                    self._slots.move_to_end(key)
                    break
                if len(self._slots) < self.capacity:
                    slot = ModelSlot(users=1)
                    self._slots[key] = slot
                    load = True
                    break
                idle = next(
                    (
                        item
                        for item, value in self._slots.items()
                        if not value.loading and value.users == 0
                    ),
                    None,
                )
                if idle is not None:
                    retired = self._slots.pop(idle)
                    self._dispose(retired.model)
                    retired.model = None
                    continue
                self._condition.wait(0.05)
        if load:
            try:
                model = self._load_registry_model(entry)
            except BaseException as error:
                with self._condition:
                    slot.error = error
                    slot.loading = False
                    self._slots.pop(key, None)
                    self._condition.notify_all()
                raise
            with self._condition:
                slot.model = model
                slot.loading = False
                self._condition.notify_all()
        try:
            with slot.inference_lock:
                if cancelled is not None and cancelled.is_set():
                    raise RuntimeError('Model request cancelled')
                yield slot.model
        finally:
            with self._condition:
                slot.users -= 1
                self._slots.move_to_end(key)
                self._condition.notify_all()

    @staticmethod
    def _dispose(model) -> None:
        temporary = getattr(model, '_registry_artifact_directory', None)
        if temporary is not None:
            temporary.cleanup()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            while any(
                slot.users or slot.loading for slot in self._slots.values()
            ):
                self._condition.wait(0.05)
            for slot in self._slots.values():
                self._dispose(slot.model)
            self._slots.clear()

    def _load_registry_model(self, entry):
        """Load a verified immutable artifact copy."""
        # A fresh model reference binds this request to the selected version.
        # Temporary artifacts survive inference (TensorRT may load lazily).
        path = artifact_path(entry)
        temporary = tempfile.TemporaryDirectory(prefix='visionnaire-model-')
        copied = Path(temporary.name) / ('model' + path.suffix)
        try:
            digest = hashlib.sha256()
            with path.open('rb') as source, copied.open('wb') as target:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    target.write(chunk)
            if digest.hexdigest() != entry.sha256:
                raise ValueError('Artifact version mismatch')
            if USE_SAHI:
                model = AutoDetectionModel.from_pretrained(
                    'yolo26',
                    model_path=str(copied),
                    device=get_inference_device(),
                )
            else:
                model = YOLO(str(copied))
            model._registry_artifact_directory = temporary
            return model
        except Exception as exc:
            temporary.cleanup()
            raise HTTPException(
                409,
                detail={'code': 'MODEL_UNAVAILABLE'},
            ) from exc
