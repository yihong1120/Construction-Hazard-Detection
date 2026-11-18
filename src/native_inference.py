"""Native engine adapter at the verified shared-worker model boundary."""
from __future__ import annotations

import ctypes
import importlib.util
import logging
import math
import sysconfig
import time
from collections import deque
from contextlib import contextmanager
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

from src.gpu_resources import try_acquire_resource
from src.settings.gpu_resources import GpuResourceSettings
from src.settings.inference import WorkerSettings
from src.yolo_shared_frames import _SharedFrameArray
from src.yolo_worker_protocol import Detection
from src.yolo_worker_protocol import FrameArray

logger = logging.getLogger(__name__)


def _preload_runtime(directories: tuple[str, ...]) -> None:
    """Load pip-packaged runtimes before the extension's ELF dependencies."""
    roots = [Path(path) for path in directories]
    site = Path(sysconfig.get_path('platlib'))
    roots.extend((site / 'nvidia/cu13/lib', site / 'tensorrt_libs'))
    for library in (
        'libcudart.so.13', 'libnvrtc.so.13',
        'libnvinfer.so.11', 'libnvinfer_plugin.so.11',
    ):
        for root in roots:
            candidate = root / library
            if candidate.is_file():
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                break


def load_inference_extension(settings: WorkerSettings) -> Any | None:
    if settings.backend == 'python':
        return None
    if importlib.util.find_spec('construction_inference') is None:
        if settings.backend == 'auto':
            return None
        raise RuntimeError(
            'Rust inference requires the native/inference GPU wheel; '
            'install it or select YOLO_WORKER_BACKEND=python',
        )
    if settings.native_library_dirs:
        _preload_runtime(settings.native_library_dirs)
    try:
        extension = importlib.import_module('construction_inference')
    except ImportError:
        if settings.native_library_dirs:
            raise
        _preload_runtime(())
        extension = importlib.import_module('construction_inference')
    if getattr(extension, 'ABI_VERSION', None) != 1:
        raise RuntimeError('Unsupported construction_inference ABI version')
    if not getattr(extension, 'HAS_TENSORRT', False):
        if settings.backend == 'auto':
            return None
        raise RuntimeError('Rust inference wheel needs the tensorrt feature')
    if not callable(getattr(extension, 'TensorRTEngine', None)):
        raise RuntimeError('Incomplete construction_inference extension')
    unsupported = getattr(extension, 'UnsupportedEngineError', None)
    if (
        not isinstance(unsupported, type)
        or not issubclass(unsupported, Exception)
    ):
        raise RuntimeError('Incomplete construction_inference error interface')
    return extension


class RustYoloModel:
    """One engine per worker, returning the existing camera detection rows."""

    def __init__(
        self, extension: Any, path: Path, settings: WorkerSettings,
        *, device: str, rect: bool,
    ) -> None:
        if not device.startswith('cuda:') or not device[5:].isdigit():
            raise ValueError(
                'Rust inference requires an explicit cuda:N device',
            )
        self._async = settings.native_async
        self._micro_batch_size = settings.native_micro_batch_size
        self._rect = rect
        if self._async and getattr(extension, 'ASYNC_API_VERSION', None) != 1:
            raise RuntimeError(
                'Rebuild the inference wheel for asynchronous execution',
            )
        self._resources = GpuResourceSettings()
        self._admission_timeout = settings.gpu_admission_timeout_seconds
        self._memory_lease = None
        if self._resources.memory_limit_mb:
            if (
                settings.backend != 'rust'
                or not settings.gpu_memory_reservation_mb
            ):
                raise ValueError(
                    'Memory admission requires explicit Rust backend '
                    'and model reservation',
                )
            unit = self._resources.memory_unit_mb
            capacity = self._resources.memory_limit_mb // unit
            units = math.ceil(settings.gpu_memory_reservation_mb / unit)
            self._memory_lease = try_acquire_resource(
                'memory', capacity, units,
            )
            if self._memory_lease is None:
                raise RuntimeError('GPU model memory budget exhausted')
        try:
            execution = {'asynchronous': True} if self._async else {}
            self.engine = extension.TensorRTEngine(
                str(path), max_batch=settings.batch_size,
                imgsz=settings.imgsz,
                device=int(device[5:]), rect=rect,
                confidence=0.25, **execution,
            )
        except BaseException:
            if self._memory_lease is not None:
                self._memory_lease.close()
            raise
        self._closed = False
        self._buffer_api = (
            getattr(extension, 'BUFFER_API_VERSION', None) == 1
            and callable(getattr(self.engine, 'predict_buffers', None))
        )
        self.borrowed_frames = 0
        self.snapshot_bytes = 0
        self._catalog_metadata: dict[str, object] | None = None

    @contextmanager
    def _execution_lease(self):
        if self._closed:
            raise RuntimeError('Rust inference engine is closed')
        lease = None
        if self._resources.inference_slots:
            deadline = time.monotonic() + self._admission_timeout
            while lease is None:
                lease = try_acquire_resource(
                    'inference', self._resources.inference_slots,
                )
                if lease is None:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('GPU inference admission timed out')
                    time.sleep(.002)
        try:
            yield
        finally:
            if lease is not None:
                lease.close()

    def predict_detections(
        self, frames: list[FrameArray],
    ) -> list[list[Detection]]:
        with self._execution_lease():
            return self._predict_detections(frames)

    def predict_device_detections(self, frames):
        if not self._async:
            raise RuntimeError('Device frames require asynchronous execution')
        with self._execution_lease():
            capacity = self.engine.max_batch
            if self._micro_batch_size:
                capacity = min(capacity, self._micro_batch_size)
            result = []
            for offset in range(0, len(frames), capacity):
                batch = frames[offset:offset + capacity]
                rows = self.engine.predict_device_frames(batch)
                if len(rows) != len(batch):
                    raise RuntimeError(
                        'Rust engine returned an incomplete batch',
                    )
                result.extend(rows)
            return result

    def _predict_detections(
        self, frames: list[FrameArray],
    ) -> list[list[Detection]]:
        if self._closed:
            raise RuntimeError('Rust inference engine is closed')
        for frame in frames:
            if (
                frame.dtype != np.uint8 or frame.ndim != 3
                or frame.shape[2] != 3 or not frame.size
                or frame.shape[0] * frame.shape[1] > 64 * 1024 * 1024
            ):
                raise ValueError(
                    'Rust inference requires uint8 HWC BGR frames',
                )
        detections: list[list[Detection]] = []
        capacity = self.engine.max_batch
        if capacity < 1:
            raise RuntimeError('Invalid Rust engine batch capacity')
        if self._async and self._micro_batch_size:
            capacity = min(capacity, self._micro_batch_size)
        pending: deque[tuple[int, int]] = deque()
        previous_shape = None
        try:
            for offset in range(0, len(frames), capacity):
                batch = frames[offset:offset + capacity]
                shape = (
                    len(batch), tuple(
                        frame.shape for frame in batch
                    ) if self._rect else None,
                )
                if self._async:
                    if previous_shape != shape:
                        while pending:
                            ticket, count = pending.popleft()
                            detections.extend(self._collect(ticket, count))
                    if len(pending) == 2:
                        ticket, count = pending.popleft()
                        detections.extend(self._collect(ticket, count))
                    pending.append(
                        (self._native_batch(batch, submit=True), len(batch)),
                    )
                    previous_shape = shape
                else:
                    result = self._native_batch(batch)
                    if len(result) != len(batch):
                        raise RuntimeError(
                            'Rust engine returned an incomplete batch',
                        )
                    detections.extend(result)
            while pending:
                ticket, count = pending.popleft()
                detections.extend(self._collect(ticket, count))
        finally:
            while pending:
                ticket, _ = pending.popleft()
                with suppress(Exception):
                    self.engine.collect(ticket)
        return detections

    def _collect(self, ticket, count):
        result = self.engine.collect(ticket)
        if len(result) != count:
            raise RuntimeError('Rust engine returned an incomplete batch')
        return result

    def _native_batch(self, batch, *, submit=False):
        if self._buffer_api and all(
            isinstance(frame, _SharedFrameArray)
            and getattr(frame, 'shared_memory_handle', None) is not None
            and frame.flags.c_contiguous
            for frame in batch
        ):
            # Only worker mappings carry an exclusive producer lease.
            # Rust finishes reading them with the GIL held and sends its
            # own resized pixels to CUDA after releasing the GIL.
            views = [memoryview(frame).toreadonly() for frame in batch]
            self.borrowed_frames += len(batch)
            try:
                operation = (
                    self.engine.submit_buffers if submit
                    else self.engine.predict_buffers
                )
                result = operation(views)
            finally:
                for view in views:
                    view.release()
        else:
            # Generic arrays may be mutated by another Python thread.
            inputs = [
                (frame.shape[0], frame.shape[1], frame.tobytes(order='C'))
                for frame in batch
            ]
            self.snapshot_bytes += sum(len(data) for _, _, data in inputs)
            operation = self.engine.submit if submit else self.engine.predict
            result = operation(inputs)
        return result

    def close(self) -> None:
        if not self._closed:
            self.engine.close()
            if self._memory_lease is not None:
                self._memory_lease.close()
            self._closed = True

    def __del__(self):
        if not getattr(self, '_closed', True):
            with suppress(Exception):
                self.close()


def create_rust_model(
    path: Path, settings: WorkerSettings, *, device: str, rect: bool,
) -> RustYoloModel | None:
    resources = GpuResourceSettings()
    if (
        resources.memory_limit_mb or resources.inference_slots
    ) and settings.backend != 'rust':
        raise ValueError('GPU admission requires explicit Rust backend')
    if settings.backend == 'python':
        return None
    eligible = path.suffix == '.engine' and device.startswith('cuda:')
    if not eligible:
        if settings.backend == 'rust':
            raise ValueError('Rust inference requires a verified CUDA .engine')
        return None
    extension = load_inference_extension(settings)
    if extension is None:
        return None
    try:
        return RustYoloModel(
            extension, path, settings, device=device, rect=rect,
        )
    except extension.UnsupportedEngineError:
        if settings.backend == 'rust':
            raise
        logger.info('Engine unsupported by Rust; selecting Python: %s', path)
        return None
