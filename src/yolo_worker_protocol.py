from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from typing import Self
from typing import TypeAlias
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from src.ultralytics_args import precision_kwargs
from src.ultralytics_args import PrecisionValue

YOLO_WORKER_STOP_MESSAGE = '__stop__'
_WORKER_PRECISION_ALIASES = {
    '32': 'f32',
    'float32': 'f32',
    'fp32': 'f32',
    'f32': 'f32',
    '16': 'f16',
    'float16': 'f16',
    'fp16': 'f16',
    'f16': 'f16',
    '8': 'int8',
    'engine': 'int8',
    'int8': 'int8',
}

Detection: TypeAlias = list[float]
FrameArray: TypeAlias = NDArray[np.uint8]
WorkerPrecisionMode: TypeAlias = str


def _parse_worker_precision(
    raw_value: str | None,
) -> WorkerPrecisionMode | None:
    """Return a simplified worker precision mode from environment text."""
    if raw_value is None:
        return None
    value = raw_value.strip().lower()
    if value in {'', 'none', 'null', 'default', 'auto', 'legacy'}:
        return None
    try:
        return _WORKER_PRECISION_ALIASES[value]
    except KeyError as exc:
        supported = ', '.join(sorted(set(_WORKER_PRECISION_ALIASES)))
        raise ValueError(
            f"Unsupported YOLO_WORKER_PRECISION: {raw_value!r}. "
            f"Use f32, f16, or int8. Supported aliases: {supported}.",
        ) from exc


def _worker_precision_config(
    mode: WorkerPrecisionMode,
) -> tuple[Path, str, dict[str, PrecisionValue]]:
    """Map a simplified precision mode to model path and predict kwargs."""
    if mode == 'f32':
        return Path('models/pt'), '.pt', precision_kwargs(False, 32)
    if mode == 'f16':
        return Path('models/pt'), '.pt', precision_kwargs(True, 16)
    if mode == 'int8':
        # TensorRT engine precision is baked into the .engine file. These
        # engines are dynamic batch only and require fixed square 640 inputs.
        return Path('models/int8_engine'), '.engine', {'rect': False}
    raise AssertionError(f"unhandled YOLO worker precision mode: {mode}")


class WorkerRequestPayload(TypedDict):
    """Serialisable request payload passed through the worker queue."""

    id: str
    camera_id: str
    model_key: str
    shm_name: str
    slot: int
    shape: tuple[int, ...]
    dtype: str


class WorkerResult(TypedDict, total=False):
    """Serialisable result payload sent through one camera result queue."""

    id: str
    ok: bool
    detections: list[Detection]
    error: str
    skipped: bool


class WorkerRequestSender(Protocol):
    """Queue interface needed to submit inference requests."""

    def put(
        self,
        obj: WorkerRequestPayload,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        """Put one inference request into the queue."""


class WorkerResultSender(Protocol):
    """Queue interface used by a worker to return one camera result."""

    def put(
        self,
        obj: WorkerResult,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        """Put one result payload into the queue."""


class WorkerResultReceiver(Protocol):
    """Queue interface used by a stream client to await results."""

    def get(
        self,
        block: bool = True,
        timeout: float | None = None,
    ) -> WorkerResult:
        """Return one worker result payload."""


class WorkerQueue(Protocol):
    """Queue interface used by the worker to consume and requeue messages."""

    def put(
        self,
        obj: object,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        """Put one object into the queue."""

    def get(self, block: bool = True, timeout: float | None = None) -> object:
        """Return one object from the queue."""

    def get_nowait(self) -> object:
        """Return one object immediately or raise ``queue.Empty``."""


class TensorLike(Protocol):
    """Minimal tensor interface used for Ultralytics box data."""

    def cpu(self) -> TensorLike:
        """Return a CPU-backed tensor-like object."""

    def numpy(self) -> NDArray[np.float64]:
        """Return the tensor data as a NumPy array."""


class BoxesLike(Protocol):
    """Minimal Ultralytics boxes interface used by this worker."""

    data: TensorLike


class YoloResultLike(Protocol):
    """Minimal Ultralytics prediction result interface used by this worker."""

    boxes: BoxesLike


class YoloModelLike(Protocol):
    """Minimal Ultralytics model interface used by this worker."""

    def predict(
        self,
        source: list[FrameArray],
        verbose: bool,
        device: str,
        imgsz: int,
        batch: int,
        **kwargs: PrecisionValue,
    ) -> Iterable[object]:
        """Run model inference and return Ultralytics-style results."""


@dataclass(frozen=True)
class _WorkerRequest:
    """Normalised worker request metadata.

    Attributes:
        id: Unique request identifier used to match a result.
        camera_id: Stable camera key. Only the latest request per camera is
            retained while batching.
        model_key: Model suffix used to locate the YOLO weight file.
        shm_name: POSIX shared memory name containing the frame bytes.
        shape: NumPy frame shape.
        dtype: NumPy dtype name for the shared frame.
    """

    id: str
    camera_id: str
    model_key: str
    shm_name: str
    shape: tuple[int, ...]
    dtype: str
    slot: int = 0

    @classmethod
    def from_mapping(
        cls: type[Self],
        request: WorkerRequestPayload,
    ) -> _WorkerRequest:
        """Build a request object from a queue payload.

        Args:
            request: Serialisable request payload submitted by
                ``YoloWorkerClient``.

        Returns:
            A normalised immutable worker request.
        """
        return cls(
            id=str(request['id']),
            camera_id=str(request['camera_id']),
            model_key=str(request['model_key']),
            shm_name=str(request['shm_name']),
            slot=int(request.get('slot', 0)),
            shape=tuple(int(v) for v in request['shape']),
            dtype=str(request['dtype']),
        )
