from __future__ import annotations

import asyncio
import logging
import os
import queue
import time
import uuid
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any
from typing import cast
from typing import Protocol
from typing import Self
from typing import TypeAlias
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from src.ultralytics_args import parse_quantize_value
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
            f'Unsupported YOLO_WORKER_PRECISION: {raw_value!r}. '
            f'Use f32, f16, or int8. Supported aliases: {supported}.',
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
    raise AssertionError(f'unhandled YOLO worker precision mode: {mode}')


class WorkerRequestPayload(TypedDict):
    """Serialisable request payload passed through the worker queue."""

    id: str
    camera_id: str
    model_key: str
    shm_name: str
    slot: int
    shape: tuple[int, ...]
    dtype: str
    result_queue: WorkerResultSender


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
    result_queue: WorkerResultSender
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
            result_queue=cast(WorkerResultSender, request['result_queue']),
        )


class _SharedFrameArray(np.ndarray):
    """NumPy view that keeps its shared-memory mapping alive for inference."""

    shared_memory_handle: shared_memory.SharedMemory | None


@dataclass
class _SharedFrameRing:
    """Fixed shared-memory slots owned by one stream-process camera client."""

    shape: tuple[int, ...]
    dtype: np.dtype[np.uint8]
    slots: int
    shared_memory_handle: shared_memory.SharedMemory
    _available_slots: asyncio.Queue[int]

    @classmethod
    def create(
        cls: type[Self],
        shape: tuple[int, ...],
        dtype: np.dtype[np.uint8],
        slots: int,
    ) -> _SharedFrameRing:
        """Allocate a bounded shared-memory ring for one frame format."""
        slot_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        shared_memory_handle = shared_memory.SharedMemory(
            create=True,
            size=slot_bytes * slots,
        )
        available_slots: asyncio.Queue[int] = asyncio.Queue(maxsize=slots)
        for slot in range(slots):
            available_slots.put_nowait(slot)
        return cls(
            shape=shape,
            dtype=dtype,
            slots=slots,
            shared_memory_handle=shared_memory_handle,
            _available_slots=available_slots,
        )

    @property
    def name(self) -> str:
        """Return the POSIX shared-memory name used by worker requests."""
        return self.shared_memory_handle.name

    @property
    def slot_bytes(self) -> int:
        """Return the byte capacity of one fixed frame slot."""
        return int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize

    async def acquire_slot(self, timeout_seconds: float) -> int:
        """Reserve one slot until the corresponding worker result arrives."""
        try:
            return await asyncio.wait_for(
                self._available_slots.get(),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                'YOLO worker shared frame ring is full',
            ) from exc

    def release_slot(self, slot: int) -> None:
        """Make a finished slot available for the next camera frame."""
        with suppress(asyncio.QueueFull):
            self._available_slots.put_nowait(slot)

    def write(self, slot: int, frame: FrameArray) -> None:
        """Copy one contiguous frame into its reserved shared-memory slot."""
        target = np.ndarray(
            self.shape,
            dtype=self.dtype,
            buffer=self.shared_memory_handle.buf,
            offset=slot * self.slot_bytes,
        )
        target[:] = frame

    def close(self) -> None:
        """Close and unlink this ring after all users have stopped."""
        self.shared_memory_handle.close()
        with suppress(FileNotFoundError):
            self.shared_memory_handle.unlink()


class YoloWorkerClient:
    """Asynchronous client used by stream processes for YOLO inference.

    A fixed, per-camera shared-memory ring avoids allocating one POSIX segment
    per frame. Requests contain only a ring slot descriptor; results return on
    a camera-specific queue so no stream needs to poll a shared manager dict.
    """

    def __init__(
        self,
        request_queue: WorkerRequestSender,
        result_queue: WorkerResultReceiver,
        camera_id: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Initialise the client.

        Args:
            request_queue: Queue used to submit frame metadata.
            result_queue: Dedicated queue used to receive worker results.
            camera_id: Stable site/camera key for latest-frame coalescing.
            timeout_seconds: Maximum time to wait for queue submission and
                inference result.
        """
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.camera_id = camera_id
        self.timeout_seconds = timeout_seconds
        self.ring_slots = max(
            1,
            int(os.getenv('YOLO_WORKER_RING_SLOTS', '2')),
        )
        self.ring_cleanup_delay_seconds = max(
            timeout_seconds,
            float(
                os.getenv(
                    'YOLO_WORKER_RING_SLOT_CLEANUP_SECONDS',
                    '120.0',
                ),
            ),
        )
        self._ring: _SharedFrameRing | None = None
        self._retired_rings: list[_SharedFrameRing] = []
        self._inflight_slots: dict[str, tuple[_SharedFrameRing, int]] = {}
        self._ring_lock = asyncio.Lock()
        self._closed = False

    async def detect(
        self,
        frame: FrameArray,
        model_key: str,
    ) -> list[Detection]:
        """Send one frame to the worker and await detection results.

        Args:
            frame: BGR frame to run through YOLO.
            model_key: Model key used by the worker.

        Returns:
            A list of detections in ``[x1, y1, x2, y2, score, class_id]``
            format.

        Raises:
            TimeoutError: If queue submission or result waiting times out.
            RuntimeError: If the worker returns an inference error.
        """
        request_id = uuid.uuid4().hex
        ring, slot, contiguous = await self._write_frame(request_id, frame)
        request: WorkerRequestPayload = {
            'id': request_id,
            'camera_id': self.camera_id,
            'model_key': model_key,
            'shm_name': ring.name,
            'slot': slot,
            'shape': contiguous.shape,
            'dtype': str(contiguous.dtype),
            'result_queue': cast(WorkerResultSender, self.result_queue),
        }
        request_submitted = False
        request_timed_out = False
        try:
            await self._submit_request(request)
            request_submitted = True
            return await self._wait_for_result(request_id)
        except TimeoutError:
            request_timed_out = request_submitted
            if request_timed_out:
                self._release_slot_later(request_id)
            raise
        finally:
            if not request_timed_out:
                self._release_slot(request_id)

    async def close(self) -> None:
        """Release this camera's rings when its stream process shuts down."""
        if self._closed:
            return
        self._closed = True
        rings = [*self._retired_rings]
        if self._ring is not None:
            rings.append(self._ring)
        self._retired_rings.clear()
        self._ring = None
        self._inflight_slots.clear()
        for ring in rings:
            ring.close()

    async def _write_frame(
        self,
        request_id: str,
        frame: FrameArray,
    ) -> tuple[_SharedFrameRing, int, FrameArray]:
        """Reserve a ring slot and copy one input frame into it."""
        contiguous = np.ascontiguousarray(frame)
        shape = tuple(int(value) for value in contiguous.shape)
        dtype = np.dtype(contiguous.dtype)
        async with self._ring_lock:
            if self._closed:
                raise RuntimeError('YOLO worker client is closed')
            if (
                self._ring is None
                or self._ring.shape != shape
                or self._ring.dtype != dtype
            ):
                if self._ring is not None:
                    self._retired_rings.append(self._ring)
                self._ring = _SharedFrameRing.create(
                    shape,
                    dtype,
                    self.ring_slots,
                )
            ring = self._ring
            slot = await ring.acquire_slot(self.timeout_seconds)
            ring.write(slot, contiguous)
            self._inflight_slots[request_id] = (ring, slot)
        return ring, slot, contiguous

    def _release_slot(self, request_id: str) -> None:
        """Release a result slot and reclaim retired rings when possible."""
        assignment = self._inflight_slots.pop(request_id, None)
        if assignment is None:
            return
        ring, slot = assignment
        ring.release_slot(slot)
        if ring is self._ring:
            return
        if any(
            candidate is ring
            for candidate, _ in self._inflight_slots.values()
        ):
            return
        with suppress(ValueError):
            self._retired_rings.remove(ring)
        ring.close()

    def _release_slot_later(self, request_id: str) -> None:
        """Free a timed-out slot after the worker's maximum grace period.

        Args:
            request_id: Timed-out request whose frame may still be in use.
        """
        async def release_when_safe() -> None:
            await asyncio.sleep(self.ring_cleanup_delay_seconds)
            self._release_slot(request_id)

        asyncio.create_task(release_when_safe())

    async def _submit_request(self, request: WorkerRequestPayload) -> None:
        """Submit one request payload without blocking the event loop.

        Args:
            request: Request payload to enqueue.

        Raises:
            TimeoutError: If the worker queue remains full.
        """
        try:
            await asyncio.to_thread(
                self.request_queue.put,
                request,
                True,
                self.timeout_seconds,
            )
        except queue.Full as exc:
            raise TimeoutError('YOLO worker request queue is full') from exc

    async def _wait_for_result(self, request_id: str) -> list[Detection]:
        """Wait for this camera's matching worker result.

        Args:
            request_id: Request identifier returned through the result queue.

        Returns:
            Detection rows returned by the worker.

        Raises:
            TimeoutError: If no result arrives before ``timeout_seconds``.
            RuntimeError: If the worker reports an error result.
        """
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('YOLO worker request timed out')
            try:
                result = cast(
                    WorkerResult,
                    await asyncio.to_thread(
                        self.result_queue.get,
                        True,
                        remaining,
                    ),
                )
            except queue.Empty as exc:
                raise TimeoutError('YOLO worker request timed out') from exc
            if result.get('id') != request_id:
                continue
            if result.get('ok'):
                return result.get('detections', [])
            raise RuntimeError(
                str(result.get('error', 'YOLO worker request failed')),
            )


class YoloWorker:
    """Shared-memory YOLO inference worker with latest-frame batching.

    The worker owns YOLO model instances and batches requests by model key. For
    each camera, only the newest pending frame is retained. That keeps latency
    bounded when many cameras produce frames faster than the GPU can process.
    """

    def __init__(
        self,
        request_queue: WorkerQueue | None,
        device: str | None = None,
        startup_lock: Any | None = None,
    ) -> None:
        """Initialise the worker.

        Args:
            request_queue: Queue from which request metadata is consumed.
            device: CUDA/CPU device string. Defaults to environment settings.
            startup_lock: Cross-process lock used to serialize an engine's
                first TensorRT inference.
        """
        self.request_queue = request_queue
        self.device = (
            device
            if isinstance(device, str)
            else os.getenv('YOLO_WORKER_DEVICE') or 'cuda:0'
        )
        self.startup_lock = startup_lock
        self.logger = logging.getLogger(__name__)
        self.model_cache: dict[str, YoloModelLike] = {}
        self.precision_mode = _parse_worker_precision(
            os.getenv('YOLO_WORKER_PRECISION'),
        )
        if self.precision_mode is None:
            self.model_dir = Path(
                os.getenv('YOLO_WORKER_MODEL_DIR', 'models/pt'),
            )
            self.model_suffix = os.getenv('YOLO_WORKER_MODEL_SUFFIX', '.pt')
            use_half = os.getenv(
                'YOLO_WORKER_HALF',
                'true',
            ).strip().lower() in {'1', 'true', 'yes', 'on'}
            quantize = parse_quantize_value(
                os.getenv('YOLO_WORKER_QUANTIZE'),
            )
            self.precision_args = precision_kwargs(use_half, quantize)
            self.precision_label = 'legacy'
        else:
            self.model_dir, self.model_suffix, self.precision_args = (
                _worker_precision_config(self.precision_mode)
            )
            self.precision_label = self.precision_mode
        self.pending: dict[str, _WorkerRequest] = {}
        self.batch_size = max(
            1,
            int(os.getenv('YOLO_WORKER_BATCH_SIZE', '8')),
        )
        self.batch_wait_seconds = max(
            0.0,
            int(os.getenv('YOLO_WORKER_BATCH_WAIT_MS', '10')) / 1000.0,
        )
        self.imgsz = int(os.getenv('YOLO_WORKER_IMGSZ', '640'))
        try:
            self.metrics_interval_seconds = max(
                0.0,
                float(
                    os.getenv(
                        'YOLO_WORKER_METRICS_INTERVAL_SECONDS',
                        '0',
                    ),
                ),
            )
        except ValueError:
            self.metrics_interval_seconds = 0.0
        self._metrics_started_at = time.monotonic()
        self._metrics_images = 0
        self._metrics_batches = 0
        self._metrics_predict_seconds = 0.0

    def run(self) -> None:
        """Run the worker loop until a stop message is received."""
        logging.basicConfig(level=logging.INFO)
        self.logger.info(
            '[YOLO-Worker] started on %s precision=%s model_dir=%s suffix=%s',
            self.device,
            self.precision_label,
            self.model_dir,
            self.model_suffix,
        )
        while True:
            if self.request_queue is None:
                raise RuntimeError(
                    'YOLO worker requires a request queue to run',
                )
            message: object | None = None
            timeout = 1.0 if not self.pending else self.batch_wait_seconds
            with suppress(queue.Empty):
                message = self.request_queue.get(timeout=timeout)
            if message == YOLO_WORKER_STOP_MESSAGE:
                self.logger.info('[YOLO-Worker] stopping')
                return
            if message is not None:
                self.store_latest_request(
                    _WorkerRequest.from_mapping(
                        cast(WorkerRequestPayload, message),
                    ),
                )
            self._drain_queue(
                deadline=time.monotonic() + self.batch_wait_seconds,
            )
            batch = self.pop_next_batch()
            if batch:
                self._handle_batch(batch)

    def _drain_queue(self, deadline: float) -> None:
        """Drain queued requests briefly to build a batch.

        Args:
            deadline: Monotonic time at which draining should stop.
        """
        if self.request_queue is None:
            raise RuntimeError('YOLO worker requires a request queue to drain')
        while len(self.pending) < self.batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            try:
                message = self.request_queue.get(timeout=remaining)
            except queue.Empty:
                return
            if message == YOLO_WORKER_STOP_MESSAGE:
                self.request_queue.put(YOLO_WORKER_STOP_MESSAGE)
                return
            request = cast(WorkerRequestPayload, message)
            self.store_latest_request(
                _WorkerRequest.from_mapping(request),
            )

    def store_latest_request(self, request: _WorkerRequest) -> None:
        """Keep only the newest pending request for each camera.

        Args:
            request: Request to store as the latest frame for its camera.
        """
        old_request = self.pending.get(request.camera_id)
        if old_request is not None:
            # A superseded request must return quickly so its client can keep
            # the stream alive without waiting for stale inference work.
            self._send_result(
                old_request, {
                    'id': old_request.id,
                    'ok': True,
                    'detections': [],
                    'skipped': True,
                },
            )
        self.pending[request.camera_id] = request

    def pop_next_batch(self) -> list[_WorkerRequest]:
        """Pop one model-homogeneous batch from pending requests.

        Returns:
            Pending requests that share the same model key, capped by
            ``batch_size``.
        """
        if not self.pending:
            return []
        selected_model = next(iter(self.pending.values())).model_key
        camera_ids: list[str] = []
        for camera_id, request in self.pending.items():
            if request.model_key != selected_model:
                continue
            camera_ids.append(camera_id)
            if len(camera_ids) >= self.batch_size:
                break
        return [self.pending.pop(camera_id) for camera_id in camera_ids]

    def _handle_batch(self, requests: list[_WorkerRequest]) -> None:
        """Read frames, run inference, and publish results.

        Args:
            requests: Batch candidate requests sharing a model key.
        """
        frames, valid_requests = self._read_batch_frames(requests)
        try:
            if not valid_requests:
                return
            model_key = valid_requests[0].model_key
            predict_started_at = time.monotonic()
            if model_key not in self.model_cache and self.startup_lock:
                with self.startup_lock:
                    model = self._get_model(model_key)
                    results = model.predict(
                        source=frames,
                        verbose=False,
                        device=self.device,
                        imgsz=self.imgsz,
                        batch=len(frames),
                        **self.precision_args,
                    )
            else:
                model = self._get_model(model_key)
                results = model.predict(
                    source=frames,
                    verbose=False,
                    device=self.device,
                    imgsz=self.imgsz,
                    batch=len(frames),
                    **self.precision_args,
                )
            self._record_batch_metrics(
                model_key=model_key,
                image_count=len(valid_requests),
                predict_seconds=time.monotonic() - predict_started_at,
            )
            for request, result in zip(valid_requests, results, strict=False):
                yolo_result = cast(YoloResultLike, result)
                box_data = yolo_result.boxes.data.cpu().numpy()
                detections: list[Detection] = []
                if box_data.size:
                    box_data = (
                        box_data.reshape(1, -1)
                        if box_data.ndim == 1
                        else box_data
                    )
                    score_index = -2 if box_data.shape[1] > 6 else 4
                    label_index = -1 if box_data.shape[1] > 6 else 5
                    detections = box_data[
                        :,
                        [0, 1, 2, 3, score_index, label_index],
                    ].astype(float, copy=False).tolist()
                    for detection in detections:
                        detection[5] = int(detection[5])
                self._send_result(
                    request, {
                        'id': request.id,
                        'ok': True,
                        'detections': detections,
                    },
                )
        except Exception as exc:
            self.logger.exception('[YOLO-Worker] batch request failed')
            for request in valid_requests:
                self._send_result(
                    request, {
                        'id': request.id,
                        'ok': False,
                        'error': str(exc),
                    },
                )
        finally:
            self._close_shared_frames(frames)

    def _record_batch_metrics(
        self,
        model_key: str,
        image_count: int,
        predict_seconds: float,
    ) -> None:
        """Log a compact rolling inference throughput summary when enabled."""
        if self.metrics_interval_seconds <= 0:
            return
        self._metrics_images += image_count
        self._metrics_batches += 1
        self._metrics_predict_seconds += predict_seconds
        elapsed = time.monotonic() - self._metrics_started_at
        if elapsed < self.metrics_interval_seconds:
            return
        average_batch = self._metrics_images / max(1, self._metrics_batches)
        average_predict_ms = (
            self._metrics_predict_seconds / max(1, self._metrics_batches)
        ) * 1000.0
        self.logger.info(
            '[YOLO-Worker] throughput model=%s images_per_second=%.1f '
            'average_batch=%.1f average_predict_ms=%.1f',
            model_key,
            self._metrics_images / elapsed,
            average_batch,
            average_predict_ms,
        )
        self._metrics_started_at = time.monotonic()
        self._metrics_images = 0
        self._metrics_batches = 0
        self._metrics_predict_seconds = 0.0

    def _read_batch_frames(
        self,
        requests: list[_WorkerRequest],
    ) -> tuple[list[FrameArray], list[_WorkerRequest]]:
        """Read shared-memory frames for a batch.

        Args:
            requests: Requests whose frames should be loaded.

        Returns:
            A pair of ``(frames, valid_requests)``. Unreadable requests are
            reported through their result queues and omitted.
        """
        frames: list[FrameArray] = []
        valid_requests: list[_WorkerRequest] = []
        for request in requests:
            try:
                frames.append(self._read_frame(request))
                valid_requests.append(request)
            except FileNotFoundError as exc:
                self.logger.warning(
                    '[YOLO-Worker] shared frame already removed: %s',
                    request.shm_name,
                )
                self._send_result(
                    request, {
                        'id': request.id,
                        'ok': False,
                        'error': str(exc),
                    },
                )
            except Exception as exc:
                self.logger.exception(
                    '[YOLO-Worker] failed to read shared frame',
                )
                self._send_result(
                    request, {
                        'id': request.id,
                        'ok': False,
                        'error': str(exc),
                    },
                )
        return frames, valid_requests

    def _send_result(
        self,
        request: _WorkerRequest,
        result: WorkerResult,
    ) -> None:
        """Return one result without an abandoned camera blocking IO."""
        try:
            request.result_queue.put(result, block=False)
        except queue.Full:
            self.logger.warning(
                '[YOLO-Worker] result queue full for %s; '
                'dropping stale result',
                request.camera_id,
            )

    @staticmethod
    def _read_frame(request: _WorkerRequest) -> FrameArray:
        """Map one frame from shared memory without an extra CPU copy.

        Args:
            request: Request containing shared memory metadata.

        Returns:
            A NumPy view valid until the batch inference finishes.

        Raises:
            FileNotFoundError: If the shared memory segment has already been
                removed.
        """
        shm = shared_memory.SharedMemory(name=request.shm_name)
        try:
            frame = np.ndarray(
                request.shape,
                dtype=np.dtype(request.dtype),
                buffer=shm.buf,
                offset=request.slot * int(
                    np.prod(request.shape, dtype=np.int64),
                ) * np.dtype(request.dtype).itemsize,
            ).view(_SharedFrameArray)
            frame.shared_memory_handle = shm
            return cast(FrameArray, frame)
        except Exception:
            shm.close()
            raise

    @staticmethod
    def _close_shared_frames(frames: list[FrameArray]) -> None:
        """Release shared-memory mappings after a batch has consumed them."""
        for frame in frames:
            if not isinstance(frame, _SharedFrameArray):
                continue
            shm = frame.shared_memory_handle
            if shm is None:
                continue
            frame.shared_memory_handle = None
            shm.close()

    def _get_model(self, model_key: str) -> YoloModelLike:
        """Return a cached YOLO model, loading it on first use.

        Args:
            model_key: Key used to construct the model filename.

        Returns:
            Cached Ultralytics YOLO instance.

        Raises:
            FileNotFoundError: If the configured model file does not exist.
        """
        model = self.model_cache.get(model_key)
        if model is not None:
            return model
        model_path = self.model_dir / f'best_{model_key}{self.model_suffix}'
        if not model_path.exists():
            raise FileNotFoundError(
                f'YOLO worker model not found: {model_path}',
            )
        from ultralytics import YOLO

        # TensorRT engines do not carry enough task metadata for every
        # Ultralytics release to infer detection reliably.
        model = cast(YoloModelLike, YOLO(str(model_path), task='detect'))
        self.model_cache[model_key] = model
        self.logger.info('[YOLO-Worker] loaded %s', model_key)
        return model
