from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
import time
import uuid
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import cast
from typing import Protocol
from typing import Self
from typing import TypeAlias
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


YOLO_WORKER_STOP_MESSAGE = '__stop__'

Detection: TypeAlias = list[float]
FrameArray: TypeAlias = NDArray[np.uint8]


class WorkerRequestPayload(TypedDict):
    """Serialisable request payload passed through the worker queue."""

    id: str
    camera_id: str
    model_key: str
    shm_name: str
    shape: tuple[int, ...]
    dtype: str


class WorkerResult(TypedDict, total=False):
    """Serialisable result payload shared between client and worker."""

    ok: bool
    detections: list[Detection]
    error: str
    expired: bool
    skipped: bool
    shm_name: str


class ResultStore(Protocol):
    """Shared result mapping used by worker processes and clients."""

    def __setitem__(self, key: str, value: WorkerResult) -> None:
        """Store a worker result by request id."""

    def get(
        self,
        key: str,
        default: WorkerResult | None = None,
    ) -> WorkerResult | None:
        """Return a worker result without removing it."""

    def pop(
        self,
        key: str,
        default: WorkerResult | None = None,
    ) -> WorkerResult | None:
        """Return and remove a worker result if present."""


class WorkerQueue(Protocol):
    """Small queue interface used by the client and worker."""

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
        half: bool,
        batch: int,
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
            shape=tuple(int(v) for v in request['shape']),
            dtype=str(request['dtype']),
        )


class YoloWorkerClient:
    """Asynchronous client used by stream processes for YOLO inference.

    The client copies a frame into POSIX shared memory, submits only metadata
    through a multiprocessing queue, and waits for a result in the shared
    result store. If a request times out, the shared memory is kept briefly.
    That lets the worker notice the cancellation and clean it up without noisy
    ``FileNotFoundError`` traces.
    """

    def __init__(
        self,
        request_queue: WorkerQueue,
        result_store: ResultStore,
        camera_id: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Initialise the client.

        Args:
            request_queue: Queue used to submit frame metadata.
            result_store: Shared mapping used to receive worker results.
            camera_id: Stable site/camera key for latest-frame coalescing.
            timeout_seconds: Maximum time to wait for queue submission and
                inference result.
        """
        self.request_queue = request_queue
        self.result_store = result_store
        self.camera_id = camera_id
        self.timeout_seconds = timeout_seconds
        self.shm_cleanup_delay_seconds = max(
            timeout_seconds,
            float(os.getenv('YOLO_WORKER_SHM_CLEANUP_DELAY_SECONDS', '120.0')),
        )

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
        contiguous = np.ascontiguousarray(frame)
        shm = shared_memory.SharedMemory(create=True, size=contiguous.nbytes)
        request_id = uuid.uuid4().hex
        shared_array: FrameArray = np.ndarray(
            contiguous.shape,
            dtype=contiguous.dtype,
            buffer=shm.buf,
        )
        shared_array[:] = contiguous
        request: WorkerRequestPayload = {
            'id': request_id,
            'camera_id': self.camera_id,
            'model_key': model_key,
            'shm_name': shm.name,
            'shape': contiguous.shape,
            'dtype': str(contiguous.dtype),
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
                self.result_store[request_id] = {
                    'ok': False,
                    'expired': True,
                    'shm_name': shm.name,
                }
                self._unlink_shared_memory_later(shm.name)
            raise
        finally:
            shm.close()
            if not request_timed_out:
                with suppress(FileNotFoundError):
                    shm.unlink()
                self.result_store.pop(request_id, None)

    def _unlink_shared_memory_later(self, shm_name: str) -> None:
        """Schedule delayed shared memory cleanup.

        Args:
            shm_name: Shared memory segment name to unlink later.
        """
        # The timer is a final safety net. Normally the worker sees the expired
        # request first and unlinks the segment itself.
        timer = threading.Timer(
            self.shm_cleanup_delay_seconds,
            _unlink_shared_memory,
            args=(shm_name,),
        )
        timer.daemon = True
        timer.start()

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
        """Wait for a worker result.

        Args:
            request_id: Request identifier to poll in the result store.

        Returns:
            Detection rows returned by the worker.

        Raises:
            TimeoutError: If no result arrives before ``timeout_seconds``.
            RuntimeError: If the worker reports an error result.
        """
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            result = self.result_store.pop(request_id, None)
            if result is None:
                await asyncio.sleep(0.002)
                continue
            if result.get('ok'):
                return result.get('detections', [])
            raise RuntimeError(
                str(result.get('error', 'YOLO worker request failed')),
            )
        raise TimeoutError('YOLO worker request timed out')


class YoloWorker:
    """Shared-memory YOLO inference worker with latest-frame batching.

    The worker owns YOLO model instances and batches requests by model key. For
    each camera, only the newest pending frame is retained. That keeps latency
    bounded when many cameras produce frames faster than the GPU can process.
    """

    def __init__(
        self,
        request_queue: WorkerQueue,
        result_store: ResultStore,
        device: str | None = None,
    ) -> None:
        """Initialise the worker.

        Args:
            request_queue: Queue from which request metadata is consumed.
            result_store: Shared mapping where worker results are written.
            device: CUDA/CPU device string. Defaults to environment settings.
        """
        self.request_queue = request_queue
        self.result_store = result_store
        self.device = device or os.getenv('YOLO_WORKER_DEVICE') or 'cuda:0'
        self.logger = logging.getLogger(__name__)
        self.model_cache: dict[str, YoloModelLike] = {}
        self.model_dir = Path(os.getenv('YOLO_WORKER_MODEL_DIR', 'models/pt'))
        self.model_suffix = os.getenv('YOLO_WORKER_MODEL_SUFFIX', '.pt')
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
        self.use_half = os.getenv(
            'YOLO_WORKER_HALF',
            'true',
        ).strip().lower() in {'1', 'true', 'yes', 'on'}

    def run(self) -> None:
        """Run the worker loop until a stop message is received."""
        logging.basicConfig(level=logging.INFO)
        self.logger.info('[YOLO-Worker] started on %s', self.device)
        while True:
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
        while (
            len(self.pending) < self.batch_size
            and time.monotonic() < deadline
        ):
            try:
                message = self.request_queue.get_nowait()
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
            self.result_store[old_request.id] = {
                'ok': True,
                'detections': [],
                'skipped': True,
            }
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
        if not valid_requests:
            return
        try:
            model = self._get_model(valid_requests[0].model_key)
            results = model.predict(
                source=frames,
                verbose=False,
                device=self.device,
                imgsz=self.imgsz,
                half=self.use_half,
                batch=len(frames),
            )
        except Exception as exc:
            self.logger.exception('[YOLO-Worker] batch request failed')
            for request in valid_requests:
                self.result_store[request.id] = {
                    'ok': False,
                    'error': str(exc),
                }
            return
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
            self.result_store[request.id] = {
                'ok': True,
                'detections': detections,
            }

    def _read_batch_frames(
        self,
        requests: list[_WorkerRequest],
    ) -> tuple[list[FrameArray], list[_WorkerRequest]]:
        """Read shared-memory frames for a batch.

        Args:
            requests: Requests whose frames should be loaded.

        Returns:
            A pair of ``(frames, valid_requests)``. Expired or unreadable
            requests are reported through ``result_store`` and omitted.
        """
        frames: list[FrameArray] = []
        valid_requests: list[_WorkerRequest] = []
        for request in requests:
            if self._request_expired(request):
                _unlink_shared_memory(request.shm_name)
                self.result_store.pop(request.id, None)
                continue
            try:
                frames.append(self._read_frame(request))
                valid_requests.append(request)
            except FileNotFoundError as exc:
                self.logger.warning(
                    '[YOLO-Worker] shared frame already removed: %s',
                    request.shm_name,
                )
                self.result_store[request.id] = {
                    'ok': False,
                    'error': str(exc),
                }
            except Exception as exc:
                self.logger.exception(
                    '[YOLO-Worker] failed to read shared frame',
                )
                self.result_store[request.id] = {
                    'ok': False,
                    'error': str(exc),
                }
        return frames, valid_requests

    def _request_expired(self, request: _WorkerRequest) -> bool:
        """Return whether the client already timed out this request.

        Args:
            request: Request to inspect.

        Returns:
            ``True`` when the result store marks the request as expired.
        """
        result = self.result_store.get(request.id)
        return bool(result and result.get('expired'))

    @staticmethod
    def _read_frame(request: _WorkerRequest) -> FrameArray:
        """Copy one frame from shared memory.

        Args:
            request: Request containing shared memory metadata.

        Returns:
            A private NumPy copy of the frame.

        Raises:
            FileNotFoundError: If the shared memory segment has already been
                removed.
        """
        shm = shared_memory.SharedMemory(name=request.shm_name)
        try:
            frame: FrameArray = np.ndarray(
                request.shape,
                dtype=np.dtype(request.dtype),
                buffer=shm.buf,
            )
            return frame.copy()
        finally:
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

        model = cast(YoloModelLike, YOLO(str(model_path)))
        self.model_cache[model_key] = model
        self.logger.info('[YOLO-Worker] loaded %s', model_key)
        return model


def _unlink_shared_memory(shm_name: str) -> None:
    """Unlink one shared memory segment if it still exists.

    Args:
        shm_name: POSIX shared memory segment name.
    """
    with suppress(FileNotFoundError):
        shm = shared_memory.SharedMemory(name=shm_name)
        try:
            shm.unlink()
        finally:
            shm.close()
