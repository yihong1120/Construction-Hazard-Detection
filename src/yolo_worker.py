from __future__ import annotations

import asyncio
import logging
import os
import queue
import time
import uuid
from collections.abc import Mapping
from contextlib import suppress
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np

from src.ultralytics_args import parse_quantize_value
from src.ultralytics_args import precision_kwargs
from src.yolo_shared_frames import _SharedFrameArray
from src.yolo_shared_frames import _SharedFrameRing
from src.yolo_worker_protocol import _parse_worker_precision
from src.yolo_worker_protocol import _worker_precision_config
from src.yolo_worker_protocol import _WorkerRequest
from src.yolo_worker_protocol import Detection
from src.yolo_worker_protocol import FrameArray
from src.yolo_worker_protocol import WorkerQueue
from src.yolo_worker_protocol import WorkerRequestPayload
from src.yolo_worker_protocol import WorkerRequestSender
from src.yolo_worker_protocol import WorkerResult
from src.yolo_worker_protocol import WorkerResultReceiver
from src.yolo_worker_protocol import WorkerResultSender
from src.yolo_worker_protocol import YOLO_WORKER_STOP_MESSAGE
from src.yolo_worker_protocol import YoloModelLike
from src.yolo_worker_protocol import YoloResultLike
# Compatibility exports for callers that historically imported all IPC types
# from this runtime module.


class YoloWorkerClient:
    """Asynchronous client used by stream processes for YOLO inference.

    A fixed, per-camera shared-memory ring avoids allocating one POSIX segment
    per frame. Requests contain only a ring slot descriptor; results return on
    a camera-specific queue so no stream needs to poll a shared response queue.
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
            """Perform release when safe.
            """
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
        result_queues: Mapping[str, WorkerResultSender] | None = None,
        startup_lock: Any | None = None,
    ) -> None:
        """Initialise the worker.

        Args:
            request_queue: Queue from which request metadata is consumed.
            device: CUDA/CPU device string. Defaults to environment settings.
            result_queues: Fixed per-camera queues created before worker start.
            startup_lock: Cross-process lock used to serialize an engine's
                first TensorRT inference.
        """
        self.request_queue = request_queue
        self.device = (
            device
            if isinstance(device, str)
            else os.getenv('YOLO_WORKER_DEVICE') or 'cuda:0'
        )
        self.result_queues = dict(result_queues or {})
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
        result_queue = self.result_queues.get(request.camera_id)
        if result_queue is None:
            self.logger.warning(
                '[YOLO-Worker] missing result queue for %s',
                request.camera_id,
            )
            return
        try:
            result_queue.put(result, block=False)
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
