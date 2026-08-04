from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.gpu_stream_capture import GpuFrame
from src.gpu_stream_capture import GpuLetterbox
from src.ultralytics_args import parse_quantize_value
from src.ultralytics_args import precision_kwargs
from src.ultralytics_args import PrecisionValue
from src.yolo_worker import _parse_worker_precision
from src.yolo_worker import _worker_precision_config
from src.yolo_worker import Detection


@dataclass(frozen=True)
class _GpuInferenceRequest:
    """One CUDA frame waiting for the shared model cache."""

    camera_id: str
    model_key: str
    frame: GpuFrame | np.ndarray
    result: asyncio.Future[list[Detection]]


class GpuYoloWorkerClient:
    """Camera-scoped client for one in-process GPU batch worker."""

    def __init__(
        self,
        worker: GpuYoloBatcher,
        camera_id: str,
    ) -> None:
        self.worker = worker
        self.camera_id = camera_id

    async def detect(
        self,
        frame: GpuFrame | np.ndarray,
        model_key: str,
    ) -> list[Detection]:
        """Submit one NVDEC frame or a CPU fallback frame for inference."""
        return await self.worker.detect(
            camera_id=self.camera_id,
            frame=frame,
            model_key=model_key,
        )


class GpuYoloBatcher:
    """Batch CUDA frames from many cameras through a shared YOLO cache.

    This object must stay inside one process. CUDA tensor IPC would otherwise
    introduce copies and lifetime hazards that remove the NVDEC benefit.
    """

    def __init__(
        self,
        *,
        device: str | None = None,
        batch_size: int | None = None,
        batch_wait_seconds: float | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.device = device or os.getenv('GPU_DECODE_DEVICE', 'cuda:0')
        precision_mode = _parse_worker_precision(
            os.getenv('YOLO_WORKER_PRECISION'),
        )
        if precision_mode is None:
            self.model_dir = Path(
                os.getenv('YOLO_WORKER_MODEL_DIR', 'models/pt'),
            )
            self.model_suffix = os.getenv('YOLO_WORKER_MODEL_SUFFIX', '.pt')
            use_half = os.getenv(
                'YOLO_WORKER_HALF',
                'true',
            ).strip().lower() in {'1', 'true', 'yes', 'on'}
            self.precision_args: dict[str, PrecisionValue] = precision_kwargs(
                use_half,
                parse_quantize_value(os.getenv('YOLO_WORKER_QUANTIZE')),
            )
        else:
            (
                self.model_dir,
                self.model_suffix,
                self.precision_args,
            ) = _worker_precision_config(precision_mode)

        self.imgsz = int(os.getenv('YOLO_WORKER_IMGSZ', '640'))
        self.use_half = self._uses_half_precision()
        self.batch_size = max(
            1,
            batch_size
            or int(
                os.getenv(
                    'GPU_DECODE_BATCH_SIZE',
                    os.getenv('YOLO_WORKER_BATCH_SIZE', '8'),
                ),
            ),
        )
        self.batch_wait_seconds = max(
            0.0,
            batch_wait_seconds
            if batch_wait_seconds is not None
            else int(
                os.getenv(
                    'GPU_DECODE_BATCH_WAIT_MS',
                    os.getenv('YOLO_WORKER_BATCH_WAIT_MS', '10'),
                ),
            )
            / 1000.0,
        )
        self.timeout_seconds = max(
            0.1,
            timeout_seconds
            or float(os.getenv('YOLO_WORKER_TIMEOUT_SECONDS', '30.0')),
        )
        self.max_pending_cameras = max(
            1,
            int(os.getenv('GPU_DECODE_QUEUE_SIZE', '64')),
        )
        self.logger = logging.getLogger(__name__)
        self.model_cache: dict[str, Any] = {}
        self._pending: dict[str, _GpuInferenceRequest] = {}
        self._pending_event = asyncio.Event()
        self._pending_results: set[asyncio.Future[list[Detection]]] = set()
        self._actor_task: asyncio.Task[None] | None = None
        self._closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix='gpu-yolo-batcher',
        )

    async def start(self) -> None:
        """Start the single inference actor for this process."""
        if self._closed:
            raise RuntimeError('GPU YOLO batcher has already been closed')
        if self._actor_task is None:
            self._actor_task = asyncio.create_task(self._run())
            self.logger.info(
                '[GPU-YOLO] shared batch worker started on %s '
                '(batch_size=%s, wait_ms=%s)',
                self.device,
                self.batch_size,
                int(self.batch_wait_seconds * 1000),
            )

    def client(self, camera_id: str) -> GpuYoloWorkerClient:
        """Return a client that preserves a camera's tracking identity."""
        return GpuYoloWorkerClient(self, camera_id)

    async def detect(
        self,
        *,
        camera_id: str,
        frame: GpuFrame | np.ndarray,
        model_key: str,
    ) -> list[Detection]:
        """Keep this camera's latest frame and wait for its batch result."""
        if self._closed:
            raise RuntimeError('GPU YOLO batcher is not running')
        await self.start()
        result: asyncio.Future[list[Detection]] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending_results.add(result)
        request = _GpuInferenceRequest(
            camera_id=camera_id,
            model_key=model_key,
            frame=frame,
            result=result,
        )
        try:
            self._store_latest_request(request)
            return await asyncio.wait_for(
                asyncio.shield(result),
                timeout=self.timeout_seconds,
            )
        finally:
            self._pending_results.discard(result)
            if not result.done():
                result.cancel()

    async def close(self) -> None:
        """Stop pending work and release the single inference thread."""
        self._closed = True
        actor_task = self._actor_task
        self._actor_task = None
        if actor_task is not None:
            actor_task.cancel()
            try:
                await actor_task
            except asyncio.CancelledError:
                pass
        for result in self._pending_results:
            if not result.done():
                result.set_exception(
                    RuntimeError('GPU YOLO batcher stopped before inference'),
                )
        self._pending_results.clear()
        self._pending.clear()
        self._pending_event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def _run(self) -> None:
        """Collect a model-homogeneous batch then infer it on one thread."""
        while True:
            requests = await self._next_batch()
            if not requests:
                continue
            try:
                detections = await asyncio.get_running_loop().run_in_executor(
                    self._executor,
                    self._infer_batch,
                    requests,
                )
            except Exception as exc:
                self.logger.exception('[GPU-YOLO] batch inference failed')
                for request in requests:
                    self._set_exception(request.result, exc)
                continue
            for request, camera_detections in zip(
                requests,
                detections,
                strict=True,
            ):
                if not request.result.done():
                    request.result.set_result(camera_detections)

    async def _next_batch(self) -> list[_GpuInferenceRequest]:
        """Wait briefly for one same-model batch of latest camera frames."""
        await self._wait_for_pending_requests()
        if not self._pending:
            return []
        model_key = next(iter(self._pending.values())).model_key
        deadline = time.monotonic() + self.batch_wait_seconds
        while self._pending_count(model_key) < self.batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._pending_event.clear()
            if self._pending_count(model_key) >= self.batch_size:
                break
            try:
                await asyncio.wait_for(
                    self._pending_event.wait(),
                    timeout=remaining,
                )
            except TimeoutError:
                break

        self._discard_cancelled_requests()
        selected: list[_GpuInferenceRequest] = []
        for camera_id, request in list(self._pending.items()):
            if (
                request.model_key == model_key
                and len(selected) < self.batch_size
            ):
                selected.append(request)
                self._pending.pop(camera_id, None)
        return selected

    def _store_latest_request(self, request: _GpuInferenceRequest) -> None:
        """Keep one pending frame per camera and skip an overwritten frame."""
        self._discard_cancelled_requests()
        previous = self._pending.get(request.camera_id)
        if previous is not None:
            self._set_result(previous.result, [])
        elif len(self._pending) >= self.max_pending_cameras:
            raise TimeoutError('GPU YOLO pending camera limit reached')
        self._pending[request.camera_id] = request
        self._pending_event.set()

    async def _wait_for_pending_requests(self) -> None:
        """Wait for at least one non-cancelled camera request."""
        while True:
            self._discard_cancelled_requests()
            if self._pending:
                return
            self._pending_event.clear()
            if self._pending:
                continue
            await self._pending_event.wait()

    def _pending_count(self, model_key: str) -> int:
        """Return pending latest frames that can join one model batch."""
        return sum(
            request.model_key == model_key
            for request in self._pending.values()
        )

    def _discard_cancelled_requests(self) -> None:
        """Remove timed-out clients before they consume GPU work."""
        self._pending = {
            camera_id: request
            for camera_id, request in self._pending.items()
            if not request.result.cancelled()
        }

    def _infer_batch(
        self,
        requests: list[_GpuInferenceRequest],
    ) -> list[list[Detection]]:
        """Preprocess CUDA frames and run one synchronous YOLO prediction."""
        prepared_frames: list[torch.Tensor] = []
        letterboxes: list[GpuLetterbox] = []
        for request in requests:
            prepared, letterbox = self._prepare_frame(request.frame)
            prepared_frames.append(prepared)
            letterboxes.append(letterbox)

        model = self._get_model(requests[0].model_key)
        batch = torch.cat(prepared_frames, dim=0)
        results = list(
            model.predict(
                source=batch,
                verbose=False,
                device=self.device,
                imgsz=self.imgsz,
                batch=len(requests),
                **self.precision_args,
            ),
        )
        if len(results) != len(requests):
            raise RuntimeError(
                'YOLO returned a result count that does not match its batch',
            )
        return [
            self._detections_from_result(result, letterbox)
            for result, letterbox in zip(results, letterboxes, strict=True)
        ]

    def _prepare_frame(
        self,
        frame: GpuFrame | np.ndarray,
    ) -> tuple[torch.Tensor, GpuLetterbox]:
        """Prepare a GPU frame or upload one CPU fallback frame once."""
        if isinstance(frame, GpuFrame):
            return frame.prepare_for_yolo(self.imgsz, self.use_half)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                'CPU fallback frame must be BGR shaped [height, width, 3]',
            )
        host_frame = np.ascontiguousarray(frame)
        if not host_frame.flags.writeable:
            host_frame = host_frame.copy()
        bgr_tensor = torch.from_numpy(host_frame).permute(2, 0, 1)
        rgb_tensor = bgr_tensor.to(
            device=self.device,
            non_blocking=True,
        )[[2, 1, 0]]
        return GpuFrame(
            tensor=rgb_tensor,
            timestamp=0.0,
        ).prepare_for_yolo(self.imgsz, self.use_half)

    @staticmethod
    def _detections_from_result(
        result: object,
        letterbox: GpuLetterbox,
    ) -> list[Detection]:
        """Convert one Ultralytics result back to source-frame coordinates."""
        boxes = getattr(result, 'boxes', None)
        if boxes is None or len(boxes) == 0:
            return []
        box_data = boxes.data
        if hasattr(box_data, 'cpu'):
            box_data = box_data.cpu()
        rows = letterbox.restore_rows(box_data.tolist())
        detections: list[Detection] = []
        for row in rows:
            score_index = -2 if len(row) > 6 else 4
            label_index = -1 if len(row) > 6 else 5
            detections.append(
                [
                    float(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[score_index]),
                    int(row[label_index]),
                ],
            )
        return detections

    def _uses_half_precision(self) -> bool:
        """Return whether CUDA preprocessing should produce float16 tensors."""
        value = self.precision_args.get(
            'quantize',
            self.precision_args.get('half'),
        )
        return value in {True, 16, '16', 'fp16', 'f16', 'w16a16'}

    def _get_model(self, model_key: str) -> Any:
        """Load one model per key and retain it for every camera using it."""
        model = self.model_cache.get(model_key)
        if model is not None:
            return model
        model_path = self.model_dir / f'best_{model_key}{self.model_suffix}'
        if not model_path.exists():
            raise FileNotFoundError(
                f'GPU YOLO model not found: {model_path}',
            )
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        self.model_cache[model_key] = model
        self.logger.info('[GPU-YOLO] loaded %s', model_key)
        return model

    @staticmethod
    def _set_exception(
        result: asyncio.Future[list[Detection]],
        exc: Exception,
    ) -> None:
        """Report an actor error without overwriting an expired request."""
        if not result.done():
            result.set_exception(exc)

    @staticmethod
    def _set_result(
        result: asyncio.Future[list[Detection]],
        detections: list[Detection],
    ) -> None:
        """Complete a superseded request without spending GPU work on it."""
        if not result.done():
            result.set_result(detections)
