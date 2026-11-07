from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from contextlib import nullcontext
from dataclasses import dataclass
from dataclasses import field

from fastapi import HTTPException

from examples.YOLO_server_api.detection import DetectionPolicy
from examples.YOLO_server_api.detection import run_detection
from examples.YOLO_server_api.image_input import DecodedImageBudget
from examples.YOLO_server_api.image_input import prepare_image
from examples.YOLO_server_api.model_registry import ModelEntry
from examples.YOLO_server_api.models import DetectionModelManager
from src.settings.inference import ApiInferenceSettings


@dataclass(frozen=True)
class InferenceResult:
    rows: list[list[float | int]]
    timing: dict[str, float]
    image_size: tuple[int, int]


@dataclass(frozen=True)
class ModelSelection:
    entry: ModelEntry
    runtime: InferenceRuntime

    async def detect(self, data: bytes):
        return await self.runtime.infer(self.entry, data)


@dataclass
class ModelGate:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


class InferenceRuntime:
    """Schedule model ownership before consuming a native worker thread."""

    def __init__(
        self,
        models: DetectionModelManager | None = None,
        settings: ApiInferenceSettings | None = None,
    ):
        self.settings = settings or ApiInferenceSettings()
        self.models = models or DetectionModelManager(
            self.settings.max_loaded_models,
        )
        self._lock = threading.Lock()
        self._admitted = 0
        self._closed = False
        self._workers = asyncio.Semaphore(self.settings.concurrency)
        self.decoded = DecodedImageBudget(self.settings.max_decoded_bytes)
        self._gates: dict[tuple[str, str, str], ModelGate] = {}
        self._requests: set[InferenceRequest] = set()
        self.executor = ThreadPoolExecutor(
            max_workers=self.settings.concurrency,
            thread_name_prefix='inference',
        )

    @asynccontextmanager
    async def admission(self):
        with self._lock:
            if self._closed or self._admitted >= self.settings.queue_capacity:
                raise HTTPException(503, detail={'code': 'INFERENCE_BUSY'})
            self._admitted += 1
        request = InferenceRequest(self)
        self._requests.add(request)
        try:
            yield request
        finally:
            # Once submitted, native completion owns all permits and leases.
            if request.future is None:
                request.release()

    async def prepare(self, entry: ModelEntry) -> ModelSelection:
        async with self.admission() as request:
            await request.run(entry, None)
        return ModelSelection(entry, self)

    async def infer(self, entry: ModelEntry, data: bytes):
        async with self.admission() as request:
            return await request.run(entry, data)

    async def close(self) -> None:
        with self._lock:
            self._closed = True

        cleanup = asyncio.create_task(self._finish())
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise

    def _release_native(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.models.close()

    async def _finish(self):
        waiters = []
        for request in tuple(self._requests):
            if request.future is None and request.task is not None:
                request.task.cancel()
                waiters.append(request.task)
        await asyncio.gather(*waiters, return_exceptions=True)
        await asyncio.to_thread(self._release_native)


class InferenceRequest:
    def __init__(self, runtime: InferenceRuntime):
        self.runtime = runtime
        self.started = False
        self.released = False
        self.cancelled = threading.Event()
        self.future: Future | None = None
        self.task: asyncio.Task | None = None
        self.key: tuple[str, str, str] | None = None
        self.gate: ModelGate | None = None
        self.owns_gate = False
        self.owns_worker = False

    def release(self):
        if self.released:
            return
        self.released = True
        if self.owns_worker:
            self.runtime._workers.release()
        if self.gate is not None:
            if self.owns_gate:
                self.gate.lock.release()
            self.gate.users -= 1
            if not self.gate.users:
                assert self.key is not None
                self.runtime._gates.pop(self.key, None)
        self.runtime._requests.discard(self)
        with self.runtime._lock:
            self.runtime._admitted -= 1

    async def run(self, entry: ModelEntry, data: bytes | None):
        if self.started or self.released:
            raise RuntimeError('Inference request was already submitted')
        if (
            data is not None
            and len(data) > self.runtime.settings.max_upload_bytes
        ):
            raise HTTPException(
                413,
                detail={'code': 'IMAGE_UPLOAD_LIMIT_EXCEEDED'},
            )
        self.started = True
        self.task = asyncio.current_task()
        self.key = (entry.id, entry.version, entry.sha256)
        self.gate = self.runtime._gates.setdefault(self.key, ModelGate())
        self.gate.users += 1

        try:
            await self.gate.lock.acquire()
            self.owns_gate = True
            await self.runtime._workers.acquire()
            self.owns_worker = True
            if self.runtime._closed:
                raise HTTPException(503, detail={'code': 'INFERENCE_BUSY'})
            self.future = self.runtime.executor.submit(self._work, entry, data)
            result = asyncio.wrap_future(self.future)

            result.add_done_callback(self._completed)
            return await asyncio.shield(result)
        except asyncio.CancelledError:
            self.cancelled.set()
            if self.future is not None:
                self.future.cancel()
            raise
        finally:
            if self.future is None:
                self.release()

    def _work(self, entry: ModelEntry, data: bytes | None):
        prepared = (
            prepare_image(data, self.runtime.settings.max_image_pixels)
            if data is not None
            else None
        )
        lease = (
            self.runtime.decoded.reserve(prepared.decoded_bytes)
            if prepared is not None
            else nullcontext()
        )
        with lease:
            frame = prepared.decode() if prepared is not None else None
            try:
                with self.runtime.models.lease(
                    entry,
                    self.cancelled,
                ) as model:
                    if frame is None:
                        return None
                    rows, timing = run_detection(
                        frame,
                        model,
                        DetectionPolicy.from_classes(entry.classes),
                    )
                    return InferenceResult(
                        rows,
                        timing,
                        (frame.shape[1], frame.shape[0]),
                    )
            finally:
                del frame

    def _completed(self, done):
        self.release()
        if not done.cancelled():
            done.exception()
