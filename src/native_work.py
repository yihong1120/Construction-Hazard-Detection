from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from fastapi import HTTPException

from src.async_tasks import finish_on_cancel
from src.async_tasks import observe_task_exception


class NativeWorkPool:
    """Hold admission until native completion, even after disconnect."""

    def __init__(
        self,
        concurrency=2,
        capacity=8,
        *,
        code='NATIVE_WORK_BUSY',
        name='native',
    ):
        if concurrency < 1 or capacity < concurrency:
            raise ValueError('Invalid native work capacity')
        self.code = code
        self.capacity = capacity
        self._lock = threading.Lock()
        self._pending = 0
        self._closed = False
        self.executor = ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix=name,
        )

    async def run[T](self, work: Callable[[], T]) -> T:
        item = _NativeWork(work)

        with self._lock:
            if self._closed or self._pending >= self.capacity:
                raise HTTPException(503, detail={'code': self.code})
            self._pending += 1
            try:
                future = self.executor.submit(item.run)
            except BaseException:
                self._pending -= 1
                raise

        future.add_done_callback(self._completed)
        result = asyncio.wrap_future(future)
        result.add_done_callback(observe_task_exception)
        try:
            return await asyncio.shield(result)
        except asyncio.CancelledError:
            # Keep queued admission until its work item is actually dequeued.
            # Cancelling its Future would leave an unbounded cancelled backlog
            # in ThreadPoolExecutor while immediately admitting replacements.
            item.cancelled.set()
            raise

    def _completed(self, _):
        with self._lock:
            self._pending -= 1

    async def close(self):
        with self._lock:
            self._closed = True
        await finish_on_cancel(
            asyncio.to_thread(
                self.executor.shutdown,
                wait=True,
                cancel_futures=True,
            ),
        )


class _NativeWork[T]:
    def __init__(self, work: Callable[[], T]):
        self.work = work
        self.cancelled = threading.Event()

    def run(self) -> T:
        if self.cancelled.is_set():
            raise asyncio.CancelledError
        return self.work()
