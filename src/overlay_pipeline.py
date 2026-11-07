from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.stream_overlay_frames import OverlayFrameCache
from src.stream_runtime_state import OverlaySnapshot

logger = logging.getLogger(__name__)


class OverlayRenderPool:
    def __init__(self, cache: OverlayFrameCache) -> None:
        self.cache = cache
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix='overlay',
        )

    async def render(
        self,
        snapshot: OverlaySnapshot | None,
        languages: set[str],
    ) -> dict[str, np.ndarray]:

        future = asyncio.get_running_loop().run_in_executor(
            self.executor,
            self._draw,
            snapshot,
            languages,
        )
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # A running draw owns the cache until it finishes.
            await future
            raise

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

    def _draw(self, snapshot: OverlaySnapshot | None, languages: set[str]):
        self.cache.prune(languages)
        return (
            self.cache.render(
                snapshot,
                languages,
            )
            if snapshot is not None
            else {}
        )


class LatestOverlayPublication:

    def __init__(
        self,
        publish: Callable[
            [np.ndarray],
            Awaitable[None],
        ],
    ) -> None:
        self.publish = publish
        self.queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=1)
        self.task = asyncio.create_task(self._run())

    def offer(self, frame: np.ndarray) -> None:
        if self.queue.full():
            self.queue.get_nowait()
        self.queue.put_nowait(frame)

    async def _run(self) -> None:
        while True:
            frame = await self.queue.get()
            try:
                await self.publish(frame)
            except Exception:
                logger.exception('Overlay publication failed')
                await asyncio.sleep(0.5)

    async def close(self) -> None:
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
