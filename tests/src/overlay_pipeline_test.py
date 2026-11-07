from __future__ import annotations

import asyncio
import threading
from functools import partial
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.overlay_pipeline import LatestOverlayPublication
from src.overlay_pipeline import OverlayRenderPool


async def _slow_language_does_not_block_run():
    _callbacks = _SlowLanguageDoesNotBlockOthersAndCallbacks()
    _callbacks.entered = asyncio.Event()
    _callbacks.release = asyncio.Event()
    fast_done = asyncio.Event()
    _callbacks.slow_frames = []

    slow = _callbacks._call_slow

    fast = partial(_slow_language_does_not_block_fast, fast_done)
    first, second = (
        LatestOverlayPublication(
            slow,
        ),
        LatestOverlayPublication(fast),
    )
    try:
        first.offer(np.array([1]))
        await asyncio.wait_for(_callbacks.entered.wait(), 1)
        first.offer(np.array([2]))
        first.offer(np.array([3]))
        second.offer(np.array([1]))
        await asyncio.wait_for(fast_done.wait(), 1)
        assert _callbacks.slow_frames == [1]
        _callbacks.release.set()
        await asyncio.sleep(0)
        assert _callbacks.slow_frames == [1, 3]
        assert first.queue.maxsize == 1
    finally:
        await first.close()
        await second.close()
    assert first.task.done() and second.task.done()


async def _slow_language_does_not_block_fast(fast_done, /, frame):
    fast_done.set()


async def _render_thread_cancellation_preserves_cache_run():
    _callbacks = _RenderThreadCancellationPreservesCacheOwnershipRunCallbacks()
    _callbacks.entered, _callbacks.release = (
        threading.Event(),
        threading.Event(),
    )
    _callbacks.loop_thread = threading.get_ident()

    render = _callbacks._call_render
    cache = SimpleNamespace(prune=MagicMock(), render=render)
    pool = OverlayRenderPool(cache)
    task = asyncio.create_task(pool.render(object(), {'en'}))
    try:
        await asyncio.wait_for(asyncio.to_thread(_callbacks.entered.wait), 1)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        _callbacks.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await pool.render(None, set()) == {}
    finally:
        _callbacks.release.set()
        pool.close()


class _SlowLanguageDoesNotBlockOthersAndCallbacks:
    entered: Any
    release: Any
    slow_frames: Any

    async def _call_slow(self, frame):
        self.slow_frames.append(int(frame[0]))
        if len(self.slow_frames) == 1:
            self.entered.set()
            await self.release.wait()


class _RenderThreadCancellationPreservesCacheOwnershipRunCallbacks:
    entered: Any
    loop_thread: Any
    release: Any

    def _call_render(self, snapshot, languages):
        assert threading.get_ident() != self.loop_thread
        self.entered.set()
        assert self.release.wait(2)
        return {'en': np.array([1])}


def test_slow_language_does_not_block_others_and_pending_frames_coalesce():

    run = _slow_language_does_not_block_run
    asyncio.run(run())


def test_render_thread_cancellation_preserves_cache_ownership():

    run = _render_thread_cancellation_preserves_cache_run
    asyncio.run(run())
