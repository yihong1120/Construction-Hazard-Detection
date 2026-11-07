from __future__ import annotations

import asyncio
import unittest
from functools import partial
from typing import Any

from src.async_tasks import finish_on_cancel


async def _cancelled_owner_consumes_late_failure_work(
    release,
    started,
    /,
):
    started.set()
    await release.wait()
    raise OSError('late failure')


class _RepeatedCancellationWaitsForIrreversibleWorkCallbacks:
    completed: Any
    release: Any
    started: Any

    async def _call_work(self):
        self.started.set()
        await self.release.wait()
        self.completed.append(True)


class OwnedOperationTest(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_cancellation_waits_for_irreversible_work(self):
        _callbacks = _RepeatedCancellationWaitsForIrreversibleWorkCallbacks()
        _callbacks.started, _callbacks.release = (
            asyncio.Event(),
            asyncio.Event(),
        )
        _callbacks.completed = []

        work = _callbacks._call_work

        task = asyncio.create_task(finish_on_cancel(work()))
        await _callbacks.started.wait()
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
        _callbacks.release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(_callbacks.completed, [True])

    async def test_cancelled_owner_consumes_late_failure(self):
        started, release = asyncio.Event(), asyncio.Event()

        work = partial(
            _cancelled_owner_consumes_late_failure_work,
            release,
            started,
        )

        task = asyncio.create_task(finish_on_cancel(work()))
        await started.wait()
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
