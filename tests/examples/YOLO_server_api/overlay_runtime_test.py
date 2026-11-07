from __future__ import annotations

import asyncio
import threading
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from examples.YOLO_server_api.overlay_runtime import OverlayRuntime


def _cancelled_native_work_keeps_capacity_render(
    entered,
    release,
    /,
):
    entered.append(1)
    release.wait(5)


def _cancelled_native_work_keeps_capacity_callback(entered, /):
    return len(entered) == 2


def _cancelled_native_work_keeps_capacity_callback2(runtime, /):
    return runtime._pending == 0


def _cancelled_native_work_keeps_capacity_render2():
    return 'recovered'


def _cancelled_queue_does_not_admit_head(
    entered,
    release,
    /,
):
    entered.set()
    release.wait(5)


def _cancelled_queue_does_not_admit_append(called, /):
    return called.append(1)


def _cancelled_queue_does_not_admit_callback(runtime, /):
    return runtime._pending == 2


def _return_none():
    return None


def _cancelled_shutdown_still_drains_native_render(
    entered,
    release,
    /,
):
    entered.set()
    release.wait(5)


def _cancelled_shutdown_still_drains_native_get__closed(runtime, /):
    return runtime._closed


def _work_and_submission_errors_release_broken():
    raise ValueError('invalid geometry')


class OverlayRuntimeTest(unittest.IsolatedAsyncioTestCase):
    async def wait_until(self, predicate):
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(0.005)

    async def test_cancelled_native_work_keeps_capacity_until_completion(self):
        runtime = OverlayRuntime(2, 2)
        release = threading.Event()
        entered = []

        render = partial(
            _cancelled_native_work_keeps_capacity_render,
            entered,
            release,
        )

        work = SimpleNamespace(render=render)
        tasks = [asyncio.create_task(runtime.render(work)) for _ in range(2)]
        try:
            await self.wait_until(
                partial(
                    _cancelled_native_work_keeps_capacity_callback,
                    entered,
                ),
            )
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            with self.assertRaises(HTTPException) as caught:
                await runtime.render(work)
            self.assertEqual(caught.exception.status_code, 503)
            self.assertEqual(runtime._pending, 2)
            release.set()
            await self.wait_until(
                partial(
                    _cancelled_native_work_keeps_capacity_callback2,
                    runtime,
                ),
            )
            await runtime.render(
                SimpleNamespace(
                    render=_cancelled_native_work_keeps_capacity_render2,
                ),
            )
        finally:
            release.set()
            await runtime.close()

    async def test_cancelled_queue_does_not_admit_an_unbounded_backlog(self):
        runtime = OverlayRuntime(1, 2)
        release = threading.Event()
        entered = threading.Event()
        called = []

        head = partial(_cancelled_queue_does_not_admit_head, entered, release)

        task = asyncio.create_task(
            runtime.render(SimpleNamespace(render=head)),
        )
        try:
            await self.wait_until(entered.is_set)
            queued = asyncio.create_task(
                runtime.render(
                    SimpleNamespace(
                        render=partial(
                            _cancelled_queue_does_not_admit_append,
                            called,
                        ),
                    ),
                ),
            )
            await self.wait_until(
                partial(_cancelled_queue_does_not_admit_callback, runtime),
            )
            queued.cancel()
            await asyncio.gather(queued, return_exceptions=True)
            for _ in range(5):
                with self.assertRaises(HTTPException):
                    await runtime.render(SimpleNamespace(render=_return_none))
            release.set()
            await task
            await self.wait_until(
                partial(
                    _cancelled_native_work_keeps_capacity_callback2,
                    runtime,
                ),
            )
            self.assertEqual(called, [])
        finally:
            release.set()
            await runtime.close()

    async def test_cancelled_shutdown_still_drains_native_work(
        self,
    ):
        runtime = OverlayRuntime(1, 1)
        release = threading.Event()
        entered = threading.Event()

        render = partial(
            _cancelled_shutdown_still_drains_native_render,
            entered,
            release,
        )

        task = asyncio.create_task(
            runtime.render(SimpleNamespace(render=render)),
        )
        await self.wait_until(entered.is_set)
        closing = asyncio.create_task(runtime.close())
        await self.wait_until(
            partial(
                _cancelled_shutdown_still_drains_native_get__closed,
                runtime,
            ),
        )
        closing.cancel()
        await asyncio.sleep(0.01)
        self.assertFalse(closing.done())
        with self.assertRaises(HTTPException):
            await runtime.render(SimpleNamespace(render=_return_none))
        release.set()
        await task
        with self.assertRaises(asyncio.CancelledError):
            await closing
        self.assertEqual(runtime._pending, 0)

    async def test_work_and_submission_errors_release_capacity(
        self,
    ):
        runtime = OverlayRuntime(1, 1)
        try:

            broken = _work_and_submission_errors_release_broken

            with self.assertRaises(ValueError):
                await runtime.render(SimpleNamespace(render=broken))
            self.assertEqual(runtime._pending, 0)
            with patch.object(
                runtime.executor,
                'submit',
                side_effect=RuntimeError('closed'),
            ):
                with self.assertRaises(RuntimeError):
                    await runtime.render(SimpleNamespace(render=broken))
            self.assertEqual(runtime._pending, 0)
        finally:
            await runtime.close()
        with self.assertRaises(ValueError):
            OverlayRuntime(0, 1)
        with self.assertRaises(ValueError):
            OverlayRuntime(2, 1)
