from __future__ import annotations

import asyncio
import threading
import time
import unittest
from functools import partial

from src.process_reaper import ProcessReaper


class _SlowChildrenLeaveLoopResponsiveChild:
    def __init__(self):
        self.alive = True
        self.killed = False
        self.timeouts = []

    def terminate(self):
        pass

    def join(self, timeout):
        self.timeouts.append(timeout)
        if not self.killed:
            time.sleep(min(0.04, timeout))

    def is_alive(self):
        return self.alive

    def kill(self):
        self.killed = True
        self.alive = False


class _CancellationWaitsForChildCleanupChild:
    def __init__(self, *, _test_entered, _test_release):
        self._test_entered = _test_entered
        self._test_release = _test_release

    alive = True

    def is_alive(self):
        return self.alive

    def terminate(self):
        pass

    def join(self, timeout):
        self._test_entered.set()
        self._test_release.wait(2)
        self.alive = False


class ProcessReaperTest(unittest.IsolatedAsyncioTestCase):
    async def test_slow_children_leave_loop_responsive_and_share_deadline(
        self,
    ):
        Child = _SlowChildrenLeaveLoopResponsiveChild

        children = [Child() for _ in range(12)]
        reaper = ProcessReaper(timeout=0.04)
        task = asyncio.create_task(reaper.reap(children))
        ticks = 0
        while not task.done():
            ticks += 1
            await asyncio.sleep(0.005)
        await task
        self.assertGreater(ticks, 3)
        self.assertTrue(all(child.killed for child in children))
        self.assertTrue(
            all(child.timeouts[0] < 0.01 for child in children[4:]),
        )

    async def test_cancellation_waits_for_child_cleanup(self):
        entered, release = threading.Event(), threading.Event()

        Child = partial(
            _CancellationWaitsForChildCleanupChild,
            _test_entered=entered,
            _test_release=release,
        )

        child = Child()
        reaper = ProcessReaper()
        task = asyncio.create_task(reaper.reap([child]))
        await asyncio.to_thread(entered.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(child.alive)

    async def test_real_children_are_joined_without_blocking_loop(self):
        from multiprocessing import Process

        children = [Process(target=time.sleep, args=(10,)) for _ in range(2)]
        reaper = ProcessReaper(timeout=1)
        try:
            for child in children:
                child.start()
            await reaper.reap(children)
            self.assertTrue(all(not child.is_alive() for child in children))
            self.assertTrue(
                all(child.exitcode is not None for child in children),
            )
        finally:
            for child in children:
                if child.is_alive():
                    child.kill()
                child.join(timeout=2)
