from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from src import frame_pacer as module


async def _slow_writes_skip_expired_slots_run():
    _callbacks = _SlowWritesSkipExpiredSlotsInsteadOfCallbacks()
    _callbacks.now = 0.0

    sleep = _callbacks._call_sleep
    with patch.object(
        module.asyncio,
        'get_running_loop',
        return_value=SimpleNamespace(time=_callbacks._call_time),
    ), patch.object(module.asyncio, 'sleep', sleep):
        pacer = module.FramePacer(30)
        assert pacer.timing.actual_fps == 0
        await pacer.wait()
        _callbacks.now += 0.08
        pacer.complete(True)
        assert pacer.timing.missed_deadlines == 2
        await pacer.wait()
        assert _callbacks.now == pytest.approx(0.1)
        _callbacks.now += 0.02
        pacer.begin()
        assert pacer.timing.max_lateness_seconds == pytest.approx(0.02)
        assert not pacer.report_due()


class _WriteCostIsCompensatedWithoutChangingSourceCallbacks:
    fps: Any

    async def _call_run(self):
        _callbacks = _WriteCostIsCompensatedWithoutChangingSourceCallbacks2()
        _callbacks.parent = self
        _callbacks.now = 0.0

        sleep = _callbacks._call_sleep
        loop = SimpleNamespace(time=_callbacks._call_time)
        with patch.object(
            module.asyncio,
            'get_running_loop',
            return_value=loop,
        ), patch.object(module.asyncio, 'sleep', sleep):
            pacer = module.FramePacer(self.fps)
            for _ in range(120):
                await pacer.wait()
                _callbacks.now += 0.005
                pacer.complete(True)
            assert pacer.timing.actual_fps == pytest.approx(self.fps)
            assert pacer.timing.write_seconds == pytest.approx(0.6)
            assert pacer.timing.missed_deadlines == 0
            _callbacks.now += 31
            assert pacer.report_due()
            assert not pacer.report_due()
            pacer.reset_deadline()
            await pacer.wait()
            pacer.complete(False)
            assert pacer.timing.backpressure_skips == 1


class _WriteCostIsCompensatedWithoutChangingSourceCallbacks2:
    parent: _WriteCostIsCompensatedWithoutChangingSourceCallbacks
    now: Any

    async def _call_sleep(self, delay):

        self.now += delay

    def _call_time(self):
        return self.now


class _SlowWritesSkipExpiredSlotsInsteadOfCallbacks:
    now: Any

    async def _call_sleep(self, delay):

        self.now += delay

    def _call_time(self):
        return self.now


@pytest.mark.parametrize('fps', [25.0, 30.0, 30000 / 1001, 60.0])
def test_write_cost_is_compensated_without_changing_source_fps(fps):

    _callbacks = _WriteCostIsCompensatedWithoutChangingSourceCallbacks()
    _callbacks.fps = fps
    run = _callbacks._call_run
    asyncio.run(run())


def test_slow_writes_skip_expired_slots_instead_of_bursting():

    run = _slow_writes_skip_expired_slots_run
    asyncio.run(run())
