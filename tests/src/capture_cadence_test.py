from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

import numpy as np
import pytest

from src.frame_pacer import CaptureCadence
from src.stream_capture import StreamCapture


def test_jitter_does_not_halve_same_rate_capture():
    gate = CaptureCadence()
    times = [0.] + [
        i / 30 + (-1 if i % 2 else 1) * .001
        for i in range(1, 900)
    ]
    assert sum(gate.admit(t, 1 / 30) for t in times) == 900


def test_high_rate_input_stays_bounded_and_outages_do_not_catch_up():
    gate = CaptureCadence()
    admitted = sum(gate.admit(i / 120, 1 / 30) for i in range(1200))
    assert 299 <= admitted <= 301
    assert gate.admit(100, 1 / 30)
    assert not gate.admit(100.001, 1 / 30)
    assert gate.admit(100.002, 0)
    assert gate.admit(100.003, 0)
    assert gate.admit(100.004, 1 / 15)
    assert not gate.admit(100.02, 1 / 15)


@pytest.mark.parametrize('generic', [False, True])
def test_capture_generators_use_monotonic_cadence_with_jitter(generic):
    class Finished(Exception):
        pass

    async def run():
        capture = StreamCapture('https://camera.example.test/fixture')
        capture.capture_interval = 1 / 30
        clock = [0.]
        index = [0]
        frame = np.zeros((4, 4, 3), dtype=np.uint8)

        def read():
            i = index[0]
            if i == 120:
                raise Finished
            clock[0] = i / 30 + (0 if i == 0 else (-1 if i % 2 else 1) * .001)
            index[0] += 1
            return True, frame

        async def execute(operation, *args):
            return operation(*args)

        capture.cap = SimpleNamespace(read=read)
        with (
            patch.object(capture, 'initialise_stream', new=AsyncMock()),
            patch.object(capture, 'release_resources', new=AsyncMock()),
            patch.object(capture, '_run_capture_operation', new=execute),
            patch.object(
                capture, '_should_reconnect_after_stalled_source_timestamp',
                return_value=False,
            ),
            patch.object(
                capture, '_should_reconnect_after_frozen_frame',
                return_value=False,
            ),
            patch.object(
                capture, 'select_quality_based_on_speed',
                return_value='fixture',
            ),
            patch(
                'src.stream_capture.time.monotonic',
                side_effect=lambda: clock[0],
            ),
        ):
            generator = capture.capture_generic_frames(
            ) if generic else capture.execute_capture()
            count = 0
            try:
                async for _, timestamp in generator:
                    assert isinstance(timestamp, float)
                    count += 1
            except Finished:
                pass
            assert count == 120

    asyncio.run(run())
