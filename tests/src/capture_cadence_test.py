from __future__ import annotations

import asyncio
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


class CaptureFinished(Exception):
    pass


class JitteringCapture:
    def __init__(self):
        self.timestamp = 0.0
        self.index = 0
        self.frame = np.zeros((4, 4, 3), dtype=np.uint8)

    def read(self):
        index = self.index
        if index == 120:
            raise CaptureFinished
        jitter = 0 if index == 0 else (-1 if index % 2 else 1) * .001
        self.timestamp = index / 30 + jitter
        self.index += 1
        return True, self.frame

    def monotonic(self):
        return self.timestamp

    @staticmethod
    async def execute(operation, *args):
        return operation(*args)


async def assert_capture_cadence(generic):
    capture = StreamCapture('https://camera.example.test/fixture')
    capture.capture_interval = 1 / 30
    fixture = JitteringCapture()
    with (
        patch.object(capture, 'cap', new=fixture),
        patch.object(capture, 'initialise_stream', new=AsyncMock()),
        patch.object(capture, 'release_resources', new=AsyncMock()),
        patch.object(capture, '_run_capture_operation', new=fixture.execute),
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
        patch('src.stream_capture.time.monotonic', new=fixture.monotonic),
    ):
        generator = (
            capture.capture_generic_frames() if generic
            else capture.execute_capture()
        )
        count = 0
        try:
            async for _, timestamp in generator:
                assert isinstance(timestamp, float)
                count += 1
        except CaptureFinished:
            pass
        assert count == 120


@pytest.mark.parametrize('generic', [False, True])
def test_capture_generators_use_monotonic_cadence_with_jitter(generic):
    asyncio.run(assert_capture_cadence(generic))
