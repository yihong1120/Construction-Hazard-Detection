from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

from src import stream_detection
from src.stream_runtime_state import DetectionResult
from src.stream_runtime_state import LatestFrameState
from tests.stream_fixtures_test import make_stream_runtime

pytestmark = pytest.mark.ml


async def _stop_at_latest(runtime, result):
    if result.sequence == runtime.latest_frame.sequence:
        runtime.stop.set()


async def _process_burst(depth, expected):
    frames = [np.full((2, 2, 3), n, dtype=np.uint8) for n in range(1, 7)]
    capture = MagicMock()
    capture.execute_capture.return_value.__aiter__.return_value = [
        (frame, float(n)) for n, frame in enumerate(frames, 1)
    ]
    state = LatestFrameState(history_depth=depth)
    runtime = make_stream_runtime(latest_frame=state)
    runtime.detector.generate_detections.return_value = ([], [])
    await stream_detection._capture_latest_frames(capture, state, runtime.stop)
    assert state.frame is frames[-1]
    record = AsyncMock(side_effect=_stop_at_latest)
    with patch.object(stream_detection, '_record_detection_result', record):
        await asyncio.wait_for(
            stream_detection._detect_latest_frames(runtime), timeout=1,
        )
    results = [call.kwargs['result'] for call in record.await_args_list]
    assert [r.sequence for r in results] == expected
    for result in results:
        assert result.frame is frames[result.sequence - 1]
        assert not result.frame.flags.writeable
        assert result.timestamp == float(result.sequence)
    assert state.frame is frames[-1] and not state.event.is_set()


@pytest.mark.parametrize(('depth', 'expected'), [(1, [6]), (4, [3, 4, 5, 6])])
def test_capture_and_detection_drain_a_burst_without_another_frame(
    depth, expected,
):
    with patch('src.stream_runtime_state.monotonic', return_value=100.0):
        asyncio.run(_process_burst(depth, expected))


def test_skip_expired_history_and_keep_latest_even_after_a_pause():
    state = LatestFrameState(history_depth=4)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    with patch('src.stream_runtime_state.monotonic') as clock:
        for seq, arrival in enumerate((1.0, 1.05, 1.08, 1.09), 1):
            clock.return_value = arrival
            state.publish(frame, float(seq))
        clock.return_value = 1.12
        assert state.take_for_inference(0).sequence == 2
        clock.return_value = 2.0
        assert state.take_for_inference(2).sequence == 4
        assert state.take_for_inference(4) is None
        assert not state.event.is_set()


def test_reconnect_discards_even_fresh_history_before_the_next_capture():
    state = LatestFrameState(history_depth=4)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    with patch('src.stream_runtime_state.monotonic', return_value=1.0):
        state.publish(frame, 1.0)
        state.publish(frame, 2.0)
        state.invalidate()
        state.publish(frame, 4.0)
        assert state.generation == 1
        assert state.take_for_inference(0).sequence == 4
        assert not state.event.is_set()


async def _reconnect_during_status_write(runtime, metadata):
    async with runtime.latest_frame.lock:
        runtime.latest_frame.invalidate()


def test_old_result_cannot_repopulate_overlay_after_reconnect():
    runtime = make_stream_runtime()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    runtime.latest_frame.publish(frame, 1.0)
    result = DetectionResult(frame, 1.0, 1, [], {'model_id': 'fixture'})
    with patch.object(
        stream_detection,
        'report_loaded_model',
        _reconnect_during_status_write,
    ):
        asyncio.run(stream_detection._record_detection_result(runtime, result))
    assert runtime.latest_detection.frame is None
    assert not runtime.latest_detection.event.is_set()
    runtime.danger.detect_danger.assert_not_called()
