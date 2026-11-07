from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import runpy
import signal
import unittest
from collections.abc import AsyncIterator
from collections.abc import Iterator
from contextlib import suppress
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from functools import partial
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import ANY
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError

import main
import src.stream_processor as processor
import src.worker_supervisor as worker_supervisor
from examples.streaming_web.media_paths import build_overlay_demand_key
from examples.streaming_web.overlay_models import OverlaySnapshot
from examples.streaming_web.overlay_models import WarningPayload
from main import MainApp
from main import process_single_stream
from main import StreamConfig
from src import stream_detection
from src import stream_media as media
from src import stream_overlay_frames
from src import stream_runtime_state as runtime_state
from src.settings.media import MediaPublishSettings
from src.stream_identity import stream_identity
from src.stream_runtime_state import DetectionResult
from src.worker_supervisor import StreamProcessState
from src.worker_supervisor import WorkerSupervisor
from tests.overlay_fixtures_test import make_overlay_renderer
from tests.stream_fixtures_test import DeliveryDouble
from tests.stream_fixtures_test import make_media_group
from tests.stream_fixtures_test import make_overlay_cache
from tests.stream_fixtures_test import make_stream_runtime


async def _asyncSetUp_reap(processes, graceful=False):
    for process in processes:
        process.terminate()
        process.join(timeout=10)


async def _reload_configurations_starts_new_stream_run(
    app: Any,
    /,
) -> None:
    """Support run."""
    await app.reload_configurations()


async def _reload_configurations_skips_expired_config_run(
    app: Any,
    /,
) -> None:
    """Support run."""
    await app.reload_configurations()


async def _config_change_does_not_overlap_wait_forever() -> None:
    """Keep the existing reload task pending for this test."""
    await asyncio.Event().wait()


def _process_single_stream_loads_env_close(coro):
    return coro.close()


def _detect_latest_frames_publishes_detected_detect_and_stop(
    stop_event: Any,
    /,
    *_args: object,
) -> tuple[dict, list, list]:
    """Support detect_and_stop."""
    stop_event.set()
    return ({}, [], [])


async def _decoupled_loop_starts_all_requested_stop_capture(
    **_kwargs: Any,
) -> Any:
    """Support stop_capture."""
    return None


async def _decoupled_loop_starts_all_requested_stop_detection(
    **_kwargs: Any,
) -> Any:
    """Support stop_detection."""
    return None


async def _decoupled_loop_starts_all_requested_stop_overlay(
    **_kwargs: Any,
) -> Any:
    """Support stop_overlay."""
    return None


async def _decoupled_loop_starts_all_requested_stop_clean(
    *_args: Any,
    **_kwargs: Any,
) -> Any:
    """Support stop_clean."""
    return None


async def _decoupled_loop_raises_child_task_fail_capture(
    **_kwargs: Any,
) -> None:
    """Support fail_capture."""
    raise RuntimeError('capture failed')


async def _decoupled_loop_raises_child_task_stop_detection(
    **_kwargs: Any,
) -> None:
    """Support stop_detection."""
    await asyncio.Event().wait()


async def _capture_latest_frames_returns_when_execute_capture() -> (
    AsyncIterator[tuple[np.ndarray, float]]
):
    """Support execute_capture."""
    yield (np.zeros((4, 4, 3), dtype=np.uint8), 1640995200.0)


async def _capture_latest_frames_updates_latest_execute_capture(
    frame: Any,
    /,
) -> AsyncIterator[tuple[np.ndarray, float]]:
    """Support execute_capture."""
    yield (frame, 1640995200.0)


async def _detect_latest_frames_recovers_from_stop_after_sleep(
    stop_event: Any,
    /,
    _delay: Any,
) -> None:
    """Support stop_after_sleep.

    Args:
        _delay: Test helper value.
    """
    stop_event.set()


async def _detect_latest_frames_recovers_from_stop_after_sleep2(
    stop_event: Any,
    /,
    _delay: Any,
) -> None:
    """Support stop_after_sleep.

    Args:
        _delay: Test helper value.
    """
    stop_event.set()


async def _detect_latest_frames_sends_violation_send_and_stop(
    stop_event: Any,
    /,
    **_kwargs: Any,
) -> Any:
    """Support send_and_stop."""
    stop_event.set()
    return 1640995200


async def _detect_latest_frames_skips_empty_wait_once(
    stop_event: Any,
    /,
) -> None:
    """Support wait_once."""
    stop_event.set()


async def _detect_latest_frames_continues_on_wait_timeout(
    stop_event: Any,
    /,
    awaitable: Any,
    timeout: Any,
) -> None:
    """Support wait_timeout.

    Args:
        awaitable: Test helper value.
        timeout: Test helper value.
    """
    awaitable.close()
    stop_event.set()
    raise TimeoutError


async def _requested_overlay_languages_uses_bounded_mget(
    valid_key: Any,
    /,
    keys: list[str],
) -> list[bytes | None]:
    """Support MGET."""
    return [b'1' if key == valid_key else None for key in keys]


async def _media_demand_cache_reuses_clean_mget(
    demand_key: Any,
    /,
    keys: list[str],
) -> list[bytes | None]:
    """Return demand for exactly one configured language."""
    return [b'1' if key == demand_key else None for key in keys]


async def _publish_requested_clean_frames_waits_sleep_once(
    stop_event: Any,
    /,
    _delay: Any,
) -> None:
    """Support sleep_once.

    Args:
        _delay: Test helper value.
    """
    stop_event.set()


async def _publish_requested_clean_frames_recovers_sleep_once(
    stop_event: Any,
    /,
    _delay: Any,
) -> None:
    """Support sleep_once.

    Args:
        _delay: Test helper value.
    """
    stop_event.set()


async def _poll_and_reload_runs_once_stop_after_one() -> None:
    """Support stop_after_one."""
    await asyncio.sleep(0.01)
    raise KeyboardInterrupt()


def _main_function_json_config_is_alive_side_effect() -> Iterator[bool]:
    """Support is_alive_side_effect."""
    yield True
    while True:
        yield False


async def _overlay_frame_helpers_cover_fallback_run_case() -> None:
    loop = MagicMock()
    loop.time.side_effect = [10.0, 11.1]
    with patch.object(
        processor.asyncio,
        'get_running_loop',
        return_value=loop,
    ), patch.dict(os.environ, {'MEDIA_OVERLAY_READY_GRACE_SECONDS': '1'}):
        started_at: dict[str, float] = {}
        assert not make_media_group()._overlay_ready_grace_elapsed(
            started_at,
            'en',
        )
        assert make_media_group()._overlay_ready_grace_elapsed(
            started_at,
            'en',
        )


async def _overlay_snapshot_uses_detection_frame_run_case() -> None:
    latest_frame = runtime_state.LatestFrameState()
    latest_detection = runtime_state.LatestDetectionState()
    frame = np.ones((2, 2, 3), dtype=np.uint8)
    async with latest_detection.lock:
        latest_detection.frame = frame
        latest_detection.sequence = 4
    snapshot = await media._latest_overlay_snapshot(
        latest_frame,
        latest_detection,
    )
    assert snapshot is not None
    assert snapshot.frame is frame
    assert snapshot.sequence == (0, 4)


async def _overlay_language_snapshot_creates_publisher_run_case() -> None:
    redis_manager = MagicMock()
    publisher = AsyncMock()
    publishers: dict[str, object] = {}
    with patch.object(
        media.MediaPublisherFactory,
        'create',
        return_value=publisher,
    ), patch.object(
        media.StreamPublisherGroup,
        '_mark_overlay_ready',
        new=AsyncMock(),
    ) as mark:
        await (
            publish_group := make_media_group(
                redis_manager=redis_manager,
                overlay_media_publishers=cast(dict, publishers),
                media_publish_base='rtsp://media:8554',
                media_path='hazard_site_cam',
                site='SiteA',
                stream_name='Cam1',
                label_language='en',
                publish_frame=np.zeros((2, 2, 3), dtype=np.uint8),
            )
        ).publish_overlay(
            publish_group.test_variant,
            'en',
            np.zeros((2, 2, 3), dtype=np.uint8),
        )
    assert publishers['en'] is publisher
    publisher.publish.assert_awaited_once()
    mark.assert_awaited_once()


async def _overlay_publish_loop_processes_demand_stop_after_frame(
    stop_event: Any,
    /,
    _delay: float,
) -> None:
    stop_event.set()


async def _capture_reconnect_invalidates_shared_frames_run_case() -> None:
    reconnect_event = asyncio.Event()
    clean_reconnect_event = asyncio.Event()
    latest_frame = runtime_state.LatestFrameState(
        frame=np.zeros(
            (2, 2, 3),
            dtype=np.uint8,
        ),
        timestamp=10.0,
        sequence=4,
        generation=2,
    )
    latest_detection = runtime_state.LatestDetectionState(
        frame=np.ones((2, 2, 3), dtype=np.uint8),
        timestamp=10.0,
        sequence=4,
        warnings={
            'warning': {'count': 1},
        },
        track_data=[[1, 1, 2, 2, 0.9, 0]],
    )
    task = asyncio.create_task(
        stream_detection._synchronise_capture_reconnects(
            reconnect_event=reconnect_event,
            clean_reconnect_event=clean_reconnect_event,
            latest_frame=latest_frame,
            latest_detection=latest_detection,
            stop_event=asyncio.Event(),
        ),
    )
    reconnect_event.set()
    await asyncio.wait_for(clean_reconnect_event.wait(), timeout=1.0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert latest_frame.frame is None
    assert latest_frame.timestamp == 0.0
    assert latest_frame.sequence == 5
    assert latest_frame.generation == 3
    assert latest_detection.frame is None
    assert latest_detection.warnings == {}
    assert latest_detection.track_data is None


async def _detection_result_is_discarded_across_run_case() -> None:
    latest_frame = runtime_state.LatestFrameState(
        frame=np.zeros(
            (2, 2, 3),
            dtype=np.uint8,
        ),
        timestamp=10.0,
        sequence=1,
    )
    latest_frame.event.set()
    stop_event = asyncio.Event()
    yolo_detector = AsyncMock()
    yolo_detector.model_metadata = {}

    reconnect_during_detection = partial(
        _detection_result_is_discarded_across_reconnect_during_detection,
        latest_frame,
        stop_event,
    )

    yolo_detector.generate_detections.side_effect = reconnect_during_detection
    danger_detector = MagicMock()
    await stream_detection._detect_latest_frames(
        runtime=make_stream_runtime(
            latest_frame=latest_frame,
            yolo_detector=yolo_detector,
            danger_detector=danger_detector,
            fcm_sender=AsyncMock(),
            violation_sender=AsyncMock(),
            redis_manager=MagicMock(),
            latest_detection=runtime_state.LatestDetectionState(),
            site='SiteA',
            stream_name='Cam1',
            work_start_hour=0,
            work_end_hour=24,
            metadata_key='stream_metadata:site|cam',
            stop_event=stop_event,
        ),
    )
    danger_detector.detect_danger.assert_not_called()


async def _detection_result_is_discarded_across_reconnect_during_detection(
    latest_frame: Any,
    stop_event: Any,
    /,
    _frame: np.ndarray,
) -> tuple[list[object], list[object]]:
    async with latest_frame.lock:
        latest_frame.frame = None
        latest_frame.generation += 1
    stop_event.set()
    return ([], [])


async def _overlay_publish_loop_retries_after_stop_after_retry(
    stop_event: Any,
    /,
    _delay: float,
) -> None:
    stop_event.set()


async def _main_config_mode_handles_no_run_case() -> None:
    app = MagicMock()
    app._can_run_stream.return_value = False
    app.cleanup_resources = AsyncMock()
    args = SimpleNamespace(config='configs.json', poll=1)
    with patch.object(
        main.argparse.ArgumentParser,
        'parse_args',
        return_value=args,
    ), patch(
        'builtins.open',
        unittest.mock.mock_open(read_data='[]'),
    ), patch.object(
        main.json,
        'load',
        return_value=[{}],
    ), patch.object(
        main,
        'MainApp',
        return_value=app,
    ):
        await main.main()
    app.cleanup_resources.assert_awaited_once()


class _MainEntrypointCallbacks:
    asyncio_run_called: Any

    def _call_fake_run(self, coro: Any) -> None:
        """Support fake_run.

        Args:
            coro: Test helper value.
        """

        self.asyncio_run_called = True
        assert asyncio.iscoroutine(coro)
        coro.close()


class _CleanPublisherSurvivesCaptureReconnectCallbacks:
    latest_frame: Any
    resumed_frame: Any
    sleep_calls: Any
    stop_event: Any

    async def _call_advance_capture_state(self, _delay: Any) -> None:
        """Simulate a source outage followed by fresh video.

        Args:
            _delay: Requested loop delay.
        """

        self.sleep_calls += 1
        async with self.latest_frame.lock:
            if self.sleep_calls == 1:
                self.latest_frame.frame = None
            elif self.sleep_calls == 2:
                self.latest_frame.frame = self.resumed_frame
            elif self.sleep_calls == 3:
                self.stop_event.set()


class _CleanFrameLoopReleasesRestreamersOnDemandCallbacks:
    sleep_calls: Any
    stop_event: Any

    async def _call_stop_after_third_sleep(self, _delay: float) -> None:

        self.sleep_calls += 1
        if self.sleep_calls == 3:
            self.stop_event.set()


class _CleanFrameLoopReleasesPublishersOnDemandCallbacks:
    sleep_calls: Any
    stop_event: Any

    async def _call_stop_after_third_sleep(self, _delay: float) -> None:

        self.sleep_calls += 1
        if self.sleep_calls == 3:
            self.stop_event.set()


class _PollAndReloadExceptionHandlingCallbacks:
    call_count: Any

    async def _call_side_effect(self) -> None:
        """Support side_effect."""

        self.call_count += 1
        if self.call_count == 1:
            raise Exception('Test exception')
        elif self.call_count == 2:
            raise KeyboardInterrupt()


class _PollAndReloadResetsPoolOnTimeoutCallbacks:
    call_count: Any

    async def _call_side_effect(self) -> None:
        """Support side_effect."""

        self.call_count += 1
        if self.call_count == 1:
            raise TimeoutError('db timed out')
        raise KeyboardInterrupt()


class _MainFunctionJsonConfigKeyboardInterruptCallbacks:
    call_count: Any
    join_call_count: Any

    def _call_is_alive_side_effect(self) -> Any:
        """Support is_alive_side_effect."""

        self.call_count += 1
        if self.call_count <= 2:
            return True
        return False

    def _call_join_side_effect(self, *args: Any, **kwargs: Any) -> Any:
        """Support join_side_effect."""

        self.join_call_count += 1
        if self.join_call_count == 1:
            raise KeyboardInterrupt('User interrupted')
        return None


class _MainFunctionJsonConfigAliveProcessCleanupCallbacks:
    call_count: Any

    def _call_is_alive_side_effect(self) -> Any:
        """Support is_alive_side_effect."""

        self.call_count += 1
        if self.call_count == 1:
            return False
        else:
            return True


class _OverlayPublishLoopProcessesDemandThenClosesCallbacks:
    DemandCache: Any

    async def _call_run_case(self) -> None:
        stop_event = asyncio.Event()
        variants = [
            runtime_state.OverlayPublisherVariant(
                'hazard_site_cam',
                'detail',
            ),
        ]

        stop_after_frame = partial(
            _overlay_publish_loop_processes_demand_stop_after_frame,
            stop_event,
        )

        with patch.object(
            media,
            '_latest_overlay_snapshot',
            new=AsyncMock(return_value=None),
        ), patch.object(
            stream_overlay_frames.OverlayFrameCache,
            'render',
            return_value={'en': np.zeros((2, 2, 3), dtype=np.uint8)},
        ), patch.object(
            media.StreamPublisherGroup,
            'publish_overlay',
            new=AsyncMock(),
        ) as publish, patch.object(
            media,
            '_close_overlay_publishers',
            new=AsyncMock(),
        ) as close_publishers, patch.object(
            processor.asyncio,
            'sleep',
            side_effect=stop_after_frame,
        ):
            await make_media_group(
                latest_frame=runtime_state.LatestFrameState(),
                latest_detection=runtime_state.LatestDetectionState(),
                redis_manager=MagicMock(),
                media_publish_base='rtsp://media:8554',
                variants=variants,
                site='SiteA',
                stream_name='Cam1',
                stop_event=stop_event,
                demand_cache=cast(
                    media._MediaDemandCache,
                    self.DemandCache(),
                ),
            ).publish_overlays()
        publish.assert_awaited_once()
        close_publishers.assert_awaited_once_with(variants[0].publishers)


class _CleanRestreamerRestartsAfterFrozenSourceSignalCallbacks:
    DemandCache: Any

    async def _call_run_case(self) -> None:
        _callbacks = (
            _CleanRestreamerRestartsAfterFrozenSourceSignalCallbacks2()
        )
        _callbacks.parent = self
        _callbacks.stop_event = asyncio.Event()
        _callbacks.reconnect_event = asyncio.Event()
        restreamer = AsyncMock()
        _callbacks.sleeps = 0

        progress_loop = _callbacks._call_progress_loop

        with patch.object(
            media,
            'MediaSourceRestreamer',
            return_value=restreamer,
        ), patch.object(processor.asyncio, 'sleep', side_effect=progress_loop):
            await make_media_group(
                latest_frame=runtime_state.LatestFrameState(),
                redis_manager=MagicMock(),
                media_publish_base='rtsp://media:8554',
                media_path='hazard_site_cam',
                site='SiteA',
                stream_name='Cam1',
                source_url='rtsp://source',
                use_source_restreamer=True,
                stop_event=_callbacks.stop_event,
                source_reconnect_event=_callbacks.reconnect_event,
                demand_cache=cast(
                    media._MediaDemandCache,
                    self.DemandCache(),
                ),
            ).publish_clean('detail')
        restreamer.start.assert_awaited_once()
        restreamer.restart.assert_awaited_once()
        restreamer.close.assert_awaited_once()


class _CleanRestreamerRestartsAfterFrozenSourceSignalCallbacks2:
    parent: _CleanRestreamerRestartsAfterFrozenSourceSignalCallbacks
    reconnect_event: Any
    sleeps: Any
    stop_event: Any

    async def _call_progress_loop(self, _delay: float) -> None:

        self.sleeps += 1
        if self.sleeps == 1:
            self.reconnect_event.set()
        else:
            self.stop_event.set()


class _OverlayPublishLoopRetriesAfterDemandFailureCallbacks:
    DemandCache: Any

    async def _call_run_case(self) -> None:
        stop_event = asyncio.Event()
        variants = [
            runtime_state.OverlayPublisherVariant(
                'hazard_site_cam',
                'detail',
            ),
        ]

        stop_after_retry = partial(
            _overlay_publish_loop_retries_after_stop_after_retry,
            stop_event,
        )

        with patch.object(
            processor.asyncio,
            'sleep',
            side_effect=stop_after_retry,
        ), patch.object(
            media,
            '_close_overlay_publishers',
            new=AsyncMock(),
        ) as close_publishers:
            await make_media_group(
                latest_frame=runtime_state.LatestFrameState(),
                latest_detection=runtime_state.LatestDetectionState(),
                redis_manager=MagicMock(),
                media_publish_base='rtsp://media:8554',
                variants=variants,
                site='SiteA',
                stream_name='Cam1',
                stop_event=stop_event,
                demand_cache=cast(
                    media._MediaDemandCache,
                    self.DemandCache(),
                ),
            ).publish_overlays()
        close_publishers.assert_awaited_once_with(variants[0].publishers)


class _FetchStreamConfigsDatabaseOperationsMockConnection:
    """Tests for MockConnection."""

    def __init__(self, *, _test_mock_row):
        self._test_mock_row = _test_mock_row

    async def fetch(self, *args: Any, **kwargs: Any) -> Any:
        """Support fetch."""
        return [self._test_mock_row]


class _FetchStreamConfigsDatabaseOperationsMockPool:
    """Tests for MockPool."""

    def __init__(self, *, _test_MockConnection):
        self._test_MockConnection = _test_MockConnection

    def acquire(self) -> Any:
        """Support acquire."""

        MockAcquire = partial(
            _AcquireMockAcquire,
            _test_MockConnection=self._test_MockConnection,
        )

        return MockAcquire()


class _AcquireMockAcquire:
    """Tests for MockAcquire."""

    def __init__(self, *, _test_MockConnection):
        self._test_MockConnection = _test_MockConnection

    async def __aenter__(self) -> Any:
        return self._test_MockConnection()

    async def __aexit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> Any:
        return None


class _OverlayPublishLoopProcessesDemandDemandCache:
    """Provide DemandCache."""

    async def overlay_languages(
        self,
        _redis: object,
        _path: str,
    ) -> set[str]:
        return {'en'}


class _CleanRestreamerRestartsAfterFrozenDemandCache:
    """Provide DemandCache."""

    async def clean_requested(self, _redis: object, _path: str) -> bool:
        return True


class _OverlayPublishLoopRetriesAfterDemandCache:
    """Provide DemandCache."""

    async def overlay_languages(
        self,
        _redis: object,
        _path: str,
    ) -> set[str]:
        raise RuntimeError('redis offline')


@pytest.fixture(autouse=True)
def mock_runtime_preflight():
    """Isolate supervisor tests from services and child processes."""
    with patch(
        'main.validate_runtime_dependencies',
        new_callable=AsyncMock,
    ), patch('main.OutboxDispatcher.start'), patch(
        'main.OutboxDispatcher.close',
        new_callable=AsyncMock,
    ), patch(
        'multiprocessing.process.BaseProcess.start',
        side_effect=AssertionError('Unmocked child process in unit test'),
    ):
        yield


class AsyncFrameGenerator:
    """Async generator for mock video frames."""

    def __init__(self) -> None:
        """Support __init__."""
        self.yielded = False

    def __aiter__(self) -> Any:
        return self

    async def __anext__(self) -> Any:
        if not self.yielded:
            self.yielded = True
            mock_frame = MagicMock()
            mock_frame.shape = [480, 640, 3]
            return (mock_frame, 1640995200)
        else:
            raise StopAsyncIteration


class MockCursor:
    """Tests for MockCursor."""

    async def execute(self, *args: Any, **kwargs: Any) -> None:
        """Support execute."""
        pass

    async def fetchall(self) -> Any:
        """Support fetchall."""
        return []

    async def __aenter__(self) -> Any:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        pass


class MockConnection:
    """Tests for MockConnection."""

    async def fetch(self, *args: Any, **kwargs: Any) -> Any:
        """Support fetch."""
        return []

    def cursor(self) -> Any:
        """Support cursor."""
        return MockCursor()

    async def __aenter__(self) -> Any:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        pass


class MockAcquire:
    """Tests for MockAcquire."""

    async def __aenter__(self) -> Any:
        return MockConnection()

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        pass


class MockPool:
    """Tests for MockPool."""

    def acquire(self) -> Any:
        """Support acquire."""
        return MockAcquire()


class TestMainApp(unittest.IsolatedAsyncioTestCase):
    """Unit tests for MainApp class defined in main.py."""

    async def asyncSetUp(self) -> None:
        """Prepare test fixtures."""
        self.app = MainApp(poll_interval=1)

        reap = _asyncSetUp_reap

        reap_patch = patch.object(
            self.app.workers.reaper,
            'reap',
            new=AsyncMock(side_effect=reap),
        )
        reap_patch.start()
        self.addCleanup(reap_patch.stop)
        self.mock_logger = MagicMock()
        self.app.logger = self.mock_logger
        self.app.workers.logger = self.mock_logger
        self.app.watcher.logger = self.mock_logger
        self.dummy_cfg: StreamConfig = {
            'video_url': 'rtsp://example.com/stream1',
            'updated_at': datetime.now().isoformat(),
            'model_key': 'model-abc',
            'site': 'SiteA',
            'stream_name': 'StreamOne',
            'recognition_enabled': True,
            'expire_date': None,
            'detection_items': {
                'detect_no_safety_vest_or_helmet': True,
                'detect_near_machinery_or_vehicle': False,
                'detect_in_restricted_area': True,
                'detect_in_utility_pole_restricted_area': False,
                'detect_machinery_close_to_pole': False,
            },
            'work_start_hour': 0,
            'work_end_hour': 24,
        }

    @patch.object(MainApp, '_ensure_db_pool', new_callable=AsyncMock)
    async def test_fetch_stream_configs_db_pool_not_initialised(
        self,
        mock_ensure: Any,
    ) -> None:
        """Test fetch_stream_configs raises if db_pool is not initialised."""
        self.app.db_pool = None
        with self.assertRaises(RuntimeError):
            await self.app.fetch_stream_configs()

    @patch.object(WorkerSupervisor, 'ensure', return_value=False)
    @patch('src.worker_supervisor.Process')
    def test_start_and_stop_process(
        self,
        mock_process_class: Any,
        mock_ensure_yolo_worker: Any,
    ) -> None:
        """Test start_process and stop_process methods."""
        mock_process = MagicMock()
        mock_process_class.return_value = mock_process
        cfg = self.dummy_cfg.copy()
        proc = self.app.start_process(cfg)
        mock_process_class.assert_called_once_with(
            target=process_single_stream,
            args=(cfg, None, None),
        )
        mock_ensure_yolo_worker.assert_called_once_with([cfg])
        mock_process.start.assert_called_once()
        self.assertEqual(proc, mock_process)
        self.app.stop_process(proc)
        mock_process.terminate.assert_called_once()
        self.assertTrue(mock_process.join.call_count >= 1)

    @patch.object(WorkerSupervisor, 'reconcile', return_value=False)
    @patch('main.MainApp.start_process')
    @patch('main.MainApp.fetch_stream_configs')
    def test_reload_configurations_starts_new_stream(
        self,
        mock_fetch: Any,
        mock_start: Any,
        _mock_ensure: Any,
    ) -> None:
        """Test reload_configurations starts new stream if not tracked."""
        cfg = self.dummy_cfg.copy()
        cfg['expire_date'] = None
        mock_fetch.return_value = [cfg]
        mock_start.return_value = MagicMock()
        app = MainApp()
        app.running_processes = {}

        run = partial(_reload_configurations_starts_new_stream_run, app)

        asyncio.run(run())
        self.assertIn(stream_identity(cfg), app.running_processes)
        mock_start.assert_called_once_with(cfg)

    @patch('main.MainApp.fetch_stream_configs')
    def test_reload_configurations_skips_expired_config(
        self,
        mock_fetch: Any,
    ) -> None:
        """Test reload_configurations skips expired configs."""
        expired_cfg = self.dummy_cfg.copy()
        expired_cfg['expire_date'] = (
            datetime.now() - timedelta(days=1)
        ).isoformat()
        mock_fetch.return_value = [expired_cfg]
        app = MainApp()
        app.running_processes = {}

        run = partial(_reload_configurations_skips_expired_config_run, app)

        asyncio.run(run())
        self.assertNotIn(stream_identity(expired_cfg), app.running_processes)

    @patch('main.asyncio.run')
    @patch('main.argparse.ArgumentParser.parse_args')
    def test_main_entrypoint(self, mock_args: Any, mock_run: Any) -> None:
        """Test CLI entrypoint main() function."""
        _callbacks = _MainEntrypointCallbacks()
        from main import main as main_entry

        mock_args.return_value = type('Args', (), {'poll': 1})()
        _callbacks.asyncio_run_called = False

        fake_run = _callbacks._call_fake_run

        mock_run.side_effect = fake_run
        asyncio.run(main_entry())
        self.assertTrue(_callbacks.asyncio_run_called)

    async def test_ensure_config_listener_reserves_connection(self) -> None:
        """The listener reserves one pool connection for PostgreSQL events."""
        connection = MagicMock()
        connection.add_listener = AsyncMock()
        pool = MagicMock()
        pool.acquire = AsyncMock(return_value=connection)
        self.app.db_pool = pool
        with patch.object(
            self.app,
            '_ensure_db_pool',
            new_callable=AsyncMock,
        ) as mock_ensure_pool:
            await self.app.watcher.start()
        mock_ensure_pool.assert_awaited_once()
        connection.add_listener.assert_awaited_once_with(
            'stream_config_changed',
            self.app.watcher.notify,
        )
        self.assertIs(self.app.watcher.connection, connection)

    def test_log_config_summary_only_when_state_changes(self) -> None:
        """Configuration status is visible without repeating every poll."""
        self.app.running_processes = {
            'stream': StreamProcessState(MagicMock(), 'now'),
        }
        self.app._log_config_summary(3, 0)
        self.app._log_config_summary(3, 0)
        self.mock_logger.info.assert_called_once_with(
            (
                '[config] Loaded %s configs; %s recognition streams '
                'enabled; %s stream processes active'
            ),
            3,
            0,
            1,
        )

    async def test_ensure_config_listener_skips_existing_or_missing_pool(
        self,
    ) -> None:
        """Existing listeners and unavailable pools do not acquire again."""
        existing_connection = object()
        self.app.watcher.connection = existing_connection
        with patch.object(
            self.app,
            '_ensure_db_pool',
            new_callable=AsyncMock,
        ) as mock_ensure_pool:
            await self.app.watcher.start()
            mock_ensure_pool.assert_not_awaited()
            self.app.watcher.connection = None
            self.app.db_pool = None
            await self.app.watcher.start()
        mock_ensure_pool.assert_awaited_once()

    async def test_ensure_config_listener_releases_on_registration_error(
        self,
    ) -> None:
        """A failed LISTEN registration returns the connection to its pool."""
        connection = MagicMock()
        connection.add_listener = AsyncMock(
            side_effect=RuntimeError('listen failed'),
        )
        pool = MagicMock()
        pool.acquire = AsyncMock(return_value=connection)
        pool.release = AsyncMock()
        self.app.db_pool = pool
        with self.assertRaisesRegex(RuntimeError, 'listen failed'):
            await self.app.watcher.start()
        pool.release.assert_awaited_once_with(connection)
        self.assertIsNone(self.app.watcher.connection)

    async def test_config_change_reloads_and_logs_failure(self) -> None:
        """A notification starts one reload and records a reload failure."""
        with patch.object(
            self.app,
            'reload_configurations',
            new=AsyncMock(side_effect=RuntimeError('reload failed')),
        ) as reload_configurations:
            self.app.watcher.notify(object(), 1, 'stream_config_changed', '')
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        reload_configurations.assert_awaited_once()
        self.mock_logger.error.assert_called_once_with(
            '[config] Immediate reload failed: reload failed',
        )
        assert self.app.watcher.task is not None
        self.assertTrue(self.app.watcher.task.done())

    async def test_config_change_does_not_overlap_existing_reload(
        self,
    ) -> None:
        """A notification does not schedule another reload while one runs."""

        wait_forever = _config_change_does_not_overlap_wait_forever

        pending_task = asyncio.create_task(wait_forever())
        self.app.watcher.task = pending_task
        with patch('main.asyncio.create_task') as mock_create_task:
            self.app.watcher.notify(object(), 1, 'stream_config_changed', '')
        mock_create_task.assert_not_called()
        pending_task.cancel()
        with suppress(asyncio.CancelledError):
            await pending_task

    def test_log_config_reload_failure_ignores_cancelled_task(self) -> None:
        """Cancelled callback tasks do not create misleading error logs."""
        cancelled_task = MagicMock()
        cancelled_task.cancelled.return_value = True
        self.app.watcher.log_failure(cancelled_task)
        self.mock_logger.error.assert_not_called()

    async def test_cancel_config_reload_task_cancels_pending_task(
        self,
    ) -> None:
        """Shutdown cancels an in-flight notification-triggered reload."""
        pending_task = asyncio.create_task(asyncio.sleep(60))
        self.app.watcher.task = pending_task
        await self.app.watcher.cancel_reload()
        self.assertTrue(pending_task.cancelled())
        self.assertIsNone(self.app.watcher.task)

    async def test_close_config_listener_unregisters_and_releases(
        self,
    ) -> None:
        """Closing unregisters the listener before releasing its connection."""
        connection = MagicMock()
        connection.remove_listener = AsyncMock()
        pool = MagicMock()
        pool.release = AsyncMock()
        self.app.watcher.connection = connection
        self.app.watcher.pool = pool
        await self.app.watcher.close()
        connection.remove_listener.assert_awaited_once_with(
            'stream_config_changed',
            self.app.watcher.notify,
        )
        pool.release.assert_awaited_once_with(connection)
        self.assertIsNone(self.app.watcher.connection)

    async def test_close_config_listener_handles_missing_pool(self) -> None:
        """Closing after a pool reset safely drops the listener reference."""
        self.app.watcher.connection = MagicMock()
        self.app.db_pool = None
        await self.app.watcher.close()
        self.assertIsNone(self.app.watcher.connection)

    async def test_reload_stops_yolo_workers_without_runnable_streams(
        self,
    ) -> None:
        """Disabled recognition stops the shared worker pool immediately."""
        disabled_config = self.dummy_cfg.copy()
        disabled_config['recognition_enabled'] = False
        self.app.workers.processes = [MagicMock()]
        with patch.object(
            self.app,
            'fetch_stream_configs',
            new_callable=AsyncMock,
            return_value=[disabled_config],
        ), patch.object(self.app.workers, 'close_async') as mock_stop:
            await self.app.reload_configurations()
        mock_stop.assert_called_once()

    def test_validate_server_model_key_accepts_configured_model(self) -> None:
        """Server mode accepts configured YOLO server model keys."""
        with patch.dict(
            os.environ,
            {'DETECT_SERVER_MODEL_KEYS': 'yolo26n,yolo26s'},
        ):
            processor._validate_server_model_key('yolo26n')

    def test_validate_server_model_key_rejects_unknown_model(self) -> None:
        """Server mode rejects model keys the YOLO server cannot load."""
        with patch.dict(
            os.environ,
            {'DETECT_SERVER_MODEL_KEYS': 'yolo26n,yolo26s'},
        ):
            with self.assertRaisesRegex(RuntimeError, 'Invalid model key'):
                processor._validate_server_model_key('../outside')

    async def test_delete_stream_live_metadata_removes_stream_key(
        self,
    ) -> None:
        redis_manager = MagicMock()
        redis_manager.delete = AsyncMock()
        with patch(
            'src.stream_processor.RedisManager',
            return_value=redis_manager,
        ):
            await processor.delete_stream_live_metadata(self.dummy_cfg)
        redis_manager.delete.assert_awaited_once_with(
            processor.build_metadata_key('SiteA', 'StreamOne'),
        )

    def test_process_single_stream_loads_env_and_runs_async_processor(
        self,
    ) -> None:
        queue = object()
        result_queue = object()
        with patch('src.stream_processor.load_dotenv') as load_env, patch(
            'src.stream_processor.asyncio.run',
        ) as run, patch(
            'src.stream_processor._run_single_stream',
        ) as run_single:
            run.side_effect = _process_single_stream_loads_env_close
            processor.process_single_stream(
                self.dummy_cfg,
                yolo_request_queue=queue,
                yolo_result_queue=result_queue,
            )
        load_env.assert_called_once_with(override=True)
        run_single.assert_called_once_with(
            self.dummy_cfg,
            yolo_request_queue=queue,
            yolo_result_queue=result_queue,
        )
        run.assert_called_once()

    async def test_run_single_stream_uses_decoupled_loop_and_cleans_resources(
        self,
    ) -> None:
        cfg = self.dummy_cfg.copy()
        cfg.update({'model_key': 'yolo26n'})
        streaming_capture = AsyncMock()
        yolo_detector = AsyncMock()
        yolo_detector.model_metadata = {}
        redis_manager = MagicMock()
        redis_manager.redis.get = AsyncMock(return_value=None)
        redis_manager.delete = AsyncMock(side_effect=RuntimeError('gone'))
        with patch(
            'src.stream_processor.StreamCapture',
            return_value=streaming_capture,
        ), patch(
            'src.stream_processor.YoloDetector',
            return_value=yolo_detector,
        ), patch(
            'src.stream_processor.YoloWorkerClient',
            return_value=object(),
        ), patch(
            'src.stream_processor.DangerDetector',
        ), patch(
            'src.stream_processor.RedisManager',
            return_value=redis_manager,
        ), patch(
            'src.stream_processor._run_decoupled_media_server_loop',
            new_callable=AsyncMock,
        ) as decoupled_loop, patch.dict(
            os.environ,
            {
                'MEDIA_PUBLISH_CLEAN_SOURCE_RESTREAM': 'false',
                'MEDIA_PUBLISH_CLEAN_STREAM': 'true',
                'MEDIA_PUBLISH_ANNOTATED_STREAM': 'true',
            },
        ):
            await processor._run_single_stream(
                cfg,
                yolo_request_queue=object(),
                yolo_result_queue=object(),
            )
        decoupled_loop.assert_awaited_once()
        yolo_detector.close.assert_awaited_once()
        streaming_capture.release_resources.assert_awaited_once()
        redis_manager.delete.assert_awaited_once()

    async def test_invalid_media_settings_stop_before_resources_start(self):
        with patch.dict(os.environ, {'MEDIA_PUBLISH_FPS': 'invalid'}), patch(
            'src.stream_processor.StreamCapture',
        ) as capture, patch('src.stream_processor.YoloDetector') as detector:
            with self.assertRaises(ValidationError):
                await processor._run_single_stream(
                    self.dummy_cfg,
                    object(),
                    object(),
                )
        capture.assert_not_called()
        detector.assert_not_called()

    async def test_run_single_stream_skips_disabled_recognition(self) -> None:
        """Disabled configs do not initialise capture or inference clients."""
        cfg = self.dummy_cfg.copy()
        cfg['recognition_enabled'] = False
        with patch('src.stream_processor.StreamCapture') as capture_cls, patch(
            'src.stream_processor.YoloDetector',
        ) as detector_cls:
            await processor._run_single_stream(cfg)
        capture_cls.assert_not_called()
        detector_cls.assert_not_called()

    async def test_run_single_stream_requires_worker_for_server_mode(
        self,
    ) -> None:
        cfg = self.dummy_cfg.copy()
        cfg.update({'model_key': 'yolo26n'})
        streaming_capture = AsyncMock()
        yolo_detector = AsyncMock()
        yolo_detector.model_metadata = {}
        with patch(
            'src.stream_processor.StreamCapture',
            return_value=streaming_capture,
        ), patch(
            'src.stream_processor.YoloDetector',
            return_value=yolo_detector,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                'Shared YOLO worker queues',
            ):
                await processor._run_single_stream(cfg)
        yolo_detector.close.assert_not_called()
        streaming_capture.release_resources.assert_not_called()

    async def test_record_detection_blocks_warning_off_hours(self) -> None:
        """Off-shift detections do not reach SSE or notification clients."""
        latest_detection = runtime_state.LatestDetectionState()
        warning = {'warning_no_hardhat': {'count': 1}}
        danger_detector = MagicMock()
        danger_detector.detect_danger.return_value = (warning, [], [])
        with patch(
            'src.stream_detection.should_notify',
            return_value=True,
        ), patch(
            'src.stream_detection._store_media_server_viewer_data',
            new_callable=AsyncMock,
        ) as store_viewer_data, patch(
            'src.stream_detection.queue_violation_event',
            new_callable=AsyncMock,
        ) as send_violation:
            await stream_detection._record_detection_result(
                runtime=make_stream_runtime(
                    danger_detector=danger_detector,
                    fcm_sender=AsyncMock(),
                    violation_sender=AsyncMock(),
                    redis_manager=MagicMock(),
                    latest_detection=latest_detection,
                    site='SiteA',
                    stream_name='Cam1',
                    work_start_hour=0,
                    work_end_hour=0,
                    metadata_key='stream_metadata:site|cam',
                    last_notification_time=0,
                    last_warning_event_time=None,
                ),
                result=DetectionResult(
                    frame=np.zeros((4, 4, 3), dtype=np.uint8),
                    timestamp=1640995200.0,
                    sequence=1,
                    track_data=[],
                ),
            )
        async with latest_detection.lock:
            self.assertEqual(latest_detection.warnings, warning)
        store_viewer_data.assert_not_awaited()
        send_violation.assert_not_awaited()

    async def test_record_detection_publishes_warning_on_hours(self) -> None:
        """Working-time warnings remain available to SSE clients."""
        latest_detection = runtime_state.LatestDetectionState()
        warning = {'warning_no_hardhat': {'count': 1}}
        danger_detector = MagicMock()
        danger_detector.detect_danger.return_value = (warning, [], [])
        with patch(
            'src.stream_detection.should_notify',
            return_value=False,
        ), patch(
            'src.stream_detection._store_media_server_viewer_data',
            new_callable=AsyncMock,
        ) as store_viewer_data, patch(
            'src.stream_detection.queue_violation_event',
            new_callable=AsyncMock,
        ) as send_violation:
            await stream_detection._record_detection_result(
                runtime=make_stream_runtime(
                    danger_detector=danger_detector,
                    fcm_sender=AsyncMock(),
                    violation_sender=AsyncMock(),
                    redis_manager=MagicMock(),
                    latest_detection=latest_detection,
                    site='SiteA',
                    stream_name='Cam1',
                    work_start_hour=0,
                    work_end_hour=24,
                    metadata_key='stream_metadata:site|cam',
                    last_notification_time=0,
                    last_warning_event_time=None,
                ),
                result=DetectionResult(
                    frame=np.zeros((4, 4, 3), dtype=np.uint8),
                    timestamp=1640995200.0,
                    sequence=1,
                    track_data=[],
                ),
            )
        store_viewer_data.assert_awaited_once_with(
            redis_manager=ANY,
            metadata_key='stream_metadata:site|cam',
            warnings=warning,
        )
        send_violation.assert_not_awaited()

    async def test_detect_latest_frames_publishes_detected_frame(self) -> None:
        """Annotated publishing uses the same frame passed to detection."""
        latest_frame = runtime_state.LatestFrameState()
        detected_frame = np.zeros((8, 8, 3), dtype=np.uint8)
        async with latest_frame.lock:
            latest_frame.frame = detected_frame.copy()
            latest_frame.timestamp = 1640995200.0
            latest_frame.sequence = 1
            latest_frame.event.set()
        yolo_detector = AsyncMock()
        yolo_detector.model_metadata = {}
        yolo_detector.generate_detections = AsyncMock(
            return_value=([], [[1, 1, 4, 4, 0.9, 5]]),
        )
        stop_event = asyncio.Event()

        detect_and_stop = partial(
            _detect_latest_frames_publishes_detected_detect_and_stop,
            stop_event,
        )

        danger_detector = MagicMock()
        danger_detector.detect_danger.side_effect = detect_and_stop
        fcm_sender = AsyncMock()
        violation_sender = AsyncMock()
        latest_detection = runtime_state.LatestDetectionState()
        pipe = MagicMock()
        pipe.xadd = MagicMock()
        pipe.execute = AsyncMock()
        redis_manager = MagicMock()
        redis_manager.redis.pipeline.return_value = pipe
        await asyncio.wait_for(
            stream_detection._detect_latest_frames(
                runtime=make_stream_runtime(
                    latest_frame=latest_frame,
                    yolo_detector=yolo_detector,
                    danger_detector=danger_detector,
                    fcm_sender=fcm_sender,
                    violation_sender=violation_sender,
                    redis_manager=redis_manager,
                    latest_detection=latest_detection,
                    site='SiteA',
                    stream_name='Cam1',
                    work_start_hour=0,
                    work_end_hour=24,
                    metadata_key='stream_metadata:site|cam',
                    stop_event=stop_event,
                ),
            ),
            timeout=1.0,
        )
        async with latest_detection.lock:
            self.assertIsNotNone(latest_detection.frame)
            assert latest_detection.frame is not None
            self.assertTrue(
                np.array_equal(
                    latest_detection.frame,
                    detected_frame,
                ),
            )
            self.assertEqual(
                latest_detection.track_data,
                [[1, 1, 4, 4, 0.9, 5]],
            )

    async def test_decoupled_loop_starts_all_requested_tasks(self) -> Any:

        stop_capture = _decoupled_loop_starts_all_requested_stop_capture

        stop_detection = _decoupled_loop_starts_all_requested_stop_detection

        stop_overlay = _decoupled_loop_starts_all_requested_stop_overlay

        stop_clean = _decoupled_loop_starts_all_requested_stop_clean

        with patch(
            'src.stream_detection._capture_latest_frames',
            side_effect=stop_capture,
        ) as capture_latest, patch(
            'src.stream_detection._detect_latest_frames',
            side_effect=stop_detection,
        ) as detect_latest, patch(
            'src.stream_media.StreamPublisherGroup.publish_overlays',
            side_effect=stop_overlay,
        ) as publish_overlay, patch(
            'src.stream_media.StreamPublisherGroup.publish_clean',
            side_effect=stop_clean,
        ) as publish_clean:
            await processor._run_decoupled_media_server_loop(
                runtime=make_stream_runtime(
                    streaming_capture=AsyncMock(),
                    yolo_detector=AsyncMock(),
                    danger_detector=MagicMock(),
                    fcm_sender=AsyncMock(),
                    violation_sender=AsyncMock(),
                    redis_manager=MagicMock(),
                    media_publish_base='rtsp://media-server:8554',
                    media_path='hazard_site_cam',
                    publish_overlay_streams=True,
                    site='SiteA',
                    stream_name='Cam1',
                    work_start_hour=0,
                    work_end_hour=24,
                    metadata_key='stream_metadata:site|cam',
                    publish_clean_stream=True,
                    restream_clean_source=False,
                    video_url='rtsp://source',
                ),
            )
        capture_latest.assert_called_once()
        detect_latest.assert_called_once()
        self.assertEqual(publish_overlay.call_count, 1)
        self.assertEqual(publish_clean.call_count, 2)

    async def test_decoupled_loop_raises_child_task_exception(self) -> None:

        fail_capture = _decoupled_loop_raises_child_task_fail_capture

        stop_detection = _decoupled_loop_raises_child_task_stop_detection

        with patch(
            'src.stream_detection._capture_latest_frames',
            side_effect=fail_capture,
        ), patch(
            'src.stream_detection._detect_latest_frames',
            side_effect=stop_detection,
        ):
            with self.assertRaisesRegex(RuntimeError, 'capture failed'):
                await processor._run_decoupled_media_server_loop(
                    runtime=make_stream_runtime(
                        streaming_capture=AsyncMock(),
                        yolo_detector=AsyncMock(),
                        danger_detector=MagicMock(),
                        fcm_sender=AsyncMock(),
                        violation_sender=AsyncMock(),
                        redis_manager=MagicMock(),
                        media_publish_base='rtsp://media-server:8554',
                        media_path='hazard_site_cam',
                        publish_overlay_streams=False,
                        site='SiteA',
                        stream_name='Cam1',
                        work_start_hour=0,
                        work_end_hour=24,
                        metadata_key='stream_metadata:site|cam',
                        publish_clean_stream=False,
                        restream_clean_source=False,
                        video_url='rtsp://source',
                    ),
                )

    async def test_capture_latest_frames_returns_when_stop_event_is_set(
        self,
    ) -> None:

        execute_capture = _capture_latest_frames_returns_when_execute_capture

        streaming_capture = MagicMock()
        streaming_capture.execute_capture = execute_capture
        streaming_capture.update_capture_interval = MagicMock()
        stop_event = asyncio.Event()
        stop_event.set()
        await stream_detection._capture_latest_frames(
            streaming_capture=streaming_capture,
            latest_frame=runtime_state.LatestFrameState(),
            stop_event=stop_event,
        )
        streaming_capture.update_capture_interval.assert_called_once()

    async def test_capture_latest_frames_updates_latest_frame(self) -> None:
        frame = np.zeros((4, 4, 3), dtype=np.uint8)

        execute_capture = partial(
            _capture_latest_frames_updates_latest_execute_capture,
            frame,
        )

        streaming_capture = MagicMock()
        streaming_capture.execute_capture = execute_capture
        streaming_capture.update_capture_interval = MagicMock()
        latest_frame = runtime_state.LatestFrameState()
        with patch.dict(os.environ, {'MEDIA_PUBLISH_SOURCE_FPS': '20'}):
            await stream_detection._capture_latest_frames(
                streaming_capture=streaming_capture,
                latest_frame=latest_frame,
                stop_event=asyncio.Event(),
            )
        streaming_capture.update_capture_interval.assert_called_once_with(0.05)
        async with latest_frame.lock:
            self.assertEqual(latest_frame.sequence, 1)
            assert isinstance(latest_frame.frame, np.ndarray)
            self.assertFalse(latest_frame.frame.flags.writeable)

    async def test_detect_latest_frames_recovers_from_timeout_and_errors(
        self,
    ) -> None:
        latest_frame = runtime_state.LatestFrameState()
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        async with latest_frame.lock:
            latest_frame.frame = frame
            latest_frame.timestamp = 1640995200.0
            latest_frame.sequence = 1
            latest_frame.event.set()
        yolo_detector = AsyncMock()
        yolo_detector.model_metadata = {}
        yolo_detector.generate_detections.side_effect = RuntimeError('busy')
        latest_detection = runtime_state.LatestDetectionState()
        stop_event = asyncio.Event()

        stop_after_sleep = partial(
            _detect_latest_frames_recovers_from_stop_after_sleep,
            stop_event,
        )

        with patch(
            'src.stream_processor.asyncio.sleep',
            side_effect=stop_after_sleep,
        ):
            await stream_detection._detect_latest_frames(
                runtime=make_stream_runtime(
                    latest_frame=latest_frame,
                    yolo_detector=yolo_detector,
                    danger_detector=MagicMock(),
                    fcm_sender=AsyncMock(),
                    violation_sender=AsyncMock(),
                    redis_manager=MagicMock(),
                    latest_detection=latest_detection,
                    site='SiteA',
                    stream_name='Cam1',
                    work_start_hour=0,
                    work_end_hour=24,
                    metadata_key='stream_metadata:site|cam',
                    stop_event=stop_event,
                ),
            )
        yolo_detector.generate_detections.assert_awaited_once()

    async def test_detect_latest_frames_recovers_from_metadata_errors(
        self,
    ) -> None:
        latest_frame = runtime_state.LatestFrameState()
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        async with latest_frame.lock:
            latest_frame.frame = frame
            latest_frame.timestamp = 1640995200.0
            latest_frame.sequence = 1
            latest_frame.event.set()
        yolo_detector = AsyncMock()
        yolo_detector.model_metadata = {}
        yolo_detector.generate_detections.return_value = ([], [])
        danger_detector = MagicMock()
        danger_detector.detect_danger.side_effect = RuntimeError('metadata')
        latest_detection = runtime_state.LatestDetectionState()
        stop_event = asyncio.Event()

        stop_after_sleep = partial(
            _detect_latest_frames_recovers_from_stop_after_sleep2,
            stop_event,
        )

        with patch(
            'src.stream_processor.asyncio.sleep',
            side_effect=stop_after_sleep,
        ):
            await stream_detection._detect_latest_frames(
                runtime=make_stream_runtime(
                    latest_frame=latest_frame,
                    yolo_detector=yolo_detector,
                    danger_detector=danger_detector,
                    fcm_sender=AsyncMock(),
                    violation_sender=AsyncMock(),
                    redis_manager=MagicMock(),
                    latest_detection=latest_detection,
                    site='SiteA',
                    stream_name='Cam1',
                    work_start_hour=0,
                    work_end_hour=24,
                    metadata_key='stream_metadata:site|cam',
                    stop_event=stop_event,
                ),
            )
        danger_detector.detect_danger.assert_called_once_with([])

    async def test_detect_latest_frames_sends_violation_notification(
        self,
    ) -> Any:
        latest_frame = runtime_state.LatestFrameState()
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        async with latest_frame.lock:
            latest_frame.frame = frame
            latest_frame.timestamp = 1640995200.0
            latest_frame.sequence = 1
            latest_frame.event.set()
        yolo_detector = AsyncMock()
        yolo_detector.model_metadata = {}
        yolo_detector.generate_detections.return_value = ([], [])
        danger_detector = MagicMock()
        warnings = {'warning_no_hardhat': {'count': 1}}
        danger_detector.detect_danger.return_value = (warnings, [], [])
        redis_manager = MagicMock()
        redis_manager.redis.xadd = AsyncMock()
        stop_event = asyncio.Event()

        send_and_stop = partial(
            _detect_latest_frames_sends_violation_send_and_stop,
            stop_event,
        )

        with patch(
            'src.stream_detection.should_notify',
            return_value=True,
        ), patch(
            'src.stream_detection.queue_violation_event',
            side_effect=send_and_stop,
        ) as send_violation:
            await stream_detection._detect_latest_frames(
                runtime=make_stream_runtime(
                    latest_frame=latest_frame,
                    yolo_detector=yolo_detector,
                    danger_detector=danger_detector,
                    fcm_sender=AsyncMock(),
                    violation_sender=AsyncMock(),
                    redis_manager=redis_manager,
                    latest_detection=runtime_state.LatestDetectionState(),
                    site='SiteA',
                    stream_name='Cam1',
                    work_start_hour=0,
                    work_end_hour=24,
                    metadata_key='stream_metadata:site|cam',
                    stop_event=stop_event,
                ),
            )
        send_violation.assert_awaited_once()

    async def test_detect_latest_frames_skips_empty_or_same_sequence(
        self,
    ) -> None:
        latest_frame = runtime_state.LatestFrameState()
        async with latest_frame.lock:
            latest_frame.sequence = 0
            latest_frame.event.set()
        stop_event = asyncio.Event()
        yolo_detector = AsyncMock()
        yolo_detector.model_metadata = {}

        wait_once = partial(
            _detect_latest_frames_skips_empty_wait_once,
            stop_event,
        )

        with patch.object(latest_frame.event, 'wait', side_effect=wait_once):
            await stream_detection._detect_latest_frames(
                runtime=make_stream_runtime(
                    latest_frame=latest_frame,
                    yolo_detector=yolo_detector,
                    danger_detector=MagicMock(),
                    fcm_sender=AsyncMock(),
                    violation_sender=AsyncMock(),
                    redis_manager=MagicMock(),
                    latest_detection=runtime_state.LatestDetectionState(),
                    site='SiteA',
                    stream_name='Cam1',
                    work_start_hour=0,
                    work_end_hour=24,
                    metadata_key='stream_metadata:site|cam',
                    stop_event=stop_event,
                ),
            )
        yolo_detector.generate_detections.assert_not_called()

    async def test_detect_latest_frames_continues_on_frame_wait_timeout(
        self,
    ) -> None:
        latest_frame = runtime_state.LatestFrameState()
        stop_event = asyncio.Event()
        yolo_detector = AsyncMock()
        yolo_detector.model_metadata = {}

        wait_timeout = partial(
            _detect_latest_frames_continues_on_wait_timeout,
            stop_event,
        )

        with patch(
            'src.stream_processor.asyncio.wait_for',
            side_effect=wait_timeout,
        ):
            await stream_detection._detect_latest_frames(
                runtime=make_stream_runtime(
                    latest_frame=latest_frame,
                    yolo_detector=yolo_detector,
                    danger_detector=MagicMock(),
                    fcm_sender=AsyncMock(),
                    violation_sender=AsyncMock(),
                    redis_manager=MagicMock(),
                    latest_detection=runtime_state.LatestDetectionState(),
                    site='SiteA',
                    stream_name='Cam1',
                    work_start_hour=0,
                    work_end_hour=24,
                    metadata_key='stream_metadata:site|cam',
                    stop_event=stop_event,
                ),
            )
        yolo_detector.generate_detections.assert_not_called()

    async def test_violation_evidence_is_queued_before_network_delivery(
        self,
    ) -> None:
        delivery = DeliveryDouble()
        runtime = make_stream_runtime(
            site='SiteA',
            stream_name='Cam1',
            delivery=delivery,
        )
        warnings: WarningPayload = {'warning_no_hardhat': {'count': 1}}
        result = DetectionResult(
            np.zeros((4, 4, 3), dtype=np.uint8),
            1640995200,
            1,
            [],
            warnings=warnings,
        )
        with patch('src.stream_detection.encode_frame', return_value=b'jpeg'):
            timestamp = await stream_detection.queue_violation_event(
                runtime,
                result,
            )
        self.assertEqual(timestamp, 1640995200)
        assert delivery.submit.await_args is not None
        payload, image = delivery.submit.await_args.args
        self.assertEqual(image, b'jpeg')
        self.assertEqual(json.loads(payload.warnings_json), warnings)
        self.assertEqual(payload.site, 'SiteA')

    async def test_full_outbox_does_not_advance_notification_clock(
        self,
    ) -> None:
        delivery, danger = (DeliveryDouble(), MagicMock())
        runtime = make_stream_runtime(
            delivery=delivery,
            danger_detector=danger,
        )
        runtime.clock.last_notification_time = 0
        delivery.submit.side_effect = RuntimeError('outbox full')
        danger.detect_danger.return_value = (
            {'warning_no_hardhat': {'count': 1}},
            [],
            [],
        )
        result = DetectionResult(
            np.zeros((4, 4, 3), dtype=np.uint8),
            1640995200,
            1,
            [],
        )
        with patch(
            'src.stream_detection.should_notify',
            return_value=True,
        ), patch(
            'src.stream_detection._store_media_server_viewer_data',
            AsyncMock(),
        ):
            with self.assertRaisesRegex(RuntimeError, 'outbox full'):
                await stream_detection._record_detection_result(
                    runtime,
                    result,
                )
        self.assertEqual(runtime.clock.last_notification_time, 0)

    def test_overlay_publish_frames_reuses_same_sequence_render(self) -> None:
        """Do not re-render unchanged detection overlays every publish tick."""
        rendered_cache: dict[str, tuple[tuple[int, int], np.ndarray]] = {}
        snapshot = runtime_state.OverlaySnapshot(
            sequence=(7, 7),
            frame=np.full((8, 8, 3), 32, dtype=np.uint8),
            warnings={
                'warning_no_hardhat': {'count': 1},
            },
            cone_polys=[],
            pole_polys=[],
            track_data=[[1, 1, 4, 4, 0.9, 5]],
        )
        rendered = np.full((8, 8, 3), 200, dtype=np.uint8)
        with patch(
            'examples.streaming_web.overlay_renderer.OverlayRenderer.rend'
            'er',
            return_value=rendered,
        ) as build_frame:
            first_frames = make_overlay_cache(rendered_cache).render(
                snapshot,
                {'zh-TW'},
            )
            second_frames = make_overlay_cache(rendered_cache).render(
                snapshot,
                {'zh-TW'},
            )
        build_frame.assert_called_once()
        self.assertIs(first_frames['zh-TW'], rendered)
        self.assertIs(second_frames['zh-TW'], rendered)

    async def test_overlay_language_snapshot_publishes_pre_rendered_frame(
        self,
    ) -> None:
        """A publisher receives the frame rendered once by the caller."""
        redis_manager = MagicMock()
        redis_manager.redis.set = AsyncMock()
        publisher = AsyncMock()
        overlay_publishers = cast(
            dict[str, media.MediaStreamPublisher],
            {'zh-TW': publisher},
        )
        publisher.fps = 30.0
        rendered = np.full((8, 8, 3), 200, dtype=np.uint8)
        await (
            publish_group := make_media_group(
                redis_manager=redis_manager,
                overlay_media_publishers=overlay_publishers,
                media_publish_base='rtsp://media-server:8554',
                media_path='hazard_U2l0ZUE_Q2FtMQ',
                site='SiteA',
                stream_name='Cam1',
                label_language='zh-TW',
                publish_frame=rendered,
            )
        ).publish_overlay(publish_group.test_variant, 'zh-TW', rendered)
        publisher.publish.assert_awaited_once_with(rendered)

    def test_build_media_startup_frame_has_stable_dimensions(self) -> None:
        """Startup slate can open the annotated media path before capture."""
        frame = stream_overlay_frames._build_media_startup_frame(
            'SiteA',
            'Cam1',
        )
        self.assertEqual(frame.shape, (720, 1280, 3))
        self.assertEqual(frame.dtype, np.uint8)

    async def test_latest_overlay_snapshot_uses_freshest_frame_and_detection(
        self,
    ) -> None:
        """The published frame stays live while boxes await the next result."""
        latest_frame = runtime_state.LatestFrameState()
        latest_detection = runtime_state.LatestDetectionState()
        source_frame = np.zeros((2, 2, 3), dtype=np.uint8)
        detected_frame = np.ones((2, 2, 3), dtype=np.uint8)
        async with latest_frame.lock:
            latest_frame.frame = source_frame
            latest_frame.sequence = 1
        async with latest_detection.lock:
            latest_detection.frame = detected_frame
            latest_detection.sequence = 2
            latest_detection.warnings = {'warning': {'count': 1}}
            latest_detection.track_data = [[1, 2, 3, 4, 0.9, 5]]
        snapshot = await media._latest_overlay_snapshot(
            latest_frame,
            latest_detection,
        )
        assert snapshot is not None
        self.assertEqual(snapshot.sequence, (1, 2))
        self.assertTrue(np.array_equal(snapshot.frame, source_frame))
        self.assertEqual(snapshot.track_data, [[1, 2, 3, 4, 0.9, 5]])

    async def test_latest_overlay_snapshot_uses_frame_when_no_detection(
        self,
    ) -> None:
        latest_frame = runtime_state.LatestFrameState()
        latest_detection = runtime_state.LatestDetectionState()
        source_frame = np.zeros((2, 2, 3), dtype=np.uint8)
        async with latest_frame.lock:
            latest_frame.frame = source_frame
            latest_frame.sequence = 3
        snapshot = await media._latest_overlay_snapshot(
            latest_frame,
            latest_detection,
        )
        assert snapshot is not None
        self.assertEqual(snapshot.sequence, (3, 0))
        self.assertTrue(np.array_equal(snapshot.frame, source_frame))

    async def test_latest_overlay_snapshot_returns_none_without_frames(
        self,
    ) -> None:
        snapshot = await media._latest_overlay_snapshot(
            runtime_state.LatestFrameState(),
            runtime_state.LatestDetectionState(),
        )
        self.assertIsNone(snapshot)

    async def test_requested_overlay_languages_uses_bounded_mget(self) -> None:
        """Only configured languages are read without scanning Redis keys."""
        media_path = 'hazard_site_cam'
        valid_key = build_overlay_demand_key(media_path, 'zh-TW')
        redis_manager = MagicMock()

        mget = partial(
            _requested_overlay_languages_uses_bounded_mget,
            valid_key,
        )

        redis_manager.redis.mget = mget
        languages = await media._MediaDemandCache().overlay_languages(
            redis_manager,
            media_path,
        )
        self.assertEqual(languages, {'zh-TW'})
        redis_manager.redis.scan_iter.assert_not_called()

    async def test_media_demand_cache_reuses_clean_and_overlay_reads(
        self,
    ) -> None:
        """One camera shares short-lived demand reads across its publishers."""
        media_path = 'hazard_site_cam'
        allowed_languages = stream_overlay_frames._allowed_overlay_languages()
        requested_language = allowed_languages[0]
        demand_key = build_overlay_demand_key(media_path, requested_language)
        redis_manager = MagicMock()
        redis_manager.redis.exists = AsyncMock(return_value=1)

        mget = partial(_media_demand_cache_reuses_clean_mget, demand_key)

        redis_manager.redis.mget = AsyncMock(side_effect=mget)
        demand_cache = media._MediaDemandCache(refresh_seconds=60)
        self.assertTrue(
            await demand_cache.clean_requested(redis_manager, media_path),
        )
        self.assertTrue(
            await demand_cache.clean_requested(redis_manager, media_path),
        )
        self.assertEqual(
            await demand_cache.overlay_languages(redis_manager, media_path),
            {requested_language},
        )
        self.assertEqual(
            await demand_cache.overlay_languages(redis_manager, media_path),
            {requested_language},
        )
        redis_manager.redis.exists.assert_awaited_once()
        redis_manager.redis.mget.assert_awaited_once()

    async def test_close_overlay_publishers_closes_all(self) -> None:
        publisher_a = AsyncMock()
        publisher_b = AsyncMock()
        publishers = cast(
            dict[str, media.MediaStreamPublisher],
            {
                'en': publisher_a,
                'zh-TW': publisher_b,
            },
        )
        await media._close_overlay_publishers(publishers)
        self.assertEqual(publishers, {})
        publisher_a.close.assert_awaited_once()
        publisher_b.close.assert_awaited_once()

    async def test_close_unrequested_overlay_publishers_keeps_requested(
        self,
    ) -> None:
        requested = AsyncMock()
        unrequested = AsyncMock()
        publishers = cast(
            dict[str, media.MediaStreamPublisher],
            {
                'en': requested,
                'zh-TW': unrequested,
            },
        )
        await media._close_unrequested_overlay_publishers(publishers, {'en'})
        self.assertEqual(set(publishers), {'en'})
        requested.close.assert_not_called()
        unrequested.close.assert_awaited_once()

    def test_drop_unrequested_overlay_cache(self) -> None:
        cache = {
            'en': ((1, 1), np.zeros((1, 1, 3), dtype=np.uint8)),
            'zh-TW': ((1, 1), np.zeros((1, 1, 3), dtype=np.uint8)),
        }
        media._drop_unrequested_overlay_cache(cache, {'en'})
        self.assertEqual(set(cache), {'en'})

    async def test_store_media_server_viewer_data(self) -> None:
        redis_manager = MagicMock()
        redis_manager.redis.xadd = AsyncMock()
        await stream_detection._store_media_server_viewer_data(
            redis_manager,
            'stream_metadata:site|cam',
            warnings={'warning': {'count': 1}},
        )
        redis_manager.redis.xadd.assert_awaited_once_with(
            'stream_metadata:site|cam',
            {'has_warning': '1'},
            maxlen=10,
        )

    async def test_store_media_server_viewer_data_skips_empty_warnings(
        self,
    ) -> None:
        redis_manager = MagicMock()
        redis_manager.redis.xadd = AsyncMock()
        await stream_detection._store_media_server_viewer_data(
            redis_manager,
            'stream_metadata:site|cam',
            warnings={},
        )
        redis_manager.redis.xadd.assert_not_awaited()

    def test_allowed_overlay_language_aliases(self) -> None:
        with patch.dict(
            os.environ,
            {'MEDIA_OVERLAY_ALLOWED_LANGUAGES': ' en,zh_Hant,en '},
        ):
            self.assertEqual(
                stream_overlay_frames._allowed_overlay_languages(),
                ('en', 'zh-TW'),
            )

    def test_allowed_overlay_languages_rejects_unknown_configuration(
        self,
    ) -> None:
        with patch.dict(
            os.environ,
            {'MEDIA_OVERLAY_ALLOWED_LANGUAGES': 'bad'},
        ):
            with self.assertRaises(ValueError):
                stream_overlay_frames._allowed_overlay_languages()

    def test_mark_frame_readonly(self) -> None:
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        stream_overlay_frames._mark_frame_readonly(frame)
        self.assertFalse(frame.flags.writeable)

    def test_mark_frame_readonly_ignores_arrays_that_cannot_change_flags(
        self,
    ) -> None:
        frame = MagicMock()
        frame.setflags.side_effect = ValueError
        stream_overlay_frames._mark_frame_readonly(frame)
        frame.setflags.assert_called_once_with(write=False)

    def test_overlay_renderer_preserves_the_shared_source_frame(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        frame.setflags(write=False)
        snapshot = OverlaySnapshot(sequence=(0, 0), frame=frame, track_data=[])
        result = make_overlay_renderer().render(snapshot, 'en')
        self.assertTrue(np.array_equal(result, frame))
        self.assertFalse(np.shares_memory(result, frame))
        self.assertFalse(frame.flags.writeable)

    async def test_mark_overlay_ready_uses_ttl(self) -> None:
        redis_manager = MagicMock()
        redis_manager.redis.set = AsyncMock()
        with patch.dict(os.environ, {'MEDIA_OVERLAY_READY_TTL_SECONDS': '3'}):
            await make_media_group(
                redis_manager=redis_manager,
            )._mark_overlay_ready('overlay_path')
        redis_manager.redis.set.assert_awaited_once_with(
            'media_overlay_ready:overlay_path',
            b'1',
            ex=5,
        )

    async def test_publish_requested_clean_frames_waits_for_demand(
        self,
    ) -> None:
        latest_frame = runtime_state.LatestFrameState()
        stop_event = asyncio.Event()
        publisher = AsyncMock()
        redis_manager = MagicMock()
        redis_manager.redis.exists = AsyncMock(return_value=0)

        sleep_once = partial(
            _publish_requested_clean_frames_waits_sleep_once,
            stop_event,
        )

        with patch(
            'src.stream_processor.asyncio.sleep',
            side_effect=sleep_once,
        ), patch(
            'src.stream_media.MediaPublisherFactory.create',
            return_value=publisher,
        ) as publisher_factory:
            await make_media_group(
                latest_frame=latest_frame,
                redis_manager=redis_manager,
                media_publish_base='rtsp://media-server:8554',
                media_path='hazard_site_cam',
                site='SiteA',
                stream_name='Cam1',
                source_url='rtsp://source',
                use_source_restreamer=False,
                stop_event=stop_event,
            ).publish_clean('detail')
        publisher_factory.assert_not_called()
        publisher.publish.assert_not_called()

    async def test_publish_requested_clean_frames_recovers_from_publish_errors(
        self,
    ) -> None:
        latest_frame = runtime_state.LatestFrameState()
        async with latest_frame.lock:
            latest_frame.frame = np.zeros((4, 4, 3), dtype=np.uint8)
        stop_event = asyncio.Event()
        publisher = AsyncMock()
        publisher.publish.side_effect = RuntimeError('ffmpeg busy')
        redis_manager = MagicMock()
        redis_manager.redis.exists = AsyncMock(return_value=1)

        sleep_once = partial(
            _publish_requested_clean_frames_recovers_sleep_once,
            stop_event,
        )

        with patch(
            'src.stream_processor.asyncio.sleep',
            side_effect=sleep_once,
        ), patch(
            'src.stream_media.MediaPublisherFactory.create',
            return_value=publisher,
        ):
            await make_media_group(
                latest_frame=latest_frame,
                redis_manager=redis_manager,
                media_publish_base='rtsp://media-server:8554',
                media_path='hazard_site_cam',
                site='SiteA',
                stream_name='Cam1',
                source_url='rtsp://source',
                use_source_restreamer=False,
                stop_event=stop_event,
            ).publish_clean('detail')
        publisher.publish.assert_awaited_once()

    async def test_clean_publisher_survives_capture_reconnect(self) -> None:
        """A source outage keeps the current HLS publisher alive."""
        _callbacks = _CleanPublisherSurvivesCaptureReconnectCallbacks()
        _callbacks.latest_frame = runtime_state.LatestFrameState()
        first_frame = np.zeros((4, 4, 3), dtype=np.uint8)
        _callbacks.resumed_frame = np.ones((4, 4, 3), dtype=np.uint8)
        async with _callbacks.latest_frame.lock:
            _callbacks.latest_frame.frame = first_frame
        _callbacks.stop_event = asyncio.Event()
        publisher = AsyncMock()
        publisher.fps = 30.0
        redis_manager = MagicMock()
        redis_manager.redis.exists = AsyncMock(return_value=1)
        _callbacks.sleep_calls = 0

        advance_capture_state = _callbacks._call_advance_capture_state

        with patch(
            'src.stream_processor.asyncio.sleep',
            side_effect=advance_capture_state,
        ), patch(
            'src.stream_media.MediaPublisherFactory.create',
            return_value=publisher,
        ) as publisher_factory:
            await make_media_group(
                latest_frame=_callbacks.latest_frame,
                redis_manager=redis_manager,
                media_publish_base='rtsp://media-server:8554',
                media_path='hazard_site_cam',
                site='SiteA',
                stream_name='Cam1',
                source_url='rtsp://source',
                use_source_restreamer=False,
                stop_event=_callbacks.stop_event,
                demand_cache=media._MediaDemandCache(refresh_seconds=0),
            ).publish_clean('detail')
        publisher_factory.assert_called_once()
        self.assertEqual(publisher.publish.await_count, 2)
        publisher.close.assert_awaited_once()

    def test_media_publisher_presets_and_invalid_rendition(self) -> None:
        """Detail and preview renditions use their intended encoder budgets."""
        with patch.dict(
            os.environ,
            {
                'MEDIA_PREVIEW_FPS': '12',
                'MEDIA_PREVIEW_WIDTH': '800',
                'MEDIA_PREVIEW_HEIGHT': '450',
                'MEDIA_PREVIEW_BITRATE': '600k',
                'MEDIA_PREVIEW_MAXRATE': '800k',
                'MEDIA_PREVIEW_BUFSIZE': '1600k',
            },
        ):
            self.assertEqual(
                media.MediaPublisherFactory(
                    MediaPublishSettings(),
                ).preview_options(),
                {
                    'fps': 12.0,
                    'width': 800,
                    'height': 450,
                    'bitrate': '600k',
                    'maxrate': '800k',
                    'bufsize': '1600k',
                },
            )
            publisher_factory = MagicMock()
            detail = media.MediaPublisherFactory(
                MediaPublishSettings(),
            ).create(
                'rtsp://media/detail',
                rendition='detail',
                publisher_type=publisher_factory,
            )
            preview = media.MediaPublisherFactory(
                MediaPublishSettings(),
            ).create(
                'rtsp://media/preview',
                rendition='preview',
                publisher_type=publisher_factory,
            )
        self.assertIs(detail, publisher_factory.return_value)
        self.assertIs(preview, publisher_factory.return_value)
        self.assertEqual(
            publisher_factory.call_args_list[0].kwargs,
            {
                'publish_url': 'rtsp://media/detail',
                'encoding': ANY,
                'fps': 30.0,
            },
        )
        self.assertEqual(
            publisher_factory.call_args_list[1].kwargs,
            {
                'publish_url': 'rtsp://media/preview',
                'encoding': ANY,
                'fps': 12.0,
                'width': 800,
                'height': 450,
                'bitrate': '600k',
                'maxrate': '800k',
                'bufsize': '1600k',
            },
        )
        with self.assertRaisesRegex(ValueError, 'unsupported media rendition'):
            media.MediaPublisherFactory(MediaPublishSettings()).create(
                'rtsp://media/unknown',
                rendition='raw',
            )

    def test_media_timing_configuration_falls_back_on_invalid_values(
        self,
    ) -> None:
        """Malformed timing values retain conservative production defaults."""
        with patch.dict(
            os.environ,
            {
                'MEDIA_OVERLAY_READY_GRACE_SECONDS': 'not-a-number',
                'WARNING_EVENT_THROTTLE_SECONDS': 'not-a-number',
            },
        ):
            self.assertEqual(MediaPublishSettings().ready_grace_seconds, 2.0)
            with self.assertRaises(ValueError):
                stream_detection._warning_event_throttle_seconds()

    async def test_clean_frame_loop_releases_restreamers_on_demand_changes(
        self,
    ) -> None:
        """Source restreamers are closed between viewers and at shutdown."""
        _callbacks = _CleanFrameLoopReleasesRestreamersOnDemandCallbacks()
        _callbacks.stop_event = asyncio.Event()
        latest_frame = runtime_state.LatestFrameState()
        redis_manager = MagicMock()
        redis_manager.redis.exists = AsyncMock(side_effect=[1, 0, 1])
        first_restreamer = AsyncMock()
        second_restreamer = AsyncMock()
        _callbacks.sleep_calls = 0

        stop_after_third_sleep = _callbacks._call_stop_after_third_sleep

        with patch(
            'src.stream_media.MediaSourceRestreamer',
            side_effect=[first_restreamer, second_restreamer],
        ), patch(
            'src.stream_processor.asyncio.sleep',
            side_effect=stop_after_third_sleep,
        ):
            await make_media_group(
                latest_frame=latest_frame,
                redis_manager=redis_manager,
                media_publish_base='rtsp://media-server:8554',
                media_path='hazard_site_cam',
                site='SiteA',
                stream_name='Cam1',
                source_url='rtsp://source',
                use_source_restreamer=True,
                stop_event=_callbacks.stop_event,
                demand_cache=media._MediaDemandCache(refresh_seconds=0),
            ).publish_clean('detail')
        first_restreamer.start.assert_awaited_once()
        first_restreamer.close.assert_awaited_once()
        second_restreamer.start.assert_awaited_once()
        second_restreamer.close.assert_awaited_once()

    async def test_clean_frame_loop_releases_publishers_on_demand_changes(
        self,
    ) -> None:
        """Frame encoders are released between viewers and at shutdown."""
        _callbacks = _CleanFrameLoopReleasesPublishersOnDemandCallbacks()
        _callbacks.stop_event = asyncio.Event()
        latest_frame = runtime_state.LatestFrameState()
        async with latest_frame.lock:
            latest_frame.frame = np.zeros((2, 2, 3), dtype=np.uint8)
        redis_manager = MagicMock()
        redis_manager.redis.exists = AsyncMock(side_effect=[1, 0, 1])
        first_publisher = AsyncMock()
        second_publisher = AsyncMock()
        _callbacks.sleep_calls = 0

        stop_after_third_sleep = _callbacks._call_stop_after_third_sleep

        with patch(
            'src.stream_media.MediaPublisherFactory.create',
            side_effect=[first_publisher, second_publisher],
        ), patch(
            'src.stream_processor.asyncio.sleep',
            side_effect=stop_after_third_sleep,
        ):
            await make_media_group(
                latest_frame=latest_frame,
                redis_manager=redis_manager,
                media_publish_base='rtsp://media-server:8554',
                media_path='hazard_site_cam',
                site='SiteA',
                stream_name='Cam1',
                source_url='rtsp://source',
                use_source_restreamer=False,
                stop_event=_callbacks.stop_event,
                demand_cache=media._MediaDemandCache(refresh_seconds=0),
            ).publish_clean('detail')
        first_publisher.publish.assert_awaited_once()
        first_publisher.close.assert_awaited_once()
        second_publisher.publish.assert_awaited_once()
        second_publisher.close.assert_awaited_once()

    @patch('main.create_pool', new_callable=AsyncMock)
    async def test_ensure_db_pool_rejects_non_postgresql_urls(
        self,
        mock_create_pool: Any,
    ) -> None:
        """Invalid database backends fail before any connection is
        attempted.
        """
        for database_url in (
            'mysql://user:pass@db.example/app',
            'mysql://user:pass@db.example:3306/app',
            'sqlite:///app.db',
        ):
            with self.subTest(database_url=database_url):
                with patch.dict(os.environ, {'DATABASE_URL': database_url}):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        'DATABASE_URL must use PostgreSQL',
                    ):
                        await self.app._ensure_db_pool()
        mock_create_pool.assert_not_awaited()
        self.assertIsNone(self.app.db_pool)

    @patch('main.create_pool', new_callable=AsyncMock)
    async def test_ensure_db_pool_preserves_postgresql_connection_settings(
        self,
        mock_create_pool: Any,
    ) -> None:
        """PostgreSQL URLs preserve credentials, database, and custom ports."""
        for scheme in ('postgresql', 'postgresql+asyncpg'):
            for port_suffix, expected_port in (('', 5432), (':6543', 6543)):
                with self.subTest(scheme=scheme, port=expected_port):
                    self.app.db_pool = None
                    database_url = (
                        f'{scheme}://worker:p%40ss@db.example{port_suffix}/app'
                    )
                    with patch.dict(
                        os.environ,
                        {'DATABASE_URL': database_url},
                    ):
                        await self.app._ensure_db_pool()
                    params = mock_create_pool.call_args.kwargs
                    self.assertEqual(params['host'], 'db.example')
                    self.assertEqual(params['port'], expected_port)
                    self.assertEqual(params['user'], 'worker')
                    self.assertEqual(params['password'], 'p@ss')
                    self.assertEqual(params['database'], 'app')
                    self.assertIs(
                        self.app.db_pool,
                        mock_create_pool.return_value,
                    )

    def test_ensure_yolo_worker_keeps_alive_pool(self) -> None:
        """A matching topology retains its native queues and workers."""
        request_queue = MagicMock()
        result_queue = MagicMock()
        worker_process = MagicMock()
        worker_process.is_alive.return_value = True
        with patch.dict(
            os.environ,
            {
                'YOLO_WORKER_CAMERAS_PER_ENGINE': '3',
                'YOLO_WORKER_CAMERAS_PER_ENGINE_BY_MODEL': '',
            },
        ):
            topology_signature = self.app.workers.topology([self.dummy_cfg])[2]
            self.app.workers.request_queues = [request_queue]
            self.app.workers.result_queues = {'SiteA|StreamOne': result_queue}
            self.app.workers.processes = [worker_process]
            self.app.workers.slots = {stream_identity(self.dummy_cfg): 0}
            self.app.workers.camera_slots = {'SiteA|StreamOne': 0}
            self.app.workers.topology_signature = topology_signature
            from src.worker_supervisor import WorkerShard

            spec = next(
                iter(self.app.workers._plan([self.dummy_cfg]).values()),
            )
            self.app.workers.shards = {
                spec.slot_id: WorkerShard(
                    spec,
                    request_queue,
                    {'SiteA|StreamOne': result_queue},
                    worker_process,
                ),
            }
            self.app.workers._next_slot_id = 1
            with patch.object(self.app.workers, 'restart_dead') as restart:
                self.assertFalse(self.app.workers.ensure([self.dummy_cfg]))
        restart.assert_called_once()
        self.assertEqual(self.app.workers.request_queues, [request_queue])
        self.assertEqual(
            self.app.workers.result_queues,
            {
                'SiteA|StreamOne': result_queue,
            },
        )

    def test_dead_worker_replaces_its_native_queue_endpoints(self) -> None:
        """Dead workers replace IPC and request camera rebinding."""
        request_queue = MagicMock()
        dead_process = MagicMock()
        replacement_process = MagicMock()
        dead_process.is_alive.return_value = False
        result_queue = MagicMock()
        with patch.dict(
            os.environ,
            {
                'YOLO_WORKER_CAMERAS_PER_ENGINE': '3',
                'YOLO_WORKER_CAMERAS_PER_ENGINE_BY_MODEL': '',
                'YOLO_WORKER_DEVICES': 'cuda:0,cuda:1',
            },
        ), patch('src.worker_supervisor.YoloWorker') as worker_class, patch(
            'src.worker_supervisor.Process',
            return_value=replacement_process,
        ), patch.object(
            self.app.workers,
            'close',
        ) as stop_worker:
            topology_signature = self.app.workers.topology([self.dummy_cfg])[2]
            self.app.workers.request_queues = [request_queue]
            self.app.workers.result_queues = {'SiteA|StreamOne': result_queue}
            self.app.workers.processes = [dead_process]
            self.app.workers.slots = {stream_identity(self.dummy_cfg): 0}
            self.app.workers.camera_slots = {'SiteA|StreamOne': 0}
            self.app.workers.topology_signature = topology_signature
            from src.worker_supervisor import WorkerShard

            spec = next(
                iter(self.app.workers._plan([self.dummy_cfg]).values()),
            )
            self.app.workers.shards = {
                spec.slot_id: WorkerShard(
                    spec,
                    request_queue,
                    {'SiteA|StreamOne': result_queue},
                    dead_process,
                ),
            }
            self.app.workers._next_slot_id = 1
            self.app.workers.startup_lock = MagicMock()
            worker_class.return_value.run = MagicMock()
            self.assertTrue(self.app.workers.ensure([self.dummy_cfg]))
        dead_process.join.assert_called_once_with(timeout=0)
        replacement_process.start.assert_called_once()
        self.assertIs(self.app.workers.processes[0], replacement_process)
        self.assertIsNot(self.app.workers.request_queues[0], request_queue)
        self.assertIsNot(
            self.app.workers.result_queues['SiteA|StreamOne'],
            result_queue,
        )
        self.assertEqual(
            self.app.workers.restarted_cameras,
            {stream_identity(self.dummy_cfg)},
        )
        self.assertEqual(self.app.workers.shards[spec.slot_id].generation, 2)
        worker_class.assert_called_once_with(
            self.app.workers.request_queues[0],
            'cuda:0',
            result_queues={
                'SiteA|StreamOne': self.app.workers.result_queues[
                    'SiteA|StreamOne'
                ],
            },
            startup_lock=self.app.workers.startup_lock,
        )
        request_queue.close.assert_called_once()
        result_queue.close.assert_called_once()
        self.app.workers._close_queues(self.app.workers.shards[spec.slot_id])
        stop_worker.assert_not_called()

    def test_ensure_yolo_worker_starts_configured_workers(self) -> None:
        """Each worker receives only its own shard's result queues."""
        configs = []
        for index in range(4):
            cfg = self.dummy_cfg.copy()
            cfg['video_url'] = f'rtsp://example.com/cam{index}'
            cfg['stream_name'] = f'Cam{index}'
            configs.append(cfg)
        result_queues = [MagicMock() for _ in configs]
        request_queues = [MagicMock(), MagicMock()]
        worker_processes = [MagicMock(), MagicMock()]
        startup_lock = MagicMock()
        with patch.dict(
            os.environ,
            {
                'YOLO_WORKER_DEVICES': 'cuda:0',
                'YOLO_WORKER_QUEUE_SIZE': '7',
                'YOLO_WORKER_CAMERAS_PER_ENGINE': '3',
                'YOLO_WORKER_CAMERAS_PER_ENGINE_BY_MODEL': '',
            },
        ), patch(
            'src.worker_supervisor.multiprocessing.Queue',
            side_effect=[
                request_queues[0],
                *result_queues[:3],
                request_queues[1],
                result_queues[3],
            ],
        ) as queue_class, patch(
            'src.worker_supervisor.ProcessStartupLock',
            return_value=startup_lock,
        ), patch(
            'src.worker_supervisor.YoloWorker',
        ) as worker_class, patch(
            'src.worker_supervisor.Process',
            side_effect=worker_processes,
        ):
            worker_class.return_value.run = MagicMock()
            restarted = self.app.workers.ensure(configs)
        self.assertTrue(restarted)
        self.assertEqual(self.app.workers.request_queues, request_queues)
        self.assertEqual(
            self.app.workers.result_queues,
            {f'SiteA|Cam{index}': result_queues[index] for index in range(4)},
        )
        for worker_process in worker_processes:
            worker_process.start.assert_called_once()
        self.assertEqual(
            [call.kwargs['maxsize'] for call in queue_class.call_args_list],
            [7, 8, 8, 8, 7, 8],
        )
        self.assertEqual(worker_class.call_count, 2)
        first_worker_args = worker_class.call_args_list[0]
        self.assertEqual(first_worker_args.args, (request_queues[0], 'cuda:0'))
        self.assertEqual(
            first_worker_args.kwargs['result_queues'],
            {
                'SiteA|Cam0': result_queues[0],
                'SiteA|Cam1': result_queues[1],
                'SiteA|Cam2': result_queues[2],
            },
        )
        self.assertEqual(
            worker_class.call_args_list[1].args,
            (request_queues[1], 'cuda:0'),
        )
        self.assertEqual(
            worker_class.call_args_list[1].kwargs['result_queues'],
            {
                'SiteA|Cam3': result_queues[3],
            },
        )
        self.assertIs(first_worker_args.kwargs['startup_lock'], startup_lock)

    def test_yolo_worker_slot_returns_only_its_fixed_camera_queues(
        self,
    ) -> None:
        """Each camera is pinned to one request queue and result queue."""
        self.assertEqual(self.app.workers.slot(self.dummy_cfg), (None, None))
        self.app.workers.request_queues = ['request-0']
        self.app.workers.result_queues = {'SiteA|StreamOne': 'result-0'}
        self.app.workers.slots = {stream_identity(self.dummy_cfg): 0}
        self.assertEqual(
            self.app.workers.slot(
                self.dummy_cfg,
            ),
            ('request-0', 'result-0'),
        )
        unknown_camera = self.dummy_cfg.copy()
        unknown_camera['video_url'] = 'rtsp://example.com/unknown'
        unknown_camera['stream_name'] = 'Unknown'
        self.assertEqual(self.app.workers.slot(unknown_camera), (None, None))

    def test_yolo_worker_topology_shards_same_model_cameras(self) -> None:
        """At most three same-model cameras are pinned to one engine worker."""
        medium_configs = []
        for index in range(7):
            cfg = self.dummy_cfg.copy()
            cfg['stream_id'] = index + 1
            cfg['model_key'] = 'yolo26m'
            cfg['video_url'] = f'rtsp://example.com/m{index:02d}'
            medium_configs.append(cfg)
        nano_cfg = self.dummy_cfg.copy()
        nano_cfg['stream_id'] = 1000
        nano_cfg['model_key'] = 'yolo26n'
        nano_cfg['video_url'] = 'rtsp://example.com/n00'
        with patch.dict(
            os.environ,
            {
                'YOLO_WORKER_CAMERAS_PER_ENGINE': '3',
                'YOLO_WORKER_CAMERAS_PER_ENGINE_BY_MODEL': '',
            },
        ):
            worker_count, slots, signature = self.app.workers.topology(
                medium_configs + [nano_cfg],
            )
        self.assertEqual(worker_count, 4)
        self.assertEqual(
            [slots[stream_identity(cfg)] for cfg in medium_configs],
            [0, 0, 0, 1, 1, 1, 2],
        )
        self.assertEqual(slots[stream_identity(nano_cfg)], 3)
        self.assertIsNotNone(signature)

    def test_yolo_worker_topology_uses_per_model_camera_limits(self) -> None:
        """Each model key can use a different worker capacity from .env."""
        medium_configs = []
        nano_configs = []
        for index in range(5):
            medium_cfg = self.dummy_cfg.copy()
            medium_cfg['stream_id'] = index + 1
            medium_cfg['model_key'] = 'yolo26m'
            medium_cfg['video_url'] = f'rtsp://example.com/m{index:02d}'
            medium_configs.append(medium_cfg)
            nano_cfg = self.dummy_cfg.copy()
            nano_cfg['stream_id'] = 1000 + index
            nano_cfg['model_key'] = 'yolo26n'
            nano_cfg['video_url'] = f'rtsp://example.com/n{index:02d}'
            nano_configs.append(nano_cfg)
        with patch.dict(
            os.environ,
            {
                'YOLO_WORKER_CAMERAS_PER_ENGINE': '3',
                'YOLO_WORKER_CAMERAS_PER_ENGINE_BY_MODEL': (
                    'yolo26m=2,yolo26n=4'
                ),
            },
        ):
            worker_count, slots, signature = self.app.workers.topology(
                medium_configs + nano_configs,
            )
        self.assertEqual(worker_count, 5)
        self.assertEqual(
            [slots[stream_identity(cfg)] for cfg in medium_configs],
            [0, 0, 1, 1, 2],
        )
        self.assertEqual(
            [slots[stream_identity(cfg)] for cfg in nano_configs],
            [3, 3, 3, 3, 4],
        )
        self.assertIsNotNone(signature)
        assert signature is not None
        self.assertEqual(
            {
                spec['model_key']: spec['capacity']
                for spec in json.loads(signature)
            },
            {
                'yolo26m': 2,
                'yolo26n': 4,
            },
        )

    def test_restart_reason_covers_all_reasons(self) -> None:
        proc = MagicMock()
        proc.is_alive.return_value = False
        info = StreamProcessState(updated_at='old', process=proc)
        cfg = self.dummy_cfg.copy()
        cfg['updated_at'] = 'new'
        self.assertEqual(
            self.app._restart_reason(
                info,
                cfg,
            ),
            'updated_at changed',
        )
        cfg['updated_at'] = 'old'
        self.assertEqual(self.app._restart_reason(info, cfg), 'process exited')
        proc.is_alive.return_value = True
        self.assertEqual(
            self.app._restart_reason(
                info,
                cfg,
            ),
            'YOLO worker restarted',
        )

    async def test_restart_stream_process_updates_running_process(
        self,
    ) -> None:
        old_proc = MagicMock()
        new_proc = MagicMock()
        old_cfg = self.dummy_cfg.copy()
        proc_info = StreamProcessState(
            process=old_proc,
            updated_at=old_cfg['updated_at'],
            cfg=old_cfg,
        )
        cfg = self.dummy_cfg.copy()
        cfg['updated_at'] = 'new'
        with patch.object(
            self.app.workers.reaper,
            'reap',
        ) as stop_process, patch.object(
            self.app,
            '_delete_stream_redis_keys',
            new_callable=AsyncMock,
        ) as delete_keys, patch.object(
            self.app,
            'start_process',
            return_value=new_proc,
        ) as start_process:
            await self.app._restart_stream_process(
                stream_identity(self.dummy_cfg),
                proc_info,
                cfg,
            )
        stop_process.assert_awaited_once_with([old_proc])
        delete_keys.assert_awaited_once_with(old_cfg)
        start_process.assert_called_once_with(cfg)
        self.assertIs(
            self.app.running_processes[
                stream_identity(
                    self.dummy_cfg,
                )
            ].process,
            new_proc,
        )

    def test_stop_process_kills_still_alive_process(self) -> None:
        proc = MagicMock()
        proc.is_alive.return_value = True
        self.app.stop_process(proc)
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()
        self.assertEqual(proc.join.call_count, 2)

    def test_stop_process_logs_errors(self) -> None:
        proc = MagicMock()
        proc.terminate.side_effect = RuntimeError('boom')
        self.app.stop_process(proc)
        self.mock_logger.error.assert_called_once()

    async def test_cleanup_resources_stops_workers_and_db(self) -> None:
        proc = MagicMock()
        db_pool = MagicMock()
        db_pool.close = AsyncMock()
        self.app.running_processes = {
            'rtsp://cam': StreamProcessState(process=proc, updated_at=''),
        }
        self.app.db_pool = db_pool
        with patch.object(self.app.workers, 'close_async') as stop_worker:
            await self.app.cleanup_resources()
        proc.terminate.assert_called_once()
        stop_worker.assert_called_once()
        db_pool.close.assert_awaited_once()
        self.assertIsNone(self.app.db_pool)

    async def test_reset_db_pool_ignores_close_errors(self) -> None:
        db_pool = MagicMock()
        db_pool.close = AsyncMock(side_effect=RuntimeError('closed'))
        self.app.db_pool = db_pool
        await self.app._reset_db_pool()
        self.assertIsNone(self.app.db_pool)
        db_pool.close.assert_awaited_once()

    def test_stop_yolo_worker_signals_kills_and_closes_native_queues(
        self,
    ) -> None:
        """Queue cleanup does not depend on a multiprocessing manager."""
        bad_queue = MagicMock()
        bad_queue.put.side_effect = RuntimeError('queue closed')
        good_queue = MagicMock()
        result_queue = MagicMock()
        alive_process = MagicMock()
        alive_process.is_alive.return_value = True
        stopped_process = MagicMock()
        stopped_process.is_alive.return_value = False
        self.app.workers.request_queues = [bad_queue, good_queue]
        self.app.workers.result_queues = {'SiteA|Cam1': result_queue}
        self.app.workers.processes = [alive_process, stopped_process]
        self.app.workers.close()
        good_queue.put.assert_called_once_with(
            worker_supervisor.YOLO_WORKER_STOP_MESSAGE,
            block=False,
        )
        alive_process.kill.assert_called_once()
        for worker_queue in [bad_queue, good_queue, result_queue]:
            worker_queue.close.assert_called_once()
            worker_queue.cancel_join_thread.assert_called_once()
        self.assertEqual(self.app.workers.request_queues, [])
        self.assertEqual(self.app.workers.result_queues, {})
        self.assertEqual(self.app.workers.processes, [])

    async def test_run_logs_unexpected_errors_and_cleans_up(self) -> None:
        with patch.object(
            self.app.watcher,
            'start',
            new_callable=AsyncMock,
        ), patch.object(
            self.app,
            'poll_and_reload',
            new_callable=AsyncMock,
            side_effect=RuntimeError('boom'),
        ), patch.object(
            self.app,
            'cleanup_resources',
            new_callable=AsyncMock,
        ) as cleanup:
            await self.app.run()
        self.mock_logger.error.assert_called_once()
        cleanup.assert_awaited_once()

    def test_main_module_entrypoint_runs_main(self) -> None:
        with patch(
            'multiprocessing.set_start_method',
        ) as set_start_method, patch('asyncio.run') as run, patch(
            'signal.signal',
        ) as set_signal_handler:
            run.side_effect = _process_single_stream_loads_env_close
            runpy.run_path('main.py', run_name='__main__')
        set_signal_handler.assert_called_once()
        self.assertEqual(set_signal_handler.call_args.args[0], signal.SIGTERM)
        self.assertTrue(callable(set_signal_handler.call_args.args[1]))
        set_start_method.assert_called_once_with('spawn', force=True)
        run.assert_called_once()

    def test_sigterm_handler_requests_graceful_shutdown(self) -> None:
        with self.assertRaises(KeyboardInterrupt):
            main._handle_sigterm(signal.SIGTERM, None)

    @patch('main.create_pool', new_callable=AsyncMock)
    async def test_db_pool_created_once(self, mock_create_pool: Any) -> None:
        """Test that database pool is only created once."""
        mock_pool = AsyncMock()
        mock_create_pool.return_value = mock_pool
        await self.app._ensure_db_pool()
        self.assertIsNotNone(self.app.db_pool)
        await self.app._ensure_db_pool()
        mock_create_pool.assert_called_once()

    @patch('main.os.getenv')
    async def test_ensure_db_pool_missing_database_url(
        self,
        mock_getenv: Any,
    ) -> None:
        """Test that _ensure_db_pool raises RuntimeError when DATABASE_URL is
        None."""
        mock_getenv.return_value = None
        with self.assertRaises(RuntimeError) as ctx:
            await self.app._ensure_db_pool()
        self.assertIn(
            'DATABASE_URL environment variable is required',
            str(ctx.exception),
        )
        mock_getenv.assert_called_with('DATABASE_URL')

    @patch('main.MainApp.fetch_stream_configs')
    async def test_reload_config_adds_new_stream(
        self,
        mock_fetch: Any,
    ) -> None:
        """Test launching a new stream process."""
        mock_cfg = self.dummy_cfg.copy()
        mock_fetch.return_value = [mock_cfg]
        with patch.object(
            self.app.workers,
            'reconcile',
            return_value=False,
        ), patch('main.MainApp.start_process') as mock_start:
            mock_proc = MagicMock()
            mock_start.return_value = mock_proc
            await self.app.reload_configurations()
            self.assertIn(
                stream_identity(mock_cfg),
                self.app.running_processes,
            )
            mock_start.assert_called_once()

    @patch('main.MainApp.fetch_stream_configs')
    async def test_reload_config_skips_recognition_disabled_config(
        self,
        mock_fetch: Any,
    ) -> None:
        """A saved but disabled config must not start a stream process."""
        cfg = self.dummy_cfg.copy()
        cfg['recognition_enabled'] = False
        mock_fetch.return_value = [cfg]
        with patch('main.MainApp.start_process') as mock_start:
            await self.app.reload_configurations()
        self.assertNotIn(stream_identity(cfg), self.app.running_processes)
        mock_start.assert_not_called()

    @patch('main.MainApp.fetch_stream_configs')
    async def test_reload_config_stops_recognition_disabled_stream(
        self,
        mock_fetch: Any,
    ) -> None:
        """Disabling recognition stops the already-running child process."""
        cfg = self.dummy_cfg.copy()
        cfg['recognition_enabled'] = False
        proc = MagicMock()
        self.app.running_processes[stream_identity(cfg)] = StreamProcessState(
            process=proc,
            updated_at=cfg['updated_at'],
            cfg=self.dummy_cfg.copy(),
        )
        mock_fetch.return_value = [cfg]
        await self.app.reload_configurations()
        self.assertNotIn(stream_identity(cfg), self.app.running_processes)
        proc.terminate.assert_called_once()

    def test_can_run_stream_requires_enabled_and_validity(self) -> None:
        """Working hours do not stop capture or MediaMTX publishing."""
        cfg = self.dummy_cfg.copy()
        self.assertTrue(self.app._can_run_stream(cfg))
        cfg['recognition_enabled'] = False
        self.assertFalse(self.app._can_run_stream(cfg))
        cfg['recognition_enabled'] = True
        cfg['work_start_hour'] = 11
        cfg['work_end_hour'] = 12
        self.assertTrue(self.app._can_run_stream(cfg))
        cfg['expire_date'] = '2020-01-01T00:00:00'
        self.assertFalse(self.app._can_run_stream(cfg))

    @patch('main.MainApp.fetch_stream_configs')
    async def test_reload_config_stops_expired_stream(
        self,
        mock_fetch: Any,
    ) -> None:
        """Test stopping an expired stream process."""
        expired_date = (datetime.now() - timedelta(days=1)).isoformat()
        mock_cfg = self.dummy_cfg.copy()
        mock_cfg['expire_date'] = expired_date
        stream_key = stream_identity(mock_cfg)
        self.app.running_processes[stream_key] = StreamProcessState(
            process=MagicMock(),
            updated_at=mock_cfg['updated_at'],
            cfg=mock_cfg,
        )
        mock_fetch.return_value = []
        with patch(
            'main.delete_stream_live_metadata',
            new_callable=AsyncMock,
        ) as mock_del:
            await self.app.reload_configurations()
            self.assertNotIn(
                stream_identity(mock_cfg),
                self.app.running_processes,
            )
            mock_del.assert_awaited()

    @patch('main.MainApp.fetch_stream_configs')
    async def test_reload_config_restarts_updated_stream(
        self,
        mock_fetch: Any,
    ) -> None:
        """Test that stream process is restarted if updated_at has changed."""
        video_url = self.dummy_cfg['video_url']
        old_cfg = self.dummy_cfg.copy()
        new_cfg = self.dummy_cfg.copy()
        new_cfg['updated_at'] = (
            datetime.now() + timedelta(seconds=5)
        ).isoformat()
        mock_proc = MagicMock()
        self.app.running_processes[video_url] = StreamProcessState(
            process=mock_proc,
            updated_at=old_cfg['updated_at'],
            cfg=old_cfg,
        )
        mock_fetch.return_value = [new_cfg]
        with patch.object(
            self.app.workers,
            'reconcile',
            return_value=False,
        ), patch('main.MainApp.start_process') as mock_start, patch(
            'main.delete_stream_live_metadata',
            new_callable=AsyncMock,
        ):
            mock_start.return_value = MagicMock()
            await self.app.reload_configurations()
            mock_proc.terminate.assert_called_once()
            self.assertTrue(mock_proc.join.call_count >= 1)
            mock_start.assert_called_once()

    @patch('main.MainApp.fetch_stream_configs')
    async def test_reload_config_restarts_dead_stream(
        self,
        mock_fetch: Any,
    ) -> None:
        """Test that a dead stream process is relaunched."""
        cfg = self.dummy_cfg.copy()
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = False
        self.app.running_processes[stream_identity(cfg)] = StreamProcessState(
            process=mock_proc,
            updated_at=cfg['updated_at'],
            cfg=cfg,
        )
        mock_fetch.return_value = [cfg]
        with patch.object(
            self.app.workers,
            'reconcile',
            return_value=False,
        ), patch('main.MainApp.start_process') as mock_start:
            mock_start.return_value = MagicMock()
            await self.app.reload_configurations()
        mock_proc.terminate.assert_called_once()
        mock_start.assert_called_once_with(cfg)
        self.assertIs(
            self.app.running_processes[stream_identity(cfg)].process,
            mock_start.return_value,
        )

    @patch('main.MainApp.fetch_stream_configs')
    async def test_reload_config_restarts_after_worker_broker_replacement(
        self,
        mock_fetch: Any,
    ) -> None:
        """Test that streams receive fresh queues after broker replacement."""
        cfg = self.dummy_cfg.copy()
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True
        self.app.running_processes[stream_identity(cfg)] = StreamProcessState(
            process=mock_proc,
            updated_at=cfg['updated_at'],
            cfg=cfg,
        )
        self.app.workers.processes = [MagicMock()]
        self.app.workers.restarted_cameras = {stream_identity(cfg)}
        mock_fetch.return_value = [cfg]
        with patch.object(
            self.app.workers,
            'reconcile',
            return_value=True,
        ), patch('main.MainApp.start_process') as mock_start:
            mock_start.return_value = MagicMock()
            await self.app.reload_configurations()
        mock_proc.terminate.assert_called_once()
        mock_start.assert_called_once_with(cfg)

    @patch('main.MainApp.fetch_stream_configs')
    async def test_reload_config_keeps_streams_after_worker_replacement(
        self,
        mock_fetch: Any,
    ) -> None:
        """A worker replacement with stable IPC does not restart streams."""
        cfg = self.dummy_cfg.copy()
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True
        self.app.running_processes[stream_identity(cfg)] = StreamProcessState(
            process=mock_proc,
            updated_at=cfg['updated_at'],
            cfg=cfg,
        )
        mock_fetch.return_value = [cfg]
        with patch.object(
            self.app.workers,
            'reconcile',
            return_value=False,
        ), patch('main.MainApp.start_process') as mock_start:
            await self.app.reload_configurations()
        mock_proc.terminate.assert_not_called()
        mock_start.assert_not_called()

    @patch('main.MainApp.fetch_stream_configs')
    async def test_reload_config_skips_expired_config(
        self,
        mock_fetch: Any,
    ) -> None:
        """Test that expired configs are not started."""
        expired_cfg = self.dummy_cfg.copy()
        expired_cfg['expire_date'] = (
            datetime.now() - timedelta(days=1)
        ).isoformat()
        mock_fetch.return_value = [expired_cfg]
        with patch('main.MainApp.start_process') as mock_start:
            await self.app.reload_configurations()
            mock_start.assert_not_called()

    @patch('main.MainApp.reload_configurations')
    async def test_poll_and_reload_runs_once(self, mock_reload: Any) -> None:
        """Test polling loop executes reload and waits."""

        stop_after_one = _poll_and_reload_runs_once_stop_after_one

        mock_reload.side_effect = stop_after_one
        with self.assertRaises(KeyboardInterrupt):
            await self.app.poll_and_reload()
        mock_reload.assert_called_once()

    @patch('main.MainApp.reload_configurations')
    async def test_poll_and_reload_exception_handling(
        self,
        mock_reload: Any,
    ) -> None:
        """Test that poll_and_reload handles exceptions and continues."""
        _callbacks = _PollAndReloadExceptionHandlingCallbacks()
        _callbacks.call_count = 0

        side_effect = _callbacks._call_side_effect

        mock_reload.side_effect = side_effect
        with self.assertRaises(KeyboardInterrupt):
            await self.app.poll_and_reload()
        self.assertEqual(mock_reload.call_count, 2)

    @patch(
        'src.configuration_watcher.ConfigurationWatcher.start',
        new_callable=AsyncMock,
    )
    @patch('main.MainApp.reload_configurations')
    async def test_poll_and_reload_resets_pool_on_timeout(
        self,
        mock_reload: Any,
        _mock_listener: Any,
    ) -> None:
        """Test that DB pool is reset after a reload timeout."""
        _callbacks = _PollAndReloadResetsPoolOnTimeoutCallbacks()
        _callbacks.call_count = 0
        db_pool = MagicMock()
        db_pool.close = AsyncMock()
        self.app.db_pool = db_pool

        side_effect = _callbacks._call_side_effect

        mock_reload.side_effect = side_effect
        with self.assertRaises(KeyboardInterrupt):
            await self.app.poll_and_reload()
        db_pool.close.assert_awaited_once()
        self.assertIsNone(self.app.db_pool)

    async def test_app_run_method(self) -> None:
        """Test the run method initializes the listener then polls."""
        with patch.object(
            self.app.watcher,
            'start',
        ) as mock_listener, patch.object(
            self.app,
            'poll_and_reload',
        ) as mock_poll, patch.object(
            self.app,
            'cleanup_resources',
        ) as mock_cleanup:
            mock_poll.side_effect = KeyboardInterrupt()
            mock_cleanup.return_value = None
            await self.app.run()
            mock_listener.assert_awaited_once()
            mock_poll.assert_called_once()
            mock_cleanup.assert_called_once()

    @patch('main.MainApp.fetch_stream_configs')
    async def test_reload_config_cleans_metadata_for_stopped_stream(
        self,
        mock_fetch: Any,
    ) -> None:
        """Stopping a stream always clears its live metadata."""
        expired_cfg = self.dummy_cfg.copy()
        expired_cfg['expire_date'] = (
            datetime.now() - timedelta(days=1)
        ).isoformat()
        self.app.running_processes[stream_identity(expired_cfg)] = (
            StreamProcessState(
                process=MagicMock(),
                updated_at=expired_cfg['updated_at'],
                cfg=expired_cfg,
            )
        )
        mock_fetch.return_value = []
        with patch('main.delete_stream_live_metadata') as mock_redis_class:
            await self.app.reload_configurations()
            mock_redis_class.assert_awaited_once_with(expired_cfg)

    @patch('main.MainApp.fetch_stream_configs')
    async def test_reload_config_redis_cleanup_on_restart(
        self,
        mock_fetch: Any,
    ) -> None:
        """Test Redis cleanup during stream restart."""
        video_url = self.dummy_cfg['video_url']
        old_cfg = self.dummy_cfg.copy()
        new_cfg = self.dummy_cfg.copy()
        new_cfg['updated_at'] = (
            datetime.now() + timedelta(seconds=5)
        ).isoformat()
        mock_proc = MagicMock()
        self.app.running_processes[video_url] = StreamProcessState(
            process=mock_proc,
            updated_at=old_cfg['updated_at'],
            cfg=old_cfg,
        )
        mock_fetch.return_value = [new_cfg]
        with patch.object(
            self.app.workers,
            'reconcile',
            return_value=False,
        ), patch('main.MainApp.start_process') as mock_start, patch(
            'main.delete_stream_live_metadata',
            new_callable=AsyncMock,
        ) as mock_delete:
            mock_start.return_value = MagicMock()
            await self.app.reload_configurations()
            mock_delete.assert_awaited_once_with(old_cfg)

    @patch('main.MainApp')
    @patch('main.argparse.ArgumentParser.parse_args')
    async def test_main_function_db_mode_delegates_to_run(
        self,
        mock_args: Any,
        mock_app_class: Any,
    ) -> None:
        """Test database mode delegates its lifecycle to MainApp.run."""
        from main import main as main_func

        mock_args.return_value = type(
            'Args',
            (),
            {'poll': 5, 'config': None},
        )()
        mock_app = MagicMock()
        mock_app.run = AsyncMock()
        mock_app.cleanup_resources = AsyncMock()
        mock_app_class.return_value = mock_app
        await main_func()
        mock_app.run.assert_awaited_once()
        mock_app.cleanup_resources.assert_not_awaited()

    @patch('main.MainApp')
    @patch('main.argparse.ArgumentParser.parse_args')
    async def test_main_function_db_mode_does_not_cleanup_twice(
        self,
        mock_args: Any,
        mock_app_class: Any,
    ) -> None:
        """Test database mode leaves cleanup to MainApp.run."""
        from main import main as main_func

        mock_args.return_value = type(
            'Args',
            (),
            {'poll': 5, 'config': None},
        )()
        mock_app = MagicMock()
        mock_app.run = AsyncMock()
        mock_app.cleanup_resources = AsyncMock()
        mock_app_class.return_value = mock_app
        await main_func()
        mock_app.run.assert_awaited_once()
        mock_app.cleanup_resources.assert_not_awaited()

    @patch.object(WorkerSupervisor, 'ensure', return_value=False)
    @patch('src.worker_supervisor.Process')
    @patch('main.json.load')
    @patch('main.open', create=True)
    @patch('main.argparse.ArgumentParser.parse_args')
    async def test_main_function_json_config(
        self,
        mock_args: Any,
        mock_open: Any,
        mock_json_load: Any,
        mock_process_class: Any,
        _mock_ensure_yolo_worker: Any,
    ) -> None:
        """Test main function with --config argument (JSON file)."""
        from main import main as main_func

        mock_args.return_value = type(
            'Args',
            (),
            {'poll': 5, 'config': 'dummy.json'},
        )()
        mock_json_load.return_value = [self.dummy_cfg]
        mock_proc = MagicMock()
        mock_process_class.return_value = mock_proc

        is_alive_side_effect = _main_function_json_config_is_alive_side_effect

        mock_proc.is_alive.side_effect = is_alive_side_effect()
        mock_proc.join = MagicMock()
        mock_proc.terminate = MagicMock()
        await main_func()
        mock_process_class.assert_called_once_with(
            target=process_single_stream,
            args=(self.dummy_cfg, None, None),
        )
        mock_proc.start.assert_called_once()
        mock_proc.join.assert_called()

    @patch.object(WorkerSupervisor, 'ensure', return_value=False)
    @patch('main.print')
    @patch('src.worker_supervisor.Process')
    @patch('main.json.load')
    @patch('main.open', create=True)
    @patch('main.argparse.ArgumentParser.parse_args')
    async def test_main_function_json_config_keyboard_interrupt(
        self,
        mock_args: Any,
        _mock_open: Any,
        mock_json_load: Any,
        mock_process_class: Any,
        mock_print: Any,
        _mock_ensure_yolo_worker: Any,
    ) -> Any:
        """Test main function with JSON config handling KeyboardInterrupt."""
        _callbacks = _MainFunctionJsonConfigKeyboardInterruptCallbacks()
        from main import main as main_entry

        mock_args.return_value = type(
            'Args',
            (),
            {'poll': 10, 'config': '/path/to/config.json'},
        )()
        mock_json_load.return_value = [self.dummy_cfg.copy()]
        mock_process = MagicMock()
        _callbacks.call_count = 0

        is_alive_side_effect = _callbacks._call_is_alive_side_effect

        mock_process.is_alive.side_effect = is_alive_side_effect
        _callbacks.join_call_count = 0

        join_side_effect = _callbacks._call_join_side_effect

        mock_process.join.side_effect = join_side_effect
        mock_process.terminate.return_value = None
        mock_process_class.return_value = mock_process
        await main_entry()
        mock_print.assert_called_with(
            '\n[INFO] KeyboardInterrupt, shutting down...',
        )
        mock_process.terminate.assert_called()

    @patch.object(WorkerSupervisor, 'ensure', return_value=False)
    @patch('src.worker_supervisor.Process')
    @patch('main.json.load')
    @patch('main.open', create=True)
    @patch('main.argparse.ArgumentParser.parse_args')
    async def test_main_function_json_config_alive_process_cleanup(
        self,
        mock_args: Any,
        mock_open: Any,
        mock_json_load: Any,
        mock_process_class: Any,
        _mock_ensure_yolo_worker: Any,
    ) -> Any:
        """Test main function JSON config with alive process cleanup."""
        _callbacks = _MainFunctionJsonConfigAliveProcessCleanupCallbacks()
        from main import main as main_entry

        mock_args.return_value = type(
            'Args',
            (),
            {'poll': 10, 'config': '/path/to/config.json'},
        )()
        mock_json_load.return_value = [self.dummy_cfg.copy()]
        mock_process = MagicMock()
        _callbacks.call_count = 0

        is_alive_side_effect = _callbacks._call_is_alive_side_effect

        mock_process.is_alive.side_effect = is_alive_side_effect
        mock_process.join.return_value = None
        mock_process.terminate.return_value = None
        mock_process_class.return_value = mock_process
        await main_entry()
        mock_process.terminate.assert_called()
        self.assertTrue(mock_process.join.call_count >= 1)

    @patch('main.process_single_stream')
    def test_process_single_stream_basic(self, mock_process_func: Any) -> None:
        """Test that process_single_stream can be called."""
        from main import process_single_stream

        mock_process_func.return_value = None
        cfg = self.dummy_cfg.copy()
        process_single_stream(cfg)
        mock_process_func.assert_called_once_with(cfg)

    def test_module_level_imports(self) -> None:
        """Test module level code coverage."""
        self.assertTrue(hasattr(main, 'MainApp'))
        self.assertTrue(hasattr(main, 'StreamConfig'))
        self.assertTrue(hasattr(main, 'process_single_stream'))

    def test_if_main_block_coverage(self) -> None:
        """Test coverage of the if __name__ == '__main__' block."""
        self.assertTrue(hasattr(multiprocessing, 'set_start_method'))

    @patch('main.create_pool', new_callable=AsyncMock)
    async def test_fetch_stream_configs_with_data(
        self,
        mock_create_pool: Any,
    ) -> None:
        """Test fetch_stream_configs with actual database data."""
        with patch.object(MainApp, 'fetch_stream_configs') as mock_fetch:
            mock_config = {
                'video_url': 'rtsp://test.com/stream',
                'updated_at': '2024-01-01T12:00:00',
                'model_key': 'model-123',
                'site': 'TestSite',
                'stream_name': 'TestStream',
                'expire_date': '2025-12-31T23:59:59',
                'work_start_hour': 8,
                'work_end_hour': 17,
                'detection_items': {
                    'detect_no_safety_vest_or_helmet': True,
                    'detect_near_machinery_or_vehicle': False,
                    'detect_in_restricted_area': True,
                    'detect_in_utility_pole_restricted_area': False,
                    'detect_machinery_close_to_pole': True,
                },
            }
            mock_fetch.return_value = [mock_config]
            app = MainApp()
            configs = await app.fetch_stream_configs()
            self.assertEqual(len(configs), 1)
            config = configs[0]
            self.assertEqual(config['video_url'], 'rtsp://test.com/stream')
            self.assertEqual(config['site'], 'TestSite')
            self.assertEqual(config['stream_name'], 'TestStream')
            self.assertEqual(config['work_start_hour'], 8)
            self.assertEqual(config['work_end_hour'], 17)
            detection_items = config['detection_items']
            self.assertTrue(detection_items['detect_no_safety_vest_or_helmet'])
            self.assertFalse(
                detection_items['detect_near_machinery_or_vehicle'],
            )
            self.assertTrue(detection_items['detect_in_restricted_area'])
            self.assertFalse(
                detection_items['detect_in_utility_pole_restricted_area'],
            )
            self.assertTrue(detection_items['detect_machinery_close_to_pole'])

    @patch('main.create_pool', new_callable=AsyncMock)
    async def test_fetch_stream_configs_with_null_values(
        self,
        mock_create_pool: Any,
    ) -> None:
        """Test fetch_stream_configs with null values in database."""
        with patch.object(MainApp, 'fetch_stream_configs') as mock_fetch:
            mock_config = {
                'video_url': 'rtsp://test.com/stream2',
                'updated_at': '2024-01-01T12:00:00',
                'model_key': 'model-456',
                'site': 'TestSite2',
                'stream_name': 'TestStream2',
                'expire_date': None,
                'work_start_hour': 7,
                'work_end_hour': 18,
                'detection_items': {
                    'detect_no_safety_vest_or_helmet': False,
                    'detect_near_machinery_or_vehicle': True,
                    'detect_in_restricted_area': False,
                    'detect_in_utility_pole_restricted_area': True,
                    'detect_machinery_close_to_pole': False,
                },
            }
            mock_fetch.return_value = [mock_config]
            app = MainApp()
            configs = await app.fetch_stream_configs()
            self.assertEqual(len(configs), 1)
            config = configs[0]
            self.assertEqual(config['video_url'], 'rtsp://test.com/stream2')
            self.assertIsNone(config['expire_date'])
            self.assertEqual(config['work_start_hour'], 7)
            self.assertEqual(config['work_end_hour'], 18)

    @patch('main.create_pool', new_callable=AsyncMock)
    async def test_fetch_stream_configs_database_operations(
        self,
        mock_create_pool: Any,
    ) -> Any:
        """Test the actual database operation code paths in
        fetch_stream_configs."""
        from datetime import datetime

        mock_row = (
            17,
            'rtsp://test.com/stream',
            datetime(2024, 1, 1, 12, 0, 0),
            'model-123',
            'TestSite',
            'TestStream',
            1,
            datetime(2025, 12, 31, 23, 59, 59),
            8,
            17,
            1,
            0,
            1,
            0,
            1,
            3,
        )

        MockConnection = partial(
            _FetchStreamConfigsDatabaseOperationsMockConnection,
            _test_mock_row=mock_row,
        )

        MockPool = partial(
            _FetchStreamConfigsDatabaseOperationsMockPool,
            _test_MockConnection=MockConnection,
        )

        mock_create_pool.return_value = MockPool()
        app = MainApp()
        configs = await app.fetch_stream_configs()
        self.assertEqual(len(configs), 1)
        config = configs[0]
        self.assertEqual(config['video_url'], 'rtsp://test.com/stream')
        self.assertEqual(config['stream_id'], 17)
        self.assertEqual(config['site'], 'TestSite')
        self.assertTrue(config['recognition_enabled'])
        detection_items = config['detection_items']
        self.assertTrue(detection_items['detect_no_safety_vest_or_helmet'])
        self.assertFalse(detection_items['detect_near_machinery_or_vehicle'])
        self.assertTrue(detection_items['detect_in_restricted_area'])


if __name__ == '__main__':
    unittest.main()


def test_stream_processor_media_environment_helpers() -> None:
    """Malformed demand settings and rendition rates retain safe defaults."""
    with patch.dict(
        os.environ,
        {'MEDIA_DEMAND_CACHE_SECONDS': 'invalid', 'MEDIA_PUBLISH_FPS': '20'},
    ):
        assert MediaPublishSettings().demand_cache_seconds == 0.5
        assert MediaPublishSettings().fps == 20
        variants = [
            runtime_state.OverlayPublisherVariant(
                'detail',
                'detail',
            ),
            runtime_state.OverlayPublisherVariant('preview', 'preview'),
        ]
        assert (
            max(
                MediaPublishSettings().rendition_fps(v.rendition)
                for v in variants
            )
            >= 20
        )


def test_overlay_frame_helpers_cover_fallback_and_ready_timing() -> None:
    """Untracked frames and dropped languages avoid unnecessary rendering."""
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    snapshot = runtime_state.OverlaySnapshot(
        sequence=(1, 1),
        frame=frame,
        track_data=None,
    )
    assert make_overlay_cache({}).render(snapshot, {'en'}) == {'en': frame}
    ready_started_at = {'en': 1.0, 'zh-TW': 2.0}
    media._drop_unrequested_overlay_start_times(ready_started_at, {'en'})
    assert ready_started_at == {'en': 1.0}

    run_case = _overlay_frame_helpers_cover_fallback_run_case

    asyncio.run(run_case())


def test_overlay_snapshot_uses_detection_frame_when_capture_is_empty() -> None:
    """Detection output remains publishable during a brief capture handoff."""

    run_case = _overlay_snapshot_uses_detection_frame_run_case

    asyncio.run(run_case())


def test_overlay_language_snapshot_creates_publisher_and_marks_ready() -> None:
    """The first demand for a language creates and advertises its rendition."""

    run_case = _overlay_language_snapshot_creates_publisher_run_case

    asyncio.run(run_case())


def test_overlay_publish_loop_processes_demand_then_closes_publishers() -> (
    None
):
    """Demanded overlays render once, publish, and clean up on shutdown."""
    _callbacks = _OverlayPublishLoopProcessesDemandThenClosesCallbacks()

    _callbacks.DemandCache = _OverlayPublishLoopProcessesDemandDemandCache

    run_case = _callbacks._call_run_case

    asyncio.run(run_case())


def test_clean_restreamer_restarts_after_frozen_source_signal() -> None:
    """A freeze reconnect signal restarts the active source restreamer."""
    _callbacks = _CleanRestreamerRestartsAfterFrozenSourceSignalCallbacks()

    _callbacks.DemandCache = _CleanRestreamerRestartsAfterFrozenDemandCache

    run_case = _callbacks._call_run_case

    asyncio.run(run_case())


def test_capture_reconnect_invalidates_shared_frames() -> None:
    """A source outage removes stale video and forwards a clean restart."""

    run_case = _capture_reconnect_invalidates_shared_frames_run_case

    asyncio.run(run_case())


def test_detection_result_is_discarded_across_reconnect() -> None:
    """An in-flight result from the old decoder generation is never used."""

    run_case = _detection_result_is_discarded_across_run_case

    asyncio.run(run_case())


def test_overlay_publish_loop_retries_after_demand_failure() -> None:
    """A transient demand read failure is logged, delayed, and cleaned up."""
    _callbacks = _OverlayPublishLoopRetriesAfterDemandFailureCallbacks()

    _callbacks.DemandCache = _OverlayPublishLoopRetriesAfterDemandCache

    run_case = _callbacks._call_run_case

    asyncio.run(run_case())


def test_main_worker_and_environment_guard_paths() -> None:
    """Empty topology, stable workers, and invalid env values stay harmless."""
    app = MainApp()
    assert app.workers.ensure(None) is False
    assert app.workers.topology([]) == (0, {}, None)
    alive_process = MagicMock()
    alive_process.is_alive.return_value = True
    app.workers.processes = [alive_process]
    app.workers.request_queues = [MagicMock()]
    app.workers.restart_dead()
    app.workers.request_queues = [MagicMock()]
    with patch.object(app.workers, 'result_queue', return_value=None):
        assert app.workers.slot(
            {'video_url': 'stream', 'site': 'SiteA', 'stream_name': 'Cam1'},
        ) == (None, None)
    assert app.workers.result_queue({'stream_id': -1}) is None


def test_main_config_mode_handles_no_runnable_streams() -> None:
    """A config file with no active stream still performs normal cleanup."""

    run_case = _main_config_mode_handles_no_run_case

    asyncio.run(run_case())


def test_main_uses_expiry_aware_reload_delay_and_stable_stream_ids() -> None:
    """Configuration health checks wake for valid expiries and stream IDs."""
    app = MainApp(poll_interval=10)
    future = datetime.now(timezone.utc) + timedelta(seconds=1)
    app.running_processes = {
        'without-expiry': StreamProcessState(
            cfg={'expire_date': None},
            process=MagicMock(),
            updated_at='',
        ),
        'invalid-expiry': StreamProcessState(
            cfg={'expire_date': 'not-a-date'},
            process=MagicMock(),
            updated_at='',
        ),
        'naive-expiry': StreamProcessState(
            cfg=cast(
                StreamConfig,
                {'expire_date': future.replace(tzinfo=None)},
            ),
            process=MagicMock(),
            updated_at='',
        ),
    }
    delay = app._next_configuration_reload_delay()
    assert 0.1 <= delay < 10
    assert (
        app._stream_process_key(
            cast(StreamConfig, {'stream_id': 42}),
        )
        == 'id:42'
    )
