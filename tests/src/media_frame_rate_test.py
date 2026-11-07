from __future__ import annotations

import asyncio
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

from src import media_restreamer
from src import stream_detection
from src.media_encoding import EncoderOptions
from src.media_publish_config import MediaPublisherFactory
from src.settings.media import MediaPublishSettings
from src.stream_capture import StreamCapture
from src.stream_media import StreamPublisherGroup
from src.stream_runtime_state import LatestFrameState
from tests.stream_fixtures_test import make_stream_runtime

pytestmark = pytest.mark.ml


async def _capture_reads_rate_again_after_run():
    capture = StreamCapture('rtsp://camera')
    try:
        await capture.initialise_stream(capture.stream_url)
        assert capture.source_fps == 30000 / 1001
        await capture.release_resources()
        await capture.initialise_stream(capture.stream_url)
        assert capture.source_fps == 25
    finally:
        await capture.release_resources()


async def _missing_rate_after_reconnect_retains_run(value, /):
    capture = StreamCapture('rtsp://camera')
    capture.source_fps = 25
    capture._create_capture = MagicMock(return_value=MagicMock())
    capture._read_source_fps = MagicMock(return_value=None)
    try:
        await capture.initialise_stream(capture.stream_url)
        assert capture.source_fps == 25
        capture.cap.get.return_value = value
        assert StreamCapture._read_source_fps(capture) is None
    finally:
        await capture.release_resources()


async def _source_frames_are_not_sampled_generate(frames, /):
    for index, frame in enumerate(frames):
        yield frame, index / 60


async def _source_frames_are_not_sampled_run(
    frames,
    generate,
    /,
):
    capture = MagicMock()
    capture.execute_capture = generate
    latest = LatestFrameState()
    await stream_detection._capture_latest_frames(
        capture,
        latest,
        asyncio.Event(),
    )
    capture.update_capture_interval.assert_called_once_with(0)
    assert latest.sequence == 60
    assert latest.frame is frames[-1]


async def _clean_and_overlay_publishers_follow_run():
    source = SimpleNamespace(source_fps=30)
    group = StreamPublisherGroup(
        make_stream_runtime(
            streaming_capture=source,
        ),
    )
    group._mark_overlay_ready = AsyncMock()
    variant = group.variants[0]
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    clean = await group._ensure_clean_publisher(None, 'detail')
    overlay = group.factory.create(
        'rtsp://media/overlay',
        rendition='detail',
    )
    clean.close = AsyncMock()
    overlay.close = AsyncMock()
    overlay.publish = AsyncMock()
    variant.publishers['en'] = overlay
    assert await group._ensure_clean_publisher(clean, 'detail') is clean
    await group.publish_overlay(variant, 'en', frame)
    assert variant.publishers['en'] is overlay

    source.source_fps = 25
    updated = await group._ensure_clean_publisher(clean, 'detail')
    assert updated.fps == 25
    clean.close.assert_awaited_once()
    replacement = MagicMock(fps=25, publish=AsyncMock())
    group.factory.create = MagicMock(return_value=replacement)
    await group.publish_overlay(variant, 'en', frame)
    overlay.close.assert_awaited_once()
    assert variant.publishers['en'] is replacement
    assert group.factory.create.call_args.kwargs['rendition'] == 'detail'


@pytest.fixture(autouse=True)
def automatic_rates(monkeypatch):
    for key in (
        'MEDIA_PUBLISH_FPS',
        'MEDIA_PUBLISH_CLEAN_FPS',
        'MEDIA_PUBLISH_SOURCE_FPS',
        'MEDIA_PREVIEW_FPS',
    ):
        monkeypatch.setenv(key, 'auto')


@pytest.mark.parametrize('fps', [12, 25, 30000 / 1001, 60, 0.5])
def test_all_raw_frame_outputs_use_camera_rate(fps):
    factory = MediaPublisherFactory(
        MediaPublishSettings(),
        SimpleNamespace(source_fps=fps),
    )
    for rendition in ('detail', 'preview'):
        publisher = factory.create(
            'rtsp://media/test',
            rendition=rendition,
        )
        assert publisher.fps == fps
        command = publisher._build_ffmpeg_command(
            '/bin/ffmpeg',
            640,
            360,
            encoder='libx264',
        )
        assert '-use_wallclock_as_timestamps' not in command
        input_fps = float(command[command.index('-framerate') + 1])
        assert input_fps == pytest.approx(fps, rel=1e-5)
        assert float(command[command.index('-r') + 1]) == pytest.approx(
            fps,
            rel=1e-5,
        )


def test_numeric_overrides_are_explicit_and_captured(monkeypatch):
    monkeypatch.setenv('MEDIA_PUBLISH_FPS', '20')
    settings = MediaPublishSettings()
    factory = MediaPublisherFactory(settings, SimpleNamespace(source_fps=30))
    monkeypatch.setenv('MEDIA_PUBLISH_FPS', '7')
    assert factory.fps('detail') == 20
    assert factory.fps('preview') == 20
    settings.preview_fps = 10
    assert factory.fps('preview') == 10
    assert factory.fps('detail') == 20


@pytest.mark.parametrize('value', ['', 'auto', None])
def test_automatic_rate_values(value):
    settings = MediaPublishSettings(
        fps=value,
        clean_fps=value,
        source_fps=value,
        preview_fps=value,
    )
    assert settings.rendition_fps('detail', 25) == 25
    assert settings.rendition_fps('preview', 25) == 25
    assert settings.source_fps is None
    assert settings.clean_fps is None


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
def test_invalid_rates_are_rejected(value):
    for key in ('fps', 'clean_fps', 'source_fps', 'preview_fps'):
        with pytest.raises(ValidationError):
            MediaPublishSettings(**{key: value})


@pytest.mark.parametrize('fps', [0, -1, float('nan'), float('inf'), None])
def test_unknown_camera_rate_has_safe_startup_fallback(fps):
    assert MediaPublishSettings().rendition_fps('detail', fps) == 30


def test_capture_reads_rate_again_after_reconnect(monkeypatch):
    first, second = MagicMock(), MagicMock()
    first.get.return_value = 30000 / 1001
    second.get.return_value = 25
    monkeypatch.setattr(
        StreamCapture,
        '_create_capture',
        MagicMock(side_effect=[first, second]),
    )

    run = _capture_reads_rate_again_after_run

    asyncio.run(run())


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
def test_missing_rate_after_reconnect_retains_last_known_rate(value):
    run = partial(_missing_rate_after_reconnect_retains_run, value)

    asyncio.run(run())


def test_source_frames_are_not_sampled_by_output_rate(monkeypatch):
    monkeypatch.setenv('MEDIA_PUBLISH_FPS', '15')
    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(60)]

    generate = partial(_source_frames_are_not_sampled_generate, frames)

    run = partial(_source_frames_are_not_sampled_run, frames, generate)

    asyncio.run(run())


def test_clean_and_overlay_publishers_follow_reconnected_camera():
    run = _clean_and_overlay_publishers_follow_run

    asyncio.run(run())


def test_direct_restream_preserves_timing_until_explicit_override(monkeypatch):
    monkeypatch.setenv('MEDIA_PUBLISH_KEYFRAME_INTERVAL_SECONDS', '2')
    for encoder in ('libx264', 'h264_nvenc'):
        command = media_restreamer.SourceCommandBuilder(
            EncoderOptions(),
        ).build_command(
            '/bin/ffmpeg',
            'rtsp://camera',
            'rtsp://media/output',
            encoder,
        )
        assert '-r' not in command
        assert '-use_wallclock_as_timestamps' not in command
        assert command[command.index('-fps_mode') + 1] == 'passthrough'
        assert '-force_key_frames' in command
    monkeypatch.setenv('MEDIA_PUBLISH_CLEAN_FPS', '25')
    command = media_restreamer.SourceCommandBuilder(
        EncoderOptions(),
    ).build_command(
        '/bin/ffmpeg',
        'rtsp://camera',
        'rtsp://media/output',
        'libx264',
    )
    assert command[command.index('-r') + 1] == '25'
    assert command[command.index('-g') + 1] == '50'
