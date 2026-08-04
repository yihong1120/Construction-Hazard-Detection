from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src import media_restreamer as restreamer


class _FakeProcess:
    """Tests for _FakeProcess."""

    def __init__(self, returncode: int | None = None) -> None:
        """Support __init__."""
        self.returncode = returncode
        self.terminated = False
        self.killed = False
        self.wait: Any = AsyncMock(return_value=returncode)

    def terminate(self) -> None:
        """Support terminate."""
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        """Support kill."""
        self.killed = True
        self.returncode = -9


class _SlowProcess(_FakeProcess):
    """Tests for _SlowProcess."""

    def __init__(self) -> None:
        """Support __init__."""
        super().__init__()
        self.wait_count = 0
        self.wait = self._wait

    def terminate(self) -> None:
        """Support terminate."""
        self.terminated = True

    async def _wait(self) -> int | None:
        """Support _wait."""
        self.wait_count += 1
        if self.wait_count == 1:
            await asyncio.sleep(10)
        return self.returncode


class _OneShotProcess(_FakeProcess):
    """Tests for _OneShotProcess."""

    def __init__(self) -> None:
        """Support __init__."""
        super().__init__()
        self.wait = self._wait

    async def _wait(self) -> int | None:
        """Support _wait."""
        self.returncode = 0
        return self.returncode


def test_find_ffmpeg_prefers_environment(monkeypatch: Any) -> None:
    """Exercise this test."""
    monkeypatch.setenv('MEDIA_FFMPEG_PATH', '/custom/ffmpeg')

    assert restreamer._find_ffmpeg() == '/custom/ffmpeg'


def test_find_ffmpeg_raises_when_missing(monkeypatch: Any) -> None:
    """Exercise this test."""
    monkeypatch.delenv('MEDIA_FFMPEG_PATH', raising=False)
    monkeypatch.setattr(restreamer.shutil, 'which', lambda _name: None)

    with pytest.raises(RuntimeError, match='ffmpeg'):
        restreamer._find_ffmpeg()


def test_find_ffmpeg_uses_path_lookup(monkeypatch: Any) -> None:
    """Exercise this test."""
    monkeypatch.delenv('MEDIA_FFMPEG_PATH', raising=False)
    monkeypatch.setattr(
        restreamer.shutil,
        'which',
        lambda _name: '/bin/ffmpeg',
    )

    assert restreamer._find_ffmpeg() == '/bin/ffmpeg'


def test_build_command_uses_rtsp_tcp_for_rtsp_source(monkeypatch: Any) -> None:
    """Exercise this test."""
    monkeypatch.setenv('MEDIA_PUBLISH_CLEAN_FPS', '12')

    command = restreamer._build_command(
        '/bin/ffmpeg',
        'rtsp://camera/stream',
        'rtsp://127.0.0.1:8554/out',
        'copy',
    )

    assert command[:2] == ['/bin/ffmpeg', '-hide_banner']
    assert command.count('-rtsp_transport') == 2
    assert 'copy' in command
    assert command[-1] == 'rtsp://127.0.0.1:8554/out'


def test_build_command_adds_timing_for_encoded_stream(
        monkeypatch: Any,
) -> None:
    """Exercise this test."""
    monkeypatch.setenv('MEDIA_PUBLISH_CLEAN_FPS', '12')

    command = restreamer._build_command(
        '/bin/ffmpeg',
        'https://example.test/stream.m3u8',
        'rtsp://127.0.0.1:8554/out',
        'libx264',
    )

    assert '-c:v' in command
    assert 'libx264' in command
    assert '-g' in command
    assert '24' in command


def test_build_command_uses_nvenc_options() -> None:
    """Exercise this test."""
    command = restreamer._build_command(
        '/bin/ffmpeg',
        'https://example.test/stream.m3u8',
        'rtsp://127.0.0.1:8554/out',
        'h264_nvenc',
    )

    assert 'h264_nvenc' in command
    assert '-maxrate' in command


def test_get_encoder_prefers_clean_encoder(monkeypatch: Any) -> None:
    """Exercise this test."""
    monkeypatch.setenv('MEDIA_RESTREAM_ENCODER', 'copy')
    monkeypatch.setenv('MEDIA_PUBLISH_CLEAN_ENCODER', 'libx264')

    assert restreamer._get_encoder() == 'libx264'


def test_get_encoder_uses_restream_encoder(monkeypatch: Any) -> None:
    """Exercise this test."""
    monkeypatch.delenv('MEDIA_PUBLISH_CLEAN_ENCODER', raising=False)
    monkeypatch.setenv('MEDIA_RESTREAM_ENCODER', 'copy')

    assert restreamer._get_encoder() == 'copy'


def test_stop_process_terminates_live_process() -> None:
    """Exercise this test."""
    stream = restreamer.MediaSourceRestreamer(
        'rtsp://camera/stream',
        'rtsp://127.0.0.1:8554/out',
    )
    process = _FakeProcess()
    stream._process = process

    asyncio.run(stream._stop_process())

    assert stream._process is None
    assert process.terminated
    process.wait.assert_awaited_once()


def test_stop_process_kills_after_timeout() -> None:
    """Exercise this test."""
    stream = restreamer.MediaSourceRestreamer(
        'rtsp://camera/stream',
        'rtsp://127.0.0.1:8554/out',
    )
    process = _SlowProcess()
    stream._process = process

    asyncio.run(stream._stop_process())

    assert process.terminated
    assert process.killed


def test_restart_terminates_live_process_without_closing_monitor() -> None:
    """A frozen-frame signal can reconnect the source restreamer."""
    stream = restreamer.MediaSourceRestreamer(
        'rtsp://camera/stream',
        'rtsp://127.0.0.1:8554/out',
    )
    process = _FakeProcess()
    stream._process = process

    asyncio.run(stream.restart())

    assert stream._closed is False
    assert stream._process is None
    assert process.terminated


def test_start_creates_monitor_task() -> None:
    """Exercise this test."""
    async def run_case() -> restreamer.MediaSourceRestreamer:
        """Support run_case."""
        stream = restreamer.MediaSourceRestreamer(
            'rtsp://camera/stream',
            'rtsp://127.0.0.1:8554/out',
        )
        await stream.start()
        first_task = stream._monitor_task
        await stream.start()
        assert stream._monitor_task is first_task
        await stream.close()
        return stream

    stream = asyncio.run(run_case())

    assert stream._closed is True


def test_monitor_loop_runs_one_process(monkeypatch: Any) -> None:
    """Exercise this test."""
    async def fake_create_subprocess_exec(*args, **_kwargs) -> Any:
        """Support fake_create_subprocess_exec."""
        calls.append(args)
        stream._closed = True
        return _OneShotProcess()

    calls: list[tuple[object, ...]] = []
    stream = restreamer.MediaSourceRestreamer(
        'rtsp://camera/stream',
        'rtsp://127.0.0.1:8554/out',
    )
    monkeypatch.setattr(restreamer, '_find_ffmpeg', lambda: '/bin/ffmpeg')
    monkeypatch.setattr(restreamer, '_get_encoder', lambda: 'copy')
    monkeypatch.setattr(
        restreamer.asyncio,
        'create_subprocess_exec',
        fake_create_subprocess_exec,
    )

    asyncio.run(stream._monitor_loop())

    assert calls
    assert calls[0][0] == '/bin/ffmpeg'


def test_monitor_loop_sleeps_before_restart(monkeypatch: Any) -> None:
    """Exercise this test."""
    async def fake_create_subprocess_exec(*_args, **_kwargs) -> Any:
        """Support fake_create_subprocess_exec."""
        return _OneShotProcess()

    async def fake_sleep(_delay: float) -> None:
        """Support fake_sleep."""
        sleeps.append(True)
        stream._closed = True

    sleeps: list[bool] = []
    stream = restreamer.MediaSourceRestreamer(
        'rtsp://camera/stream',
        'rtsp://127.0.0.1:8554/out',
    )
    monkeypatch.setattr(restreamer, '_find_ffmpeg', lambda: '/bin/ffmpeg')
    monkeypatch.setattr(restreamer, '_get_encoder', lambda: 'copy')
    monkeypatch.setattr(
        restreamer.asyncio,
        'create_subprocess_exec',
        fake_create_subprocess_exec,
    )
    monkeypatch.setattr(restreamer.asyncio, 'sleep', fake_sleep)

    asyncio.run(stream._monitor_loop())

    assert sleeps == [True]


def test_close_cancels_monitor_task() -> None:
    """Exercise this test."""
    async def run_case() -> restreamer.MediaSourceRestreamer:
        """Support run_case."""
        stream = restreamer.MediaSourceRestreamer(
            'rtsp://camera/stream',
            'rtsp://127.0.0.1:8554/out',
        )
        stream._monitor_task = asyncio.create_task(asyncio.sleep(60))
        await stream.close()
        return stream

    stream = asyncio.run(run_case())

    assert stream._closed is True
    assert stream._monitor_task is None
