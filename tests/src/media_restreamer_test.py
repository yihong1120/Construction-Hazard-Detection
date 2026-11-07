from __future__ import annotations

import asyncio
from functools import partial
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src import media_restreamer as restreamer
from src.media_encoding import EncoderOptions


def _return_none(_name):
    return None


def _find_ffmpeg_uses_path_lookup_callback(_name):
    return '/bin/ffmpeg'


async def _start_creates_monitor_task_run_case() -> (
    restreamer.MediaSourceRestreamer
):
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


def _monitor_loop_runs_one_process_callback(*_args):
    return '/bin/ffmpeg'


def _monitor_loop_runs_one_process_callback2(*_args):
    return 'copy'


async def _monitor_loop_sleeps_before_restart_fake_create_subprocess_exec(
    *_args: Any,
    **_kwargs: Any,
) -> Any:
    """Support fake_create_subprocess_exec."""
    return _OneShotProcess()


def _monitor_loop_releases_reserved_nvenc_callback(*_args):
    return 'nvenc'


def _return_true(*_args):
    return True


def _monitor_loop_releases_reserved_nvenc_append(releases, /, *_args):
    return releases.append(True)


def _monitor_loop_falls_back_when_callback(*_args):
    return 'h264_nvenc'


def _return_false(*_args):
    return False


async def _close_cancels_monitor_task_run_case() -> (
    restreamer.MediaSourceRestreamer
):
    """Support run_case."""
    stream = restreamer.MediaSourceRestreamer(
        'rtsp://camera/stream',
        'rtsp://127.0.0.1:8554/out',
    )
    stream._monitor_task = asyncio.create_task(asyncio.sleep(60))
    await stream.close()
    return stream


class _MonitorLoopRunsOneProcessCallbacks:
    calls: list[tuple[object, ...]]
    stream: Any

    async def _call_fake_create_subprocess_exec(
        self,
        *args: Any,
        **_kwargs: Any,
    ) -> Any:
        """Support fake_create_subprocess_exec."""
        self.calls.append(args)
        self.stream._closed = True
        return _OneShotProcess()


class _MonitorLoopSleepsBeforeRestartCallbacks:
    sleeps: list[bool]
    stream: Any

    async def _call_fake_sleep(self, _delay: float) -> None:
        """Support fake_sleep."""
        self.sleeps.append(True)
        self.stream._closed = True


class _MonitorLoopReleasesReservedNvencSessionCallbacks:
    commands: list[tuple[object, ...]]
    stream: Any

    async def _call_fake_create_subprocess_exec(
        self,
        *args: Any,
        **_kwargs: Any,
    ) -> Any:
        self.commands.append(args)
        self.stream._closed = True
        return _OneShotProcess()


class _MonitorLoopFallsBackWhenNvencBudgetCallbacks:
    commands: list[tuple[object, ...]]
    stream: Any

    async def _call_fake_create_subprocess_exec(
        self,
        *args: Any,
        **_kwargs: Any,
    ) -> Any:
        self.commands.append(args)
        self.stream._closed = True
        return _OneShotProcess()


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
    monkeypatch.setenv('MEDIA_FFMPEG_PATH', '/custom/ffmpeg')

    assert (
        restreamer.SourceCommandBuilder(
            EncoderOptions(),
        ).find_ffmpeg()
        == '/custom/ffmpeg'
    )


def test_find_ffmpeg_raises_when_missing(monkeypatch: Any) -> None:
    monkeypatch.delenv('MEDIA_FFMPEG_PATH', raising=False)
    monkeypatch.setattr(restreamer.shutil, 'which', _return_none)

    with pytest.raises(RuntimeError, match='ffmpeg'):
        restreamer.SourceCommandBuilder(EncoderOptions()).find_ffmpeg()


def test_find_ffmpeg_uses_path_lookup(monkeypatch: Any) -> None:
    monkeypatch.delenv('MEDIA_FFMPEG_PATH', raising=False)
    monkeypatch.setattr(
        restreamer.shutil,
        'which',
        _find_ffmpeg_uses_path_lookup_callback,
    )

    assert (
        restreamer.SourceCommandBuilder(
            EncoderOptions(),
        ).find_ffmpeg()
        == '/bin/ffmpeg'
    )


def test_build_command_uses_rtsp_tcp_for_rtsp_source(monkeypatch: Any) -> None:
    monkeypatch.setenv('MEDIA_PUBLISH_CLEAN_FPS', '12')
    monkeypatch.setenv('MEDIA_PUBLISH_KEYFRAME_INTERVAL_SECONDS', '2')

    command = restreamer.SourceCommandBuilder(EncoderOptions()).build_command(
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
    monkeypatch.setenv('MEDIA_PUBLISH_CLEAN_FPS', '12')
    monkeypatch.setenv('MEDIA_PUBLISH_KEYFRAME_INTERVAL_SECONDS', '2')

    command = restreamer.SourceCommandBuilder(EncoderOptions()).build_command(
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
    command = restreamer.SourceCommandBuilder(EncoderOptions()).build_command(
        '/bin/ffmpeg',
        'https://example.test/stream.m3u8',
        'rtsp://127.0.0.1:8554/out',
        'h264_nvenc',
    )

    assert 'h264_nvenc' in command
    assert '-maxrate' in command


def test_get_encoder_prefers_clean_encoder(monkeypatch: Any) -> None:
    monkeypatch.setenv('MEDIA_RESTREAM_ENCODER', 'copy')
    monkeypatch.setenv('MEDIA_PUBLISH_CLEAN_ENCODER', 'libx264')

    assert (
        restreamer.SourceCommandBuilder(
            EncoderOptions(),
        ).get_encoder()
        == 'libx264'
    )


def test_get_encoder_uses_restream_encoder(monkeypatch: Any) -> None:
    monkeypatch.delenv('MEDIA_PUBLISH_CLEAN_ENCODER', raising=False)
    monkeypatch.setenv('MEDIA_RESTREAM_ENCODER', 'copy')

    assert (
        restreamer.SourceCommandBuilder(
            EncoderOptions(),
        ).get_encoder()
        == 'copy'
    )


def test_stop_process_terminates_live_process() -> None:
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


def test_restart_ignores_closed_restreamer() -> None:
    """A closed restreamer must not touch an already stopped process."""
    stream = restreamer.MediaSourceRestreamer(
        'rtsp://camera/stream',
        'rtsp://127.0.0.1:8554/out',
    )
    process = _FakeProcess()
    stream._closed = True
    stream._process = process

    asyncio.run(stream.restart())

    assert stream._process is process
    assert process.terminated is False


def test_start_creates_monitor_task() -> None:

    run_case = _start_creates_monitor_task_run_case

    stream = asyncio.run(run_case())

    assert stream._closed is True


def test_monitor_loop_runs_one_process(monkeypatch: Any) -> None:

    _callbacks = _MonitorLoopRunsOneProcessCallbacks()
    fake_create_subprocess_exec = _callbacks._call_fake_create_subprocess_exec

    _callbacks.calls = []
    _callbacks.stream = restreamer.MediaSourceRestreamer(
        'rtsp://camera/stream',
        'rtsp://127.0.0.1:8554/out',
    )
    monkeypatch.setattr(
        restreamer.SourceCommandBuilder,
        'find_ffmpeg',
        _monitor_loop_runs_one_process_callback,
    )
    monkeypatch.setattr(
        restreamer.SourceCommandBuilder,
        'get_encoder',
        _monitor_loop_runs_one_process_callback2,
    )
    monkeypatch.setattr(
        restreamer.asyncio,
        'create_subprocess_exec',
        fake_create_subprocess_exec,
    )

    asyncio.run(_callbacks.stream._monitor_loop())

    assert _callbacks.calls
    assert _callbacks.calls[0][0] == '/bin/ffmpeg'


def test_monitor_loop_sleeps_before_restart(monkeypatch: Any) -> None:

    _callbacks = _MonitorLoopSleepsBeforeRestartCallbacks()
    fake_create_subprocess_exec = (
        _monitor_loop_sleeps_before_restart_fake_create_subprocess_exec
    )

    fake_sleep = _callbacks._call_fake_sleep

    _callbacks.sleeps = []
    _callbacks.stream = restreamer.MediaSourceRestreamer(
        'rtsp://camera/stream',
        'rtsp://127.0.0.1:8554/out',
    )
    monkeypatch.setattr(
        restreamer.SourceCommandBuilder,
        'find_ffmpeg',
        _monitor_loop_runs_one_process_callback,
    )
    monkeypatch.setattr(
        restreamer.SourceCommandBuilder,
        'get_encoder',
        _monitor_loop_runs_one_process_callback2,
    )
    monkeypatch.setattr(
        restreamer.asyncio,
        'create_subprocess_exec',
        fake_create_subprocess_exec,
    )
    monkeypatch.setattr(restreamer.asyncio, 'sleep', fake_sleep)

    asyncio.run(_callbacks.stream._monitor_loop())

    assert _callbacks.sleeps == [True]


def test_monitor_loop_releases_reserved_nvenc_session(
    monkeypatch: Any,
) -> None:
    """NVENC reservations are released after the ffmpeg process exits."""
    _callbacks = _MonitorLoopReleasesReservedNvencSessionCallbacks()

    fake_create_subprocess_exec = _callbacks._call_fake_create_subprocess_exec

    _callbacks.commands = []
    releases: list[bool] = []
    _callbacks.stream = restreamer.MediaSourceRestreamer(
        'rtsp://camera/stream',
        'rtsp://127.0.0.1:8554/out',
    )
    monkeypatch.setattr(
        restreamer.SourceCommandBuilder,
        'find_ffmpeg',
        _monitor_loop_runs_one_process_callback,
    )
    monkeypatch.setattr(
        restreamer.SourceCommandBuilder,
        'get_encoder',
        _monitor_loop_releases_reserved_nvenc_callback,
    )
    monkeypatch.setattr(
        restreamer,
        'try_acquire_nvenc_session',
        _return_true,
    )
    monkeypatch.setattr(
        restreamer,
        'release_nvenc_session',
        partial(_monitor_loop_releases_reserved_nvenc_append, releases),
    )
    monkeypatch.setattr(
        restreamer.asyncio,
        'create_subprocess_exec',
        fake_create_subprocess_exec,
    )

    asyncio.run(_callbacks.stream._monitor_loop())

    assert 'h264_nvenc' in _callbacks.commands[0]
    assert releases == [True]


def test_monitor_loop_falls_back_when_nvenc_budget_is_full(
    monkeypatch: Any,
) -> None:
    """A full local NVENC budget uses the software encoder for this stream."""
    _callbacks = _MonitorLoopFallsBackWhenNvencBudgetCallbacks()

    fake_create_subprocess_exec = _callbacks._call_fake_create_subprocess_exec

    _callbacks.commands = []
    _callbacks.stream = restreamer.MediaSourceRestreamer(
        'rtsp://camera/stream',
        'rtsp://127.0.0.1:8554/out',
    )
    monkeypatch.setattr(
        restreamer.SourceCommandBuilder,
        'find_ffmpeg',
        _monitor_loop_runs_one_process_callback,
    )
    monkeypatch.setattr(
        restreamer.SourceCommandBuilder,
        'get_encoder',
        _monitor_loop_falls_back_when_callback,
    )
    monkeypatch.setattr(
        restreamer,
        'try_acquire_nvenc_session',
        _return_false,
    )
    monkeypatch.setattr(
        restreamer.asyncio,
        'create_subprocess_exec',
        fake_create_subprocess_exec,
    )

    asyncio.run(_callbacks.stream._monitor_loop())

    assert 'libx264' in _callbacks.commands[0]


def test_close_cancels_monitor_task() -> None:

    run_case = _close_cancels_monitor_task_run_case

    stream = asyncio.run(run_case())

    assert stream._closed is True
    assert stream._monitor_task is None
