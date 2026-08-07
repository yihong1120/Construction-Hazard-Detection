from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest

from src import media_stream_publisher as publisher


class _FakeStdin:
    """Tests for _FakeStdin."""

    def __init__(self) -> None:
        """Support __init__."""
        self.closed = False
        self.writes: list[memoryview] = []
        self.drain: Any = AsyncMock()

    def is_closing(self) -> bool:
        """Support is_closing."""
        return self.closed

    def close(self) -> None:
        """Support close."""
        self.closed = True

    def write(self, payload: memoryview) -> None:
        """Support write."""
        self.writes.append(payload)


class _FakeProcess:
    """Tests for _FakeProcess."""

    def __init__(self, returncode: int | None = None) -> None:
        """Support __init__."""
        self.returncode = returncode
        self.stdin = _FakeStdin()
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


class _FakeTask:
    """Tests for _FakeTask."""

    def __init__(self) -> None:
        """Support __init__."""
        self.cancelled = False

    def cancel(self) -> None:
        """Support cancel."""
        self.cancelled = True

    def done(self) -> bool:
        """Support done."""
        return True


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


class _RaisingStdin(_FakeStdin):
    """Tests for _RaisingStdin."""

    def close(self) -> None:
        """Support close."""
        raise BrokenPipeError


class _Transport:
    """Tests for _Transport."""

    def __init__(self) -> None:
        """Support __init__."""
        self.calls = 0
        self.limits: list[tuple[int, int]] = []

    def set_write_buffer_limits(self, high: int, low: int) -> None:
        """Support set_write_buffer_limits."""
        self.limits.append((high, low))

    def get_write_buffer_size(self) -> int:
        """Support get_write_buffer_size."""
        self.calls += 1
        return 9999 if self.calls == 1 else 0


class _LoopStdin(_FakeStdin):
    """Tests for _LoopStdin."""

    def __init__(self, process: _FakeProcess) -> None:
        """Support __init__."""
        super().__init__()
        self.process = process
        self.transport = _Transport()
        self.drain = self._drain

    async def _drain(self) -> None:
        """Support _drain."""
        self.process.returncode = 0


class _ErrorStdin(_FakeStdin):
    """Tests for _ErrorStdin."""

    def __init__(self) -> None:
        """Support __init__."""
        super().__init__()
        self.drain = self._drain

    async def _drain(self) -> None:
        """Support _drain."""
        raise RuntimeError('closed')


class _FakeStderr:
    """Minimal asynchronous stderr reader for publisher tests."""

    def __init__(self, lines: list[object]) -> None:
        self.lines = iter(lines)

    async def readline(self) -> object:
        item = next(self.lines)
        if isinstance(item, Exception):
            raise item
        return item


def test_prepare_frame_resizes_and_crops_to_even_dimensions() -> None:
    """Exercise this test."""
    stream = publisher.MediaStreamPublisher(
        'rtsp://127.0.0.1:8554/out',
        width=9,
        height=7,
    )

    frame = np.zeros((11, 13, 3), dtype=np.uint8)
    prepared = stream._prepare_frame(frame)

    assert prepared.shape == (6, 8, 3)
    assert prepared.flags['C_CONTIGUOUS']


def test_resize_to_stream_size_keeps_matching_contiguous_frame() -> None:
    """Exercise this test."""
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    stream._stream_size = (4, 4)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    resized = stream._resize_to_stream_size(frame)

    assert resized.shape == (4, 4, 3)
    assert resized.flags['C_CONTIGUOUS']


def test_resize_to_stream_size_resizes_mismatched_frame() -> None:
    """Exercise this test."""
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    stream._stream_size = (6, 4)
    frame = np.zeros((8, 10, 3), dtype=np.uint8)

    resized = stream._resize_to_stream_size(frame)

    assert resized.shape == (4, 6, 3)
    assert resized.flags['C_CONTIGUOUS']


def test_resize_to_stream_size_returns_frame_when_size_missing() -> None:
    """Exercise this test."""
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    assert stream._resize_to_stream_size(frame) is frame


def test_is_process_alive_reflects_process_state() -> None:
    """Exercise this test."""
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    assert stream._is_process_alive() is False

    stream._process = _FakeProcess()
    assert stream._is_process_alive() is True

    stream._process.returncode = 0
    assert stream._is_process_alive() is False


def test_publish_starts_writer_for_first_frame(monkeypatch: Any) -> None:
    """Exercise this test."""
    async def fake_start(width: int, height: int) -> None:
        """Support fake_start."""
        starts.append((width, height))
        stream._process = _FakeProcess()
        stream._started = True
        stream._stream_size = (width, height)

    def fake_create_task(coro: Any) -> Any:
        """Support fake_create_task.

        Args:
            coro: Test helper value.
        """
        coro.close()
        return _FakeTask()

    starts: list[tuple[int, int]] = []
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    monkeypatch.setattr(stream, '_start', fake_start)
    monkeypatch.setattr(publisher.asyncio, 'create_task', fake_create_task)

    asyncio.run(stream.publish(np.zeros((4, 6, 3), dtype=np.uint8)))

    assert starts == [(6, 4)]
    assert stream._latest_frame is not None
    assert stream._writer_task is not None


def test_publish_resets_dead_process(monkeypatch: Any) -> None:
    """Exercise this test."""
    async def fake_reset() -> None:
        """Support fake_reset."""
        resets.append(True)
        stream._started = False

    async def fake_start(width: int, height: int) -> None:
        """Support fake_start."""
        starts.append((width, height))
        stream._started = True
        stream._stream_size = (width, height)

    def fake_create_task(coro: Any) -> Any:
        """Support fake_create_task.

        Args:
            coro: Test helper value.
        """
        coro.close()
        return _FakeTask()

    resets: list[bool] = []
    starts: list[tuple[int, int]] = []
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    stream._started = True
    stream._process = _FakeProcess(returncode=1)
    monkeypatch.setattr(stream, '_reset_after_process_exit', fake_reset)
    monkeypatch.setattr(stream, '_start', fake_start)
    monkeypatch.setattr(publisher.asyncio, 'create_task', fake_create_task)

    asyncio.run(stream.publish(np.zeros((4, 6, 3), dtype=np.uint8)))

    assert resets == [True]
    assert starts == [(6, 4)]


def test_publish_resizes_late_frame() -> None:
    """Exercise this test."""
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    stream._started = True
    stream._process = _FakeProcess()
    stream._stream_size = (6, 4)

    asyncio.run(stream.publish(np.zeros((8, 10, 3), dtype=np.uint8)))

    assert stream._latest_frame is not None
    assert stream._latest_frame.shape == (4, 6, 3)


def test_select_encoder_uses_x264_when_nvenc_missing(monkeypatch: Any) -> None:
    """Exercise this test."""
    monkeypatch.setenv('MEDIA_PUBLISH_ENCODER', 'nvenc')
    monkeypatch.setattr(publisher, '_ffmpeg_has_encoder', lambda *_: False)

    assert publisher._select_encoder('/usr/bin/ffmpeg') == 'libx264'


def test_select_encoder_uses_auto_nvenc_when_available(
        monkeypatch: Any,
) -> None:
    """Exercise this test."""
    monkeypatch.setenv('MEDIA_PUBLISH_ENCODER', 'auto')
    monkeypatch.setattr(publisher, '_ffmpeg_has_encoder', lambda *_: True)

    assert publisher._select_encoder('/usr/bin/ffmpeg') == 'h264_nvenc'


def test_build_ffmpeg_command_contains_rawvideo_and_rtsp(
        monkeypatch: Any,
) -> None:
    """Exercise this test."""
    monkeypatch.setenv('MEDIA_PUBLISH_ENCODER', 'libx264')
    stream = publisher.MediaStreamPublisher(
        'rtsp://127.0.0.1:8554/out',
        fps=12,
    )

    command = stream._build_ffmpeg_command('/bin/ffmpeg', 640, 480)

    assert command[:2] == ['/bin/ffmpeg', '-hide_banner']
    assert 'rawvideo' in command
    assert '640x480' in command
    assert 'libx264' in command
    assert command[-1] == 'rtsp://127.0.0.1:8554/out'


def test_build_ffmpeg_command_uses_nvenc_options(monkeypatch: Any) -> None:
    """Exercise this test."""
    monkeypatch.setenv('MEDIA_PUBLISH_ENCODER', 'h264_nvenc')
    monkeypatch.setenv('MEDIA_PUBLISH_KEYFRAME_INTERVAL_SECONDS', '2')
    monkeypatch.setattr(publisher, '_ffmpeg_has_encoder', lambda *_: True)
    stream = publisher.MediaStreamPublisher(
        'rtsp://127.0.0.1:8554/out',
        fps=8,
    )

    command = stream._build_ffmpeg_command('/bin/ffmpeg', 320, 240)

    assert 'h264_nvenc' in command
    assert '16' in command


def test_build_ffmpeg_command_honours_preview_rate_budget(
        monkeypatch: Any,
) -> None:
    """A preview publisher gets an independent capped bitrate."""
    monkeypatch.setenv('MEDIA_PUBLISH_ENCODER', 'libx264')
    monkeypatch.setenv('MEDIA_PUBLISH_KEYFRAME_INTERVAL_SECONDS', '2')
    stream = publisher.MediaStreamPublisher(
        'rtsp://127.0.0.1:8554/preview',
        fps=15,
        width=640,
        height=360,
        bitrate='500k',
        maxrate='700k',
        bufsize='1400k',
    )

    command = stream._build_ffmpeg_command('/bin/ffmpeg', 640, 360)

    assert '500k' in command
    assert '700k' in command
    assert '1400k' in command
    assert '30' in command


def test_ffmpeg_has_encoder_handles_subprocess_failure(
        monkeypatch: Any,
) -> None:
    """Exercise this test."""
    def fail(*_args: object, **_kwargs: object) -> object:
        """Support fail."""
        raise OSError('missing')

    monkeypatch.setattr(publisher.subprocess, 'run', fail)

    assert (
        publisher._ffmpeg_has_encoder('/missing/ffmpeg', 'h264_nvenc')
        is False
    )


def test_ffmpeg_has_encoder_reads_stdout(monkeypatch: Any) -> None:
    """Exercise this test."""
    result = SimpleNamespace(stdout=' V..... h264_nvenc NVIDIA NVENC H.264')
    monkeypatch.setattr(publisher.subprocess, 'run', lambda *_a, **_k: result)

    assert publisher._ffmpeg_has_encoder('/bin/ffmpeg', 'h264_nvenc') is True


def test_stop_process_terminates_live_process() -> None:
    """Exercise this test."""
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    process = _FakeProcess()
    stream._process = process

    asyncio.run(stream._stop_process())

    assert stream._process is None
    assert process.stdin.closed
    assert process.terminated
    process.wait.assert_awaited_once()


def test_stop_process_kills_process_after_timeout() -> None:
    """Exercise this test."""
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    process = _SlowProcess()
    stream._process = process

    asyncio.run(stream._stop_process())

    assert process.terminated
    assert process.killed


def test_stop_process_ignores_stdin_close_pipe_errors() -> None:
    """Exercise this test."""
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    process = _FakeProcess(returncode=0)
    process.stdin = _RaisingStdin()
    stream._process = process

    asyncio.run(stream._stop_process())

    assert stream._process is None


def test_stop_process_cancels_stderr_tasks() -> None:
    """The publisher releases pending stderr readers while stopping."""
    async def run_case() -> tuple[_FakeTask, bool]:
        stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
        no_process_task = _FakeTask()
        stream._stderr_task = no_process_task  # type: ignore[assignment]
        await stream._stop_process()

        process = _FakeProcess(returncode=0)
        stderr_task = asyncio.create_task(asyncio.sleep(60))
        stream._process = process
        stream._stderr_task = stderr_task
        await stream._stop_process()
        return no_process_task, stderr_task.cancelled()

    no_process_task, stderr_cancelled = asyncio.run(run_case())

    assert no_process_task.cancelled is True
    assert stderr_cancelled is True


def test_drain_stderr_records_ffmpeg_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """FFmpeg stderr remains drained and retains its latest useful message."""
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    process = SimpleNamespace(stderr=_FakeStderr([b'warning\n', b'\n', b'']))

    asyncio.run(stream._drain_stderr(process))

    assert stream.last_error == 'warning'
    assert 'ffmpeg: warning' in capsys.readouterr().out


def test_drain_stderr_handles_missing_or_failing_reader() -> None:
    """An unavailable stderr pipe cannot crash a publisher task."""
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    asyncio.run(stream._drain_stderr(SimpleNamespace()))

    process = SimpleNamespace(
        stderr=_FakeStderr([RuntimeError('reader lost')]),
    )
    asyncio.run(stream._drain_stderr(process))

    assert stream.last_error == 'stderr reader failed: reader lost'


def test_drain_stderr_propagates_task_cancellation() -> None:
    """Stopping a publisher does not turn cancellation into a fake error."""
    async def run_case() -> None:
        stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
        wait_for_read = asyncio.Event()
        process = SimpleNamespace(
            stderr=SimpleNamespace(readline=lambda: wait_for_read.wait()),
        )
        task = asyncio.create_task(stream._drain_stderr(process))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run_case())


def test_close_clears_state_without_process() -> None:
    """Exercise this test."""
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    stream._latest_frame = np.zeros((2, 2, 3), dtype=np.uint8)
    stream._stream_size = (2, 2)
    stream._started = True

    asyncio.run(stream.close())

    assert stream._latest_frame is None
    assert stream._stream_size is None
    assert stream._started is False


def test_close_cancels_writer_task() -> None:
    """Exercise this test."""
    async def run_case() -> publisher.MediaStreamPublisher:
        """Support run_case."""
        stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
        stream._writer_task = asyncio.create_task(asyncio.sleep(60))
        await stream.close()
        return stream

    stream = asyncio.run(run_case())

    assert stream._writer_task is None


def test_reset_after_process_exit_cancels_writer_task() -> None:
    """Exercise this test."""
    async def run_case() -> publisher.MediaStreamPublisher:
        """Support run_case."""
        stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
        stream._writer_task = asyncio.create_task(asyncio.sleep(60))
        stream._stream_size = (4, 4)
        stream._started = True
        await stream._reset_after_process_exit()
        return stream

    stream = asyncio.run(run_case())

    assert stream._writer_task is None
    assert stream._stream_size is None
    assert stream._started is False


def test_start_uses_environment_ffmpeg(monkeypatch: Any) -> None:
    """Exercise this test."""
    async def fake_create_subprocess_exec(*args, **_kwargs) -> Any:
        """Support fake_create_subprocess_exec."""
        calls.append(args)
        return _FakeProcess()

    calls: list[tuple[object, ...]] = []
    monkeypatch.setenv('MEDIA_FFMPEG_PATH', '/custom/ffmpeg')
    monkeypatch.setenv('MEDIA_PUBLISH_ENCODER', 'libx264')
    monkeypatch.setattr(
        publisher.asyncio,
        'create_subprocess_exec',
        fake_create_subprocess_exec,
    )
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')

    asyncio.run(stream._start(320, 240))

    assert calls[0][0] == '/custom/ffmpeg'
    assert stream._stream_size == (320, 240)
    assert stream._started is True


def test_start_raises_when_ffmpeg_missing(monkeypatch: Any) -> None:
    """Exercise this test."""
    monkeypatch.delenv('MEDIA_FFMPEG_PATH', raising=False)
    monkeypatch.setattr(publisher.shutil, 'which', lambda _name: None)
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')

    with pytest.raises(RuntimeError, match='ffmpeg'):
        asyncio.run(stream._start(320, 240))


def test_writer_loop_writes_latest_frame(monkeypatch: Any) -> None:
    """Exercise this test."""
    async def no_sleep(_delay: float) -> None:
        """Support no_sleep."""
        return None

    process = _FakeProcess()
    process.stdin = _LoopStdin(process)
    stream = publisher.MediaStreamPublisher(
        'rtsp://127.0.0.1:8554/out',
        fps=100,
    )
    stream._process = process
    stream._latest_frame = np.zeros((2, 2, 3), dtype=np.uint8)
    monkeypatch.setattr(publisher.asyncio, 'sleep', no_sleep)

    asyncio.run(stream._writer_loop())

    assert process.stdin.writes
    assert process.stdin.transport.limits


def test_writer_loop_stops_when_process_missing() -> None:
    """Exercise this test."""
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    stream._started = True

    asyncio.run(stream._writer_loop())

    assert stream._started is False


def test_writer_loop_stops_when_stdin_fails() -> None:
    """Exercise this test."""
    process = _FakeProcess()
    process.stdin = _ErrorStdin()
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    stream._process = process
    stream._latest_frame = np.zeros((2, 2, 3), dtype=np.uint8)

    asyncio.run(stream._writer_loop())

    assert stream._started is False
    assert stream._process is None


def test_writer_loop_reraises_cancelled_error() -> None:
    """Exercise this test."""
    async def run_case() -> None:
        """Support run_case."""
        stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
        stream._process = _FakeProcess()
        task = asyncio.create_task(stream._writer_loop())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run_case())


def test_keyframe_interval_uses_default_for_invalid_environment(
        monkeypatch: Any,
) -> None:
    """An invalid GOP interval cannot prevent the publisher from starting."""
    monkeypatch.setenv('MEDIA_PUBLISH_KEYFRAME_INTERVAL_SECONDS', 'invalid')

    assert publisher._keyframe_interval_seconds() == 2.0


def test_publish_does_not_start_writer_when_first_frame_fails(
        monkeypatch: Any,
) -> None:
    """A failed initial stdin write leaves no orphan writer task."""
    async def fake_start(_width: int, _height: int) -> None:
        stream._started = True

    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    monkeypatch.setattr(stream, '_start', fake_start)
    monkeypatch.setattr(
        stream, '_write_first_frame',
        AsyncMock(return_value=False),
    )

    asyncio.run(stream.publish(np.zeros((2, 2, 3), dtype=np.uint8)))

    assert stream._writer_task is None


def test_start_falls_back_after_nvenc_is_marked_unavailable(
        monkeypatch: Any,
) -> None:
    """A previous NVENC startup error forces the next process onto x264."""
    async def fake_create_subprocess_exec(*args, **_kwargs) -> Any:
        commands.append(args)
        return _FakeProcess()

    commands: list[tuple[object, ...]] = []
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    stream._nvenc_unavailable = True
    monkeypatch.setenv('MEDIA_FFMPEG_PATH', '/custom/ffmpeg')
    monkeypatch.setattr(
        publisher, '_select_encoder',
        lambda _path: 'h264_nvenc',
    )
    monkeypatch.setattr(
        publisher.asyncio,
        'create_subprocess_exec',
        fake_create_subprocess_exec,
    )

    asyncio.run(stream._start(320, 240))
    asyncio.run(stream.close())

    assert 'libx264' in commands[0]
    assert stream._uses_nvenc is False


def test_start_releases_reserved_nvenc_when_process_creation_fails(
        monkeypatch: Any,
) -> None:
    """A failed ffmpeg launch cannot leak the local NVENC reservation."""
    async def fail_create_subprocess_exec(*_args, **_kwargs) -> Any:
        raise OSError('ffmpeg unavailable')

    releases: list[bool] = []
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    monkeypatch.setenv('MEDIA_FFMPEG_PATH', '/custom/ffmpeg')
    monkeypatch.setattr(
        publisher, '_select_encoder',
        lambda _path: 'h264_nvenc',
    )
    monkeypatch.setattr(publisher, 'try_acquire_nvenc_session', lambda: True)
    monkeypatch.setattr(
        publisher,
        'release_nvenc_session',
        lambda: releases.append(True),
    )
    monkeypatch.setattr(
        publisher.asyncio,
        'create_subprocess_exec',
        fail_create_subprocess_exec,
    )

    with pytest.raises(OSError, match='ffmpeg unavailable'):
        asyncio.run(stream._start(320, 240))

    assert releases == [True]
    assert stream._uses_nvenc is False


def test_vaapi_helpers_cover_device_options_and_errors(
        monkeypatch: Any,
) -> None:
    """VAAPI commands use defaults and reject inaccessible devices."""
    monkeypatch.setenv('MEDIA_PUBLISH_VAAPI_BITRATE', '')
    monkeypatch.setenv('MEDIA_PUBLISH_VAAPI_MAXRATE', '')
    monkeypatch.setenv('MEDIA_PUBLISH_VAAPI_BUFSIZE', '')
    monkeypatch.setenv('MEDIA_PUBLISH_VAAPI_DEVICE', '/dev/dri/custom')
    options = publisher._build_vaapi_options()

    assert publisher._default_vaapi_bitrate in options
    assert publisher._default_vaapi_maxrate in options
    assert publisher._default_vaapi_bufsize in options
    assert publisher._vaapi_device() == '/dev/dri/custom'

    monkeypatch.setattr(publisher.os, 'access', lambda *_args: False)
    with pytest.raises(RuntimeError, match='not accessible'):
        publisher._ensure_vaapi_device_access()

    monkeypatch.setattr(publisher.os, 'access', lambda *_args: True)
    publisher._ensure_vaapi_device_access()
    command = publisher.MediaStreamPublisher(
        'rtsp://127.0.0.1:8554/out',
    )._build_ffmpeg_command('/bin/ffmpeg', 320, 240, encoder='h264_vaapi')
    assert '-vaapi_device' in command


def test_select_encoder_rejects_unavailable_vaapi(monkeypatch: Any) -> None:
    """An explicit VAAPI choice must report a missing ffmpeg encoder."""
    monkeypatch.setenv('MEDIA_PUBLISH_ENCODER', 'vaapi')
    monkeypatch.setattr(publisher, '_ffmpeg_has_encoder', lambda *_args: False)

    with pytest.raises(RuntimeError, match='h264_vaapi'):
        publisher._select_encoder('/bin/ffmpeg')


def test_stderr_and_first_frame_failures_mark_publisher_unavailable() -> None:
    """NVENC errors and failed first writes are handled without a writer."""
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    stream._uses_nvenc = True
    process = SimpleNamespace(
        stderr=_FakeStderr([b'Cannot load libcuda\n', b'']),
    )
    asyncio.run(stream._drain_stderr(process))
    assert stream._nvenc_unavailable is True
    assert publisher._is_nvenc_unavailable_error('unrelated error') is False

    stream._process = _FakeProcess()
    stream._process.stdin = _ErrorStdin()
    stream._started = True
    assert asyncio.run(
        stream._write_first_frame(np.zeros((2, 2, 3), dtype=np.uint8)),
    ) is False
    assert stream._started is False


def test_start_covers_hardware_encoder_edge_cases(monkeypatch: Any) -> None:
    """VAAPI and exhausted NVENC sessions select the intended startup path."""
    async def fake_create_subprocess_exec(*args, **_kwargs) -> Any:
        commands.append(args)
        return _FakeProcess()

    commands: list[tuple[object, ...]] = []
    monkeypatch.setenv('MEDIA_FFMPEG_PATH', '/custom/ffmpeg')
    monkeypatch.setattr(
        publisher.asyncio,
        'create_subprocess_exec',
        fake_create_subprocess_exec,
    )
    vaapi = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/vaapi')
    with (
        monkeypatch.context() as context,
    ):
        context.setattr(
            publisher,
            '_select_encoder',
            lambda _path: 'h264_vaapi',
        )
        context.setattr(publisher, '_ensure_vaapi_device_access', lambda: None)
        asyncio.run(vaapi._start(320, 240))
    asyncio.run(vaapi.close())
    assert '-vaapi_device' in commands[0]

    nvenc = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/nvenc')
    with (
        monkeypatch.context() as context,
    ):
        context.setattr(
            publisher,
            '_select_encoder',
            lambda _path: 'h264_nvenc',
        )
        context.setattr(publisher, 'try_acquire_nvenc_session', lambda: False)
        asyncio.run(nvenc._start(320, 240))
    asyncio.run(nvenc.close())
    assert 'libx264' in commands[1]

    no_stdin = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    no_stdin._process = SimpleNamespace(stdin=None)
    no_stdin._started = True
    assert asyncio.run(
        no_stdin._write_first_frame(np.zeros((2, 2, 3), dtype=np.uint8)),
    ) is False


def test_select_encoder_accepts_available_vaapi(monkeypatch: Any) -> None:
    """An explicit VAAPI configuration is retained when ffmpeg supports it."""
    monkeypatch.setenv('MEDIA_PUBLISH_ENCODER', 'h264_vaapi')
    monkeypatch.setattr(publisher, '_ffmpeg_has_encoder', lambda *_args: True)

    assert publisher._select_encoder('/bin/ffmpeg') == 'h264_vaapi'
