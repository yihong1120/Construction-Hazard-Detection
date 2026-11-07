from __future__ import annotations

import asyncio
from functools import partial
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest

from src import media_stream_publisher as publisher
from src.media_encoding import EncoderOptions
from src.settings.media import EncoderSettings


def _publish_starts_writer_for_first_fake_create_task(coro: Any) -> Any:
    """Support fake_create_task.

    Args:
        coro: Test helper value.
    """
    coro.close()
    return _FakeTask()


def _publish_resets_dead_process_fake_create_task(coro: Any) -> Any:
    """Support fake_create_task.

    Args:
        coro: Test helper value.
    """
    coro.close()
    return _FakeTask()


def _return_false(*_):
    return False


def _return_true(*_):
    return True


def _ffmpeg_has_encoder_handles_subprocess_fail(
    *_args: object,
    **_kwargs: object,
) -> object:
    """Support fail."""
    raise OSError('missing')


def _ffmpeg_has_encoder_reads_stdout_callback(result, /, *_a, **_k):
    return result


async def _stop_process_cancels_stderr_tasks_run_case() -> (
    tuple[_FakeTask, bool]
):
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


async def _drain_stderr_propagates_task_cancellation_run_case() -> None:
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    wait_for_read = asyncio.Event()
    process = SimpleNamespace(
        stderr=SimpleNamespace(
            readline=partial(
                _drain_stderr_propagates_task_cancellation_wait,
                wait_for_read,
            ),
        ),
    )
    task = asyncio.create_task(stream._drain_stderr(process))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def _drain_stderr_propagates_task_cancellation_wait(wait_for_read, /):
    return wait_for_read.wait()


async def _close_cancels_writer_task_run_case() -> (
    publisher.MediaStreamPublisher
):
    """Support run_case."""
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    stream._writer_task = asyncio.create_task(asyncio.sleep(60))
    await stream.close()
    return stream


async def _reset_after_process_exit_cancels_run_case() -> (
    publisher.MediaStreamPublisher
):
    """Support run_case."""
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    stream._writer_task = asyncio.create_task(asyncio.sleep(60))
    stream._stream_size = (4, 4)
    stream._started = True
    await stream._reset_after_process_exit()
    return stream


def _return_none(_name):
    return None


async def _writer_loop_writes_latest_frame_no_sleep(_delay: float) -> None:
    """Support no_sleep."""
    return None


async def _writer_loop_reraises_cancelled_error_run_case() -> None:
    """Support run_case."""
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    stream._process = _FakeProcess()
    task = asyncio.create_task(stream._writer_loop())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def _start_falls_back_after_nvenc_callback(*_args):
    return 'h264_nvenc'


async def _start_releases_reserved_nvenc_when_fail_create_subprocess_exec(
    *_args: Any,
    **_kwargs: Any,
) -> Any:
    raise OSError('ffmpeg unavailable')


def _return_true2():
    return True


def _start_releases_reserved_nvenc_when_append(releases, /):
    return releases.append(True)


def _return_false2(*_args):
    return False


def _return_true3(*_args):
    return True


def _start_covers_hardware_encoder_edge_callback(*_args):
    return 'h264_vaapi'


def _return_none2(*_args):
    return None


def _return_false3():
    return False


async def _vaapi_device_failure_selects_software_launch(
    commands,
    /,
    *command,
    **kwargs,
):
    commands.append(command)
    return SimpleNamespace(stderr=None)


class _PublishStartsWriterForFirstFrameCallbacks:
    starts: list[tuple[int, int]]
    stream: Any

    async def _call_fake_start(self, width: int, height: int) -> None:
        """Support fake_start."""
        self.starts.append((width, height))
        self.stream._process = _FakeProcess()
        self.stream._started = True
        self.stream._stream_size = (width, height)


class _PublishResetsDeadProcessCallbacks:
    resets: list[bool]
    starts: list[tuple[int, int]]
    stream: Any

    async def _call_fake_reset(self) -> None:
        """Support fake_reset."""
        self.resets.append(True)
        self.stream._started = False

    async def _call_fake_start(self, width: int, height: int) -> None:
        """Support fake_start."""
        self.starts.append((width, height))
        self.stream._started = True
        self.stream._stream_size = (width, height)


class _StartUsesEnvironmentFfmpegCallbacks:
    calls: list[tuple[object, ...]]

    async def _call_fake_create_subprocess_exec(
        self,
        *args: Any,
        **_kwargs: Any,
    ) -> Any:
        """Support fake_create_subprocess_exec."""
        self.calls.append(args)
        return _FakeProcess()


class _PublishDoesNotStartWriterWhenFirstCallbacks:
    stream: Any

    async def _call_fake_start(self, _width: int, _height: int) -> None:
        self.stream._started = True


class _StartFallsBackAfterNvencIsMarkedCallbacks:
    commands: list[tuple[object, ...]]

    async def _call_fake_create_subprocess_exec(
        self,
        *args: Any,
        **_kwargs: Any,
    ) -> Any:
        self.commands.append(args)
        return _FakeProcess()


class _StartCoversHardwareEncoderEdgeCasesCallbacks:
    commands: list[tuple[object, ...]]

    async def _call_fake_create_subprocess_exec(
        self,
        *args: Any,
        **_kwargs: Any,
    ) -> Any:
        self.commands.append(args)
        return _FakeProcess()


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
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    stream._stream_size = (4, 4)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    resized = stream._resize_to_stream_size(frame)

    assert resized.shape == (4, 4, 3)
    assert resized.flags['C_CONTIGUOUS']


def test_resize_to_stream_size_resizes_mismatched_frame() -> None:
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    stream._stream_size = (6, 4)
    frame = np.zeros((8, 10, 3), dtype=np.uint8)

    resized = stream._resize_to_stream_size(frame)

    assert resized.shape == (4, 6, 3)
    assert resized.flags['C_CONTIGUOUS']


def test_resize_to_stream_size_returns_frame_when_size_missing() -> None:
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    assert stream._resize_to_stream_size(frame) is frame


def test_is_process_alive_reflects_process_state() -> None:
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    assert stream._is_process_alive() is False

    stream._process = _FakeProcess()
    assert stream._is_process_alive() is True

    stream._process.returncode = 0
    assert stream._is_process_alive() is False


def test_publish_starts_writer_for_first_frame(monkeypatch: Any) -> None:

    _callbacks = _PublishStartsWriterForFirstFrameCallbacks()
    fake_start = _callbacks._call_fake_start

    fake_create_task = _publish_starts_writer_for_first_fake_create_task

    _callbacks.starts = []
    _callbacks.stream = publisher.MediaStreamPublisher(
        'rtsp://127.0.0.1:8554/out',
    )
    monkeypatch.setattr(_callbacks.stream, '_start', fake_start)
    monkeypatch.setattr(publisher.asyncio, 'create_task', fake_create_task)

    asyncio.run(_callbacks.stream.publish(np.zeros((4, 6, 3), dtype=np.uint8)))

    assert _callbacks.starts == [(6, 4)]
    assert _callbacks.stream._latest_frame is not None
    assert _callbacks.stream._writer_task is not None


def test_publish_resets_dead_process(monkeypatch: Any) -> None:

    _callbacks = _PublishResetsDeadProcessCallbacks()
    fake_reset = _callbacks._call_fake_reset

    fake_start = _callbacks._call_fake_start

    fake_create_task = _publish_resets_dead_process_fake_create_task

    _callbacks.resets = []
    _callbacks.starts = []
    _callbacks.stream = publisher.MediaStreamPublisher(
        'rtsp://127.0.0.1:8554/out',
    )
    _callbacks.stream._started = True
    _callbacks.stream._process = _FakeProcess(returncode=1)
    monkeypatch.setattr(
        _callbacks.stream,
        '_reset_after_process_exit',
        fake_reset,
    )
    monkeypatch.setattr(_callbacks.stream, '_start', fake_start)
    monkeypatch.setattr(publisher.asyncio, 'create_task', fake_create_task)

    asyncio.run(_callbacks.stream.publish(np.zeros((4, 6, 3), dtype=np.uint8)))

    assert _callbacks.resets == [True]
    assert _callbacks.starts == [(6, 4)]


def test_publish_resizes_late_frame() -> None:
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    stream._started = True
    stream._process = _FakeProcess()
    stream._stream_size = (6, 4)

    asyncio.run(stream.publish(np.zeros((8, 10, 3), dtype=np.uint8)))

    assert stream._latest_frame is not None
    assert stream._latest_frame.shape == (4, 6, 3)


def test_select_encoder_uses_x264_when_nvenc_missing(monkeypatch: Any) -> None:
    monkeypatch.setenv('MEDIA_PUBLISH_ENCODER', 'nvenc')
    monkeypatch.setattr(publisher, '_ffmpeg_has_encoder', _return_false)

    assert (
        publisher._select_encoder(
            '/usr/bin/ffmpeg',
            EncoderSettings(),
        )
        == 'libx264'
    )


def test_select_encoder_uses_auto_nvenc_when_available(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv('MEDIA_PUBLISH_ENCODER', 'auto')
    monkeypatch.setattr(publisher, '_ffmpeg_has_encoder', _return_true)

    assert (
        publisher._select_encoder(
            '/usr/bin/ffmpeg',
            EncoderSettings(),
        )
        == 'h264_nvenc'
    )


def test_build_ffmpeg_command_contains_rawvideo_and_rtsp(
    monkeypatch: Any,
) -> None:
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
    assert command[command.index('-pkt_size') + 1] == '1200'
    assert command[-1] == 'rtsp://127.0.0.1:8554/out'


def test_rtp_packet_size_is_mtu_safe(monkeypatch: Any) -> None:
    """Invalid and unsafe payload sizes use safe RTP limits."""
    monkeypatch.setenv('MEDIA_PUBLISH_RTP_PACKET_SIZE', 'invalid')
    assert EncoderSettings().rtp_packet_size == 1200

    monkeypatch.setenv('MEDIA_PUBLISH_RTP_PACKET_SIZE', '9000')
    assert EncoderSettings().rtp_packet_size == 1400

    monkeypatch.setenv('MEDIA_PUBLISH_RTP_PACKET_SIZE', '1')
    assert EncoderSettings().rtp_packet_size == 576


def test_build_ffmpeg_command_uses_nvenc_options(monkeypatch: Any) -> None:
    monkeypatch.setenv('MEDIA_PUBLISH_ENCODER', 'h264_nvenc')
    monkeypatch.setenv('MEDIA_PUBLISH_KEYFRAME_INTERVAL_SECONDS', '2')
    monkeypatch.setattr(publisher, '_ffmpeg_has_encoder', _return_true)
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

    fail = _ffmpeg_has_encoder_handles_subprocess_fail

    monkeypatch.setattr(publisher.subprocess, 'run', fail)

    assert (
        publisher._ffmpeg_has_encoder('/missing/ffmpeg', 'h264_nvenc') is False
    )


def test_ffmpeg_has_encoder_reads_stdout(monkeypatch: Any) -> None:
    result = SimpleNamespace(stdout=' V..... h264_nvenc NVIDIA NVENC H.264')
    monkeypatch.setattr(
        publisher.subprocess,
        'run',
        partial(_ffmpeg_has_encoder_reads_stdout_callback, result),
    )

    assert publisher._ffmpeg_has_encoder('/bin/ffmpeg', 'h264_nvenc') is True


def test_stop_process_terminates_live_process() -> None:
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    process = _FakeProcess()
    stream._process = process

    asyncio.run(stream._stop_process())

    assert stream._process is None
    assert process.stdin.closed
    assert process.terminated
    process.wait.assert_awaited_once()


def test_stop_process_kills_process_after_timeout() -> None:
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    process = _SlowProcess()
    stream._process = process

    asyncio.run(stream._stop_process())

    assert process.terminated
    assert process.killed


def test_stop_process_ignores_stdin_close_pipe_errors() -> None:
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    process = _FakeProcess(returncode=0)
    process.stdin = _RaisingStdin()
    stream._process = process

    asyncio.run(stream._stop_process())

    assert stream._process is None


def test_stop_process_cancels_stderr_tasks() -> None:
    """The publisher releases pending stderr readers while stopping."""

    run_case = _stop_process_cancels_stderr_tasks_run_case

    no_process_task, stderr_cancelled = asyncio.run(run_case())

    assert no_process_task.cancelled is True
    assert stderr_cancelled is True


def test_drain_stderr_records_ffmpeg_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """FFmpeg stderr remains drained and retains its latest useful message."""
    caplog.set_level('INFO', logger=publisher.__name__)
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    process = SimpleNamespace(stderr=_FakeStderr([b'warning\n', b'\n', b'']))

    asyncio.run(stream._drain_stderr(process))

    assert stream.last_error == 'warning'
    assert 'ffmpeg: warning' in caplog.text


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

    run_case = _drain_stderr_propagates_task_cancellation_run_case

    asyncio.run(run_case())


def test_close_clears_state_without_process() -> None:
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    stream._latest_frame = np.zeros((2, 2, 3), dtype=np.uint8)
    stream._stream_size = (2, 2)
    stream._started = True

    asyncio.run(stream.close())

    assert stream._latest_frame is None
    assert stream._stream_size is None
    assert stream._started is False


def test_close_cancels_writer_task() -> None:

    run_case = _close_cancels_writer_task_run_case

    stream = asyncio.run(run_case())

    assert stream._writer_task is None


def test_reset_after_process_exit_cancels_writer_task() -> None:

    run_case = _reset_after_process_exit_cancels_run_case

    stream = asyncio.run(run_case())

    assert stream._writer_task is None
    assert stream._stream_size is None
    assert stream._started is False


def test_start_uses_environment_ffmpeg(monkeypatch: Any) -> None:

    _callbacks = _StartUsesEnvironmentFfmpegCallbacks()
    fake_create_subprocess_exec = _callbacks._call_fake_create_subprocess_exec

    _callbacks.calls = []
    monkeypatch.setenv('MEDIA_FFMPEG_PATH', '/custom/ffmpeg')
    monkeypatch.setenv('MEDIA_PUBLISH_ENCODER', 'libx264')
    monkeypatch.setattr(
        publisher.asyncio,
        'create_subprocess_exec',
        fake_create_subprocess_exec,
    )
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')

    asyncio.run(stream._start(320, 240))

    assert _callbacks.calls[0][0] == '/custom/ffmpeg'
    assert stream._stream_size == (320, 240)
    assert stream._started is True


def test_start_raises_when_ffmpeg_missing(monkeypatch: Any) -> None:
    monkeypatch.delenv('MEDIA_FFMPEG_PATH', raising=False)
    monkeypatch.setattr(publisher.shutil, 'which', _return_none)
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')

    with pytest.raises(RuntimeError, match='ffmpeg'):
        asyncio.run(stream._start(320, 240))


def test_writer_loop_writes_latest_frame(monkeypatch: Any) -> None:

    no_sleep = _writer_loop_writes_latest_frame_no_sleep

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
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    stream._started = True

    asyncio.run(stream._writer_loop())

    assert stream._started is False


def test_writer_loop_stops_when_stdin_fails() -> None:
    process = _FakeProcess()
    process.stdin = _ErrorStdin()
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    stream._process = process
    stream._latest_frame = np.zeros((2, 2, 3), dtype=np.uint8)

    asyncio.run(stream._writer_loop())

    assert stream._started is False
    assert stream._process is None


def test_writer_loop_reraises_cancelled_error() -> None:

    run_case = _writer_loop_reraises_cancelled_error_run_case

    asyncio.run(run_case())


def test_keyframe_interval_uses_default_for_invalid_environment(
    monkeypatch: Any,
) -> None:
    """An invalid GOP interval cannot prevent the publisher from starting."""
    monkeypatch.setenv('MEDIA_PUBLISH_KEYFRAME_INTERVAL_SECONDS', 'invalid')

    assert EncoderSettings().keyframe_interval_seconds == 2.0


def test_publish_does_not_start_writer_when_first_frame_fails(
    monkeypatch: Any,
) -> None:
    """A failed initial stdin write leaves no orphan writer task."""
    _callbacks = _PublishDoesNotStartWriterWhenFirstCallbacks()

    fake_start = _callbacks._call_fake_start

    _callbacks.stream = publisher.MediaStreamPublisher(
        'rtsp://127.0.0.1:8554/out',
    )
    monkeypatch.setattr(_callbacks.stream, '_start', fake_start)
    monkeypatch.setattr(
        _callbacks.stream,
        '_write_first_frame',
        AsyncMock(return_value=False),
    )

    asyncio.run(_callbacks.stream.publish(np.zeros((2, 2, 3), dtype=np.uint8)))

    assert _callbacks.stream._writer_task is None


def test_start_falls_back_after_nvenc_is_marked_unavailable(
    monkeypatch: Any,
) -> None:
    """A previous NVENC startup error forces the next process onto x264."""
    _callbacks = _StartFallsBackAfterNvencIsMarkedCallbacks()

    fake_create_subprocess_exec = _callbacks._call_fake_create_subprocess_exec

    _callbacks.commands = []
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    stream._nvenc_unavailable = True
    monkeypatch.setenv('MEDIA_FFMPEG_PATH', '/custom/ffmpeg')
    monkeypatch.setattr(
        publisher,
        '_select_encoder',
        _start_falls_back_after_nvenc_callback,
    )
    monkeypatch.setattr(
        publisher.asyncio,
        'create_subprocess_exec',
        fake_create_subprocess_exec,
    )

    asyncio.run(stream._start(320, 240))
    asyncio.run(stream.close())

    assert 'libx264' in _callbacks.commands[0]
    assert stream._uses_nvenc is False


def test_start_releases_reserved_nvenc_when_process_creation_fails(
    monkeypatch: Any,
) -> None:
    """A failed ffmpeg launch cannot leak the local NVENC reservation."""

    fail_create_subprocess_exec = (
        _start_releases_reserved_nvenc_when_fail_create_subprocess_exec
    )

    releases: list[bool] = []
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    monkeypatch.setenv('MEDIA_FFMPEG_PATH', '/custom/ffmpeg')
    monkeypatch.setattr(
        publisher,
        '_select_encoder',
        _start_falls_back_after_nvenc_callback,
    )
    monkeypatch.setattr(publisher, 'try_acquire_nvenc_session', _return_true2)
    monkeypatch.setattr(
        publisher,
        'release_nvenc_session',
        partial(_start_releases_reserved_nvenc_when_append, releases),
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
    options = EncoderOptions().vaapi_options()

    assert options[options.index('-b:v') + 1] == '4M'
    assert options[options.index('-maxrate') + 1] == '8M'
    assert options[options.index('-bufsize') + 1] == '16M'
    assert publisher._vaapi_device(EncoderSettings()) == '/dev/dri/custom'

    monkeypatch.setattr(publisher.os, 'access', _return_false2)
    with pytest.raises(RuntimeError, match='not accessible'):
        publisher._ensure_vaapi_device_access(EncoderSettings())

    monkeypatch.setattr(publisher.os, 'access', _return_true3)
    publisher._ensure_vaapi_device_access(EncoderSettings())
    command = publisher.MediaStreamPublisher(
        'rtsp://127.0.0.1:8554/out',
    )._build_ffmpeg_command('/bin/ffmpeg', 320, 240, encoder='h264_vaapi')
    assert '-vaapi_device' in command


def test_select_encoder_rejects_unavailable_vaapi(monkeypatch: Any) -> None:
    """An explicit VAAPI choice must report a missing ffmpeg encoder."""
    monkeypatch.setenv('MEDIA_PUBLISH_ENCODER', 'vaapi')
    monkeypatch.setattr(publisher, '_ffmpeg_has_encoder', _return_false2)

    with pytest.raises(RuntimeError, match='h264_vaapi'):
        publisher._select_encoder('/bin/ffmpeg', EncoderSettings())


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
    assert (
        asyncio.run(
            stream._write_first_frame(np.zeros((2, 2, 3), dtype=np.uint8)),
        )
        is False
    )
    assert stream._started is False


def test_start_covers_hardware_encoder_edge_cases(monkeypatch: Any) -> None:
    """VAAPI and exhausted NVENC sessions select the intended startup path."""
    _callbacks = _StartCoversHardwareEncoderEdgeCasesCallbacks()

    fake_create_subprocess_exec = _callbacks._call_fake_create_subprocess_exec

    _callbacks.commands = []
    monkeypatch.setenv('MEDIA_FFMPEG_PATH', '/custom/ffmpeg')
    monkeypatch.setattr(
        publisher.asyncio,
        'create_subprocess_exec',
        fake_create_subprocess_exec,
    )
    monkeypatch.setenv('MEDIA_PUBLISH_VAAPI_DEVICE', '/dev/dri/test-intel')
    vaapi = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/vaapi')
    with monkeypatch.context() as context:
        context.setattr(
            publisher,
            '_select_encoder',
            _start_covers_hardware_encoder_edge_callback,
        )
        context.setattr(
            publisher,
            '_ensure_vaapi_device_access',
            _return_none2,
        )
        context.setenv('MEDIA_PUBLISH_VAAPI_DEVICE', '/dev/dri/test-intel')
        asyncio.run(vaapi._start(320, 240))
    asyncio.run(vaapi.close())
    assert '-vaapi_device' in _callbacks.commands[0]

    nvenc = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/nvenc')
    with monkeypatch.context() as context:
        context.setattr(
            publisher,
            '_select_encoder',
            _start_falls_back_after_nvenc_callback,
        )
        context.setattr(publisher, 'try_acquire_nvenc_session', _return_false3)
        asyncio.run(nvenc._start(320, 240))
    asyncio.run(nvenc.close())
    assert 'libx264' in _callbacks.commands[1]

    no_stdin = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    no_stdin._process = SimpleNamespace(stdin=None)
    no_stdin._started = True
    assert (
        asyncio.run(
            no_stdin._write_first_frame(np.zeros((2, 2, 3), dtype=np.uint8)),
        )
        is False
    )


def test_select_encoder_accepts_available_vaapi(monkeypatch: Any) -> None:
    """An explicit VAAPI configuration is retained when ffmpeg supports it."""
    monkeypatch.setenv('MEDIA_PUBLISH_ENCODER', 'h264_vaapi')
    monkeypatch.setattr(publisher, '_ffmpeg_has_encoder', _return_true3)

    assert (
        publisher._select_encoder(
            '/bin/ffmpeg',
            EncoderSettings(),
        )
        == 'h264_vaapi'
    )


def test_vaapi_device_failure_selects_software_on_next_start(
    monkeypatch: Any,
) -> None:
    """A driver failure is remembered and does not loop on broken hardware."""
    stream = publisher.MediaStreamPublisher('rtsp://127.0.0.1:8554/out')
    stream._uses_vaapi = True
    process = SimpleNamespace(
        stderr=_FakeStderr(
            [
                b'Failed to initialise VAAPI connection: -1\n',
                b'',
            ],
        ),
    )
    asyncio.run(stream._drain_stderr(process))
    assert stream._vaapi_unavailable
    monkeypatch.setenv('MEDIA_FFMPEG_PATH', '/custom/ffmpeg')
    monkeypatch.setattr(
        publisher,
        '_select_encoder',
        _start_covers_hardware_encoder_edge_callback,
    )
    commands: list[tuple[object, ...]] = []

    launch = partial(_vaapi_device_failure_selects_software_launch, commands)

    monkeypatch.setattr(publisher.asyncio, 'create_subprocess_exec', launch)
    asyncio.run(stream._start(640, 360))
    assert 'libx264' in commands[0]
    assert 'h264_vaapi' not in commands[0]


def test_vaapi_auto_selects_intel_instead_of_nvidia(
    tmp_path,
    monkeypatch: Any,
) -> None:
    """Render-node numbering can change when integrated graphics is enabled."""
    for name, vendor in [('renderD128', '0x10de'), ('renderD129', '0x8086')]:
        device = tmp_path / name / 'device'
        device.mkdir(parents=True)
        (device / 'vendor').write_text(vendor)
    assert (
        publisher._find_intel_vaapi_device(
            tmp_path,
        )
        == '/dev/dri/renderD129'
    )
    (tmp_path / 'renderD129/device/vendor').write_text('0x10de')
    with pytest.raises(RuntimeError, match='Enable integrated graphics'):
        publisher._find_intel_vaapi_device(tmp_path)
