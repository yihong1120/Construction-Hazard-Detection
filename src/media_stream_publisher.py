from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from collections.abc import Awaitable
from typing import Final
from typing import Protocol

import cv2
import numpy as np

from src.nvenc_session import release_nvenc_session
from src.nvenc_session import try_acquire_nvenc_session


_default_fps: Final[float] = 10.0
_default_crf: Final[int] = 28
_default_preset: Final[str] = 'veryfast'
_default_encoder: Final[str] = 'libx264'
_default_nvenc_preset: Final[str] = 'p1'
_default_nvenc_cq: Final[int] = 30
_default_vaapi_device: Final[str] = '/dev/dri/renderD128'
_default_vaapi_bitrate: Final[str] = '4M'
_default_vaapi_maxrate: Final[str] = '8M'
_default_vaapi_bufsize: Final[str] = '16M'


def _keyframe_interval_seconds() -> float:
    """Return the maximum time between H.264 keyframes."""
    try:
        return max(
            0.1,
            float(os.getenv('MEDIA_PUBLISH_KEYFRAME_INTERVAL_SECONDS', '2')),
        )
    except ValueError:
        return 2.0


class _FfmpegStdin(Protocol):
    """Writable stdin interface used to feed frames to ffmpeg."""

    def is_closing(self) -> bool:
        """Return whether no more writes are accepted."""

    def close(self) -> None:
        """Close the input stream."""

    def write(self, payload: memoryview) -> None:
        """Write a frame payload."""

    def drain(self) -> Awaitable[None]:
        """Wait until buffered writes are accepted."""


class _FfmpegProcess(Protocol):
    """Subprocess operations used by the frame publisher."""

    @property
    def returncode(self) -> int | None:
        """Return the process exit status when available."""

    @property
    def stdin(self) -> _FfmpegStdin | None:
        """Return the stream accepting raw frame data."""

    def terminate(self) -> None:
        """Request graceful process termination."""

    def kill(self) -> None:
        """Force process termination."""

    def wait(self) -> Awaitable[int | None]:
        """Wait for process completion."""


class _CancellableTask(Awaitable[None], Protocol):
    """Background task operations needed while closing the publisher."""

    def cancel(self) -> bool:
        """Request cancellation of the task."""

    def done(self) -> bool:
        """Return whether the task has completed."""


class MediaStreamPublisher:
    """Publish processed frames to a media server as H.264 over RTSP."""

    def __init__(
        self,
        publish_url: str,
        fps: float | None = None,
        width: int | None = None,
        height: int | None = None,
        bitrate: str | None = None,
        maxrate: str | None = None,
        bufsize: str | None = None,
    ) -> None:
        """Initialise a frame publisher.

        Args:
            publish_url: RTSP URL receiving encoded frames.
            fps: Target output frame rate.
            width: Optional fixed output width.
            height: Optional fixed output height.
            bitrate: Optional target video bitrate for this rendition.
            maxrate: Optional maximum video bitrate for this rendition.
            bufsize: Optional video rate-control buffer for this rendition.
        """
        self.publish_url = publish_url
        self.fps = max(
            1.0,
            fps or float(os.getenv('MEDIA_PUBLISH_FPS', str(_default_fps))),
        )
        self.width = width
        self.height = height
        self.bitrate = bitrate
        self.maxrate = maxrate
        self.bufsize = bufsize
        self._process: _FfmpegProcess | None = None
        self._writer_task: asyncio.Task[None] | None = None
        self._stderr_task: _CancellableTask | None = None
        self._latest_frame: np.ndarray | None = None
        self._stream_size: tuple[int, int] | None = None
        self._started = False
        self._uses_nvenc = False
        self._nvenc_unavailable = False
        self.last_error: str | None = None
        self._state_lock = asyncio.Lock()

    async def publish(self, frame: np.ndarray) -> None:
        """Update the frame being continuously published to RTSP."""
        prepared = self._prepare_frame(frame)
        async with self._state_lock:
            if self._started and not self._is_process_alive():
                await self._reset_after_process_exit()

            if not self._started:
                await self._start(prepared.shape[1], prepared.shape[0])
                self._latest_frame = prepared
                if not await self._write_first_frame(prepared):
                    return
                self._writer_task = asyncio.create_task(self._writer_loop())
            elif self._stream_size is not None:
                prepared = self._resize_to_stream_size(prepared)
        self._latest_frame = prepared

    async def close(self) -> None:
        """Close the ffmpeg publisher process."""
        writer_task = self._writer_task
        self._writer_task = None
        if (
            writer_task is not None
            and writer_task is not asyncio.current_task()
        ):
            writer_task.cancel()
            try:
                await writer_task
            except asyncio.CancelledError:
                pass

        await self._stop_process()
        self._latest_frame = None
        self._stream_size = None
        self._started = False

    def _is_process_alive(self) -> bool:
        """Return True when ffmpeg is still available for stdin writes."""
        return self._process is not None and self._process.returncode is None

    async def _reset_after_process_exit(self) -> None:
        """Reset state so the next frame starts a fresh ffmpeg process."""
        writer_task = self._writer_task
        self._writer_task = None
        if (
            writer_task is not None
            and writer_task is not asyncio.current_task()
        ):
            if not writer_task.done():
                writer_task.cancel()
                try:
                    await writer_task
                except asyncio.CancelledError:
                    pass
        await self._stop_process()
        self._stream_size = None
        self._started = False

    async def _stop_process(self) -> None:
        """Terminate the active ffmpeg process if one exists."""
        process = self._process
        self._process = None
        stderr_task = self._stderr_task
        self._stderr_task = None
        if process is None:
            self._cancel_detached_stderr_task(stderr_task)
            return
        self._close_process_stdin(process)
        await self._terminate_process(process)
        await self._await_stderr_task(stderr_task)
        self._release_nvenc_session()

    @staticmethod
    def _cancel_detached_stderr_task(
        stderr_task: _CancellableTask | None,
    ) -> None:
        """Cancel a stderr reader when its ffmpeg process is already gone."""
        if (
            stderr_task is not None
            and stderr_task is not asyncio.current_task()
        ):
            stderr_task.cancel()

    @staticmethod
    def _close_process_stdin(process: _FfmpegProcess) -> None:
        """Close ffmpeg stdin without surfacing a completed-pipe error."""
        stdin = process.stdin
        if stdin is None or stdin.is_closing():
            return
        try:
            stdin.close()
        except (BrokenPipeError, ConnectionResetError):
            pass

    @staticmethod
    async def _terminate_process(process: _FfmpegProcess) -> None:
        """Terminate ffmpeg, escalating after its graceful shutdown timeout."""
        if process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

    @staticmethod
    async def _await_stderr_task(
        stderr_task: _CancellableTask | None,
    ) -> None:
        """Stop and await a stderr reader owned by a closing process."""
        if stderr_task is None or stderr_task is asyncio.current_task():
            return
        if not stderr_task.done():
            stderr_task.cancel()
        try:
            await stderr_task
        except asyncio.CancelledError:
            pass

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """Normalise size and dimensions before sending to ffmpeg."""
        prepared = frame
        if self.width and self.height:
            prepared = cv2.resize(
                prepared,
                (self.width, self.height),
                interpolation=cv2.INTER_AREA,
            )
        height, width = prepared.shape[:2]
        even_width = width - (width % 2)
        even_height = height - (height % 2)
        if even_width != width or even_height != height:
            prepared = prepared[:even_height, :even_width]
        return np.ascontiguousarray(prepared)

    def _resize_to_stream_size(self, frame: np.ndarray) -> np.ndarray:
        """Resize late frames to the dimensions used by the running encoder."""
        if self._stream_size is None:
            return frame
        width, height = self._stream_size
        frame_height, frame_width = frame.shape[:2]
        if frame_width == width and frame_height == height:
            return np.ascontiguousarray(frame)
        return np.ascontiguousarray(
            cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA),
        )

    async def _start(self, width: int, height: int) -> None:
        """Start ffmpeg using the first frame's dimensions."""
        ffmpeg_binary = os.getenv('MEDIA_FFMPEG_PATH', '').strip()
        if not ffmpeg_binary:
            ffmpeg_binary = shutil.which('ffmpeg') or ''
        if not ffmpeg_binary:
            raise RuntimeError('Media publishing requires ffmpeg')

        encoder = _select_encoder(ffmpeg_binary)
        if encoder == 'h264_vaapi':
            _ensure_vaapi_device_access()
        if encoder == 'h264_nvenc' and self._nvenc_unavailable:
            encoder = 'libx264'
        elif encoder == 'h264_nvenc' and not try_acquire_nvenc_session():
            encoder = 'libx264'
            print(
                f'[media:{self.publish_url}] NVENC session budget reached; '
                'using libx264',
                flush=True,
            )
        else:
            self._uses_nvenc = encoder == 'h264_nvenc'

        command = self._build_ffmpeg_command(
            ffmpeg_binary,
            width,
            height,
            encoder=encoder,
        )
        try:
            self._process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception:
            self._release_nvenc_session()
            raise
        self._stderr_task = asyncio.create_task(
            self._drain_stderr(self._process),
        )
        self.width = width
        self.height = height
        self._stream_size = (width, height)
        self._started = True

    async def _drain_stderr(
        self,
        process: object,
    ) -> None:
        """Drain ffmpeg stderr so an encoder error cannot block the pipe."""
        stderr = getattr(process, 'stderr', None)
        if stderr is None:
            return
        try:
            while True:
                raw_line = await stderr.readline()
                if not raw_line:
                    return
                line = raw_line.decode('utf-8', errors='replace').strip()
                if not line:
                    continue
                self.last_error = line[-1000:]
                if self._uses_nvenc and _is_nvenc_unavailable_error(line):
                    self._nvenc_unavailable = True
                print(
                    f'[media:{self.publish_url}] ffmpeg: {self.last_error}',
                    flush=True,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.last_error = f'stderr reader failed: {exc}'

    async def _write_first_frame(self, frame: np.ndarray) -> bool:
        """Write one frame before a demand change can cancel the publisher."""
        process = self._process
        if process is None or process.stdin is None:
            self._started = False
            return False
        try:
            process.stdin.write(memoryview(frame).cast('B'))
            await asyncio.wait_for(
                process.stdin.drain(),
                timeout=max(1.0, (1.0 / self.fps) * 3),
            )
        except (
            asyncio.TimeoutError,
            BrokenPipeError,
            ConnectionResetError,
            RuntimeError,
        ):
            await self._stop_process()
            self._started = False
            return False
        return True

    async def _writer_loop(self) -> None:
        """Continuously feed ffmpeg so MediaMTX keeps the live path online."""
        frame_interval = 1.0 / self.fps
        next_write = asyncio.get_running_loop().time()
        try:
            while True:
                process = self._process
                latest_frame = self._latest_frame
                if not self._can_write_to_process(process):
                    self._started = False
                    return
                assert process is not None
                if latest_frame is not None:
                    if not await self._write_latest_frame(
                        process,
                        latest_frame,
                        frame_interval,
                    ):
                        await self._stop_process()
                        self._started = False
                        return
                next_write = max(
                    next_write + frame_interval,
                    asyncio.get_running_loop().time() + frame_interval,
                )
                await asyncio.sleep(
                    max(0.0, next_write - asyncio.get_running_loop().time()),
                )
        except asyncio.CancelledError:
            raise

    @staticmethod
    def _can_write_to_process(process: _FfmpegProcess | None) -> bool:
        """Return whether an ffmpeg process can accept a frame."""
        return (
            process is not None
            and process.stdin is not None
            and process.returncode is None
        )

    @staticmethod
    async def _write_latest_frame(
        process: _FfmpegProcess,
        frame: np.ndarray,
        frame_interval: float,
    ) -> bool:
        """Write one latest frame while applying a bounded pipe backlog."""
        stdin = process.stdin
        if stdin is None:
            return False
        try:
            payload = memoryview(frame).cast('B')
            transport = getattr(stdin, 'transport', None)
            if transport is not None:
                transport.set_write_buffer_limits(
                    high=len(payload) * 2,
                    low=len(payload),
                )
                if transport.get_write_buffer_size() > len(payload):
                    await asyncio.sleep(frame_interval)
                    return True
            stdin.write(payload)
            await asyncio.wait_for(
                stdin.drain(),
                timeout=max(1.0, frame_interval * 3),
            )
        except (
            asyncio.TimeoutError,
            BrokenPipeError,
            ConnectionResetError,
            RuntimeError,
        ):
            return False
        return True

    def _build_ffmpeg_command(
        self,
        ffmpeg_binary: str,
        width: int,
        height: int,
        *,
        encoder: str | None = None,
    ) -> list[str]:
        """Build an ffmpeg command that publishes H.264 to MediaMTX."""
        gop_size = max(1, round(self.fps * _keyframe_interval_seconds()))
        encoder = encoder or _select_encoder(ffmpeg_binary)
        command = [
            ffmpeg_binary,
        ]
        if encoder == 'h264_vaapi':
            command.extend([
                '-vaapi_device',
                _vaapi_device(),
            ])
        command.extend([
            '-hide_banner',
            '-loglevel',
            os.getenv('MEDIA_FFMPEG_LOGLEVEL', 'error'),
            '-re',
            '-fflags',
            '+genpts+nobuffer',
            '-use_wallclock_as_timestamps',
            '1',
            '-f',
            'rawvideo',
            '-pix_fmt',
            'bgr24',
            '-s:v',
            f'{width}x{height}',
            '-framerate',
            f'{self.fps:g}',
            '-i',
            'pipe:0',
            '-an',
        ])
        if encoder == 'h264_nvenc':
            command.extend(
                _build_nvenc_options(
                    bitrate=self.bitrate,
                    maxrate=self.maxrate,
                    bufsize=self.bufsize,
                ),
            )
        elif encoder == 'h264_vaapi':
            command.extend(
                _build_vaapi_options(
                    bitrate=self.bitrate,
                    maxrate=self.maxrate,
                    bufsize=self.bufsize,
                ),
            )
        else:
            command.extend(
                _build_x264_options(
                    bitrate=self.bitrate,
                    maxrate=self.maxrate,
                    bufsize=self.bufsize,
                ),
            )
        command.extend([
            '-g',
            str(gop_size),
            '-keyint_min',
            str(gop_size),
            '-sc_threshold',
            '0',
            '-r',
            f'{self.fps:g}',
            '-fps_mode',
            'cfr',
            '-f',
            'rtsp',
            '-rtsp_transport',
            'tcp',
            self.publish_url,
        ])
        return command

    def _release_nvenc_session(self) -> None:
        """Return this publisher's NVENC reservation exactly once."""
        if self._uses_nvenc:
            release_nvenc_session()
            self._uses_nvenc = False


def _build_x264_options(
    *,
    bitrate: str | None = None,
    maxrate: str | None = None,
    bufsize: str | None = None,
) -> list[str]:
    """Return ffmpeg options for CPU H.264 encoding."""
    options = [
        '-c:v',
        'libx264',
        '-preset',
        os.getenv('MEDIA_PUBLISH_PRESET', _default_preset),
        '-tune',
        'zerolatency',
        '-pix_fmt',
        'yuv420p',
    ]
    if bitrate:
        options.extend([
            '-b:v', bitrate,
            '-maxrate', maxrate or bitrate,
            '-bufsize', bufsize or maxrate or bitrate,
        ])
    else:
        options.extend([
            '-crf',
            str(int(os.getenv('MEDIA_PUBLISH_CRF', str(_default_crf)))),
        ])
    return options


def _build_nvenc_options(
    *,
    bitrate: str | None = None,
    maxrate: str | None = None,
    bufsize: str | None = None,
) -> list[str]:
    """Return ffmpeg options for NVIDIA H.264 encoding."""
    return [
        '-c:v',
        'h264_nvenc',
        '-preset',
        os.getenv('MEDIA_PUBLISH_NVENC_PRESET', _default_nvenc_preset),
        '-tune',
        'ull',
        '-rc',
        os.getenv('MEDIA_PUBLISH_NVENC_RC', 'vbr'),
        '-cq',
        str(int(os.getenv('MEDIA_PUBLISH_NVENC_CQ', str(_default_nvenc_cq)))),
        '-b:v',
        bitrate or os.getenv('MEDIA_PUBLISH_NVENC_BITRATE', '0') or '0',
        '-maxrate',
        maxrate or os.getenv('MEDIA_PUBLISH_NVENC_MAXRATE', '8M') or '8M',
        '-bufsize',
        bufsize or os.getenv('MEDIA_PUBLISH_NVENC_BUFSIZE', '16M') or '16M',
        '-pix_fmt',
        'yuv420p',
    ]


def _build_vaapi_options(
    *,
    bitrate: str | None = None,
    maxrate: str | None = None,
    bufsize: str | None = None,
) -> list[str]:
    """Return Intel VAAPI H.264 options for low-latency live streams."""
    target_bitrate = (
        bitrate
        or os.getenv('MEDIA_PUBLISH_VAAPI_BITRATE', _default_vaapi_bitrate)
        or _default_vaapi_bitrate
    )
    target_maxrate = (
        maxrate
        or os.getenv('MEDIA_PUBLISH_VAAPI_MAXRATE', _default_vaapi_maxrate)
        or _default_vaapi_maxrate
    )
    target_bufsize = (
        bufsize
        or os.getenv('MEDIA_PUBLISH_VAAPI_BUFSIZE', _default_vaapi_bufsize)
        or _default_vaapi_bufsize
    )
    return [
        '-vf',
        'format=nv12,hwupload',
        '-c:v',
        'h264_vaapi',
        '-rc_mode',
        os.getenv('MEDIA_PUBLISH_VAAPI_RC_MODE', 'CBR'),
        '-b:v',
        target_bitrate,
        '-maxrate',
        target_maxrate,
        '-bufsize',
        target_bufsize,
        '-bf',
        '0',
        '-async_depth',
        os.getenv('MEDIA_PUBLISH_VAAPI_ASYNC_DEPTH', '1'),
        '-profile:v',
        'high',
    ]


def _vaapi_device() -> str:
    """Return the Intel render node used for VAAPI encoding."""
    return os.getenv(
        'MEDIA_PUBLISH_VAAPI_DEVICE',
        _default_vaapi_device,
    )


def _ensure_vaapi_device_access() -> None:
    """Fail clearly when the process cannot use the configured Intel GPU."""
    device = _vaapi_device()
    if os.access(device, os.R_OK | os.W_OK):
        return
    raise RuntimeError(
        f'VAAPI device is not accessible: {device}. Add the process user to '
        'the render group, then start a new login shell.',
    )


def _select_encoder(ffmpeg_binary: str) -> str:
    """Select an available encoder for ffmpeg.

    Args:
        ffmpeg_binary: ffmpeg executable path.

    Returns:
        Encoder name accepted by ffmpeg.
    """
    configured = os.getenv(
        'MEDIA_PUBLISH_ENCODER',
        _default_encoder,
    ).strip().lower()
    if configured in {'nvenc', 'h264_nvenc'}:
        return 'h264_nvenc' if _ffmpeg_has_encoder(
            ffmpeg_binary,
            'h264_nvenc',
        ) else 'libx264'
    if configured in {'vaapi', 'h264_vaapi'}:
        if _ffmpeg_has_encoder(ffmpeg_binary, 'h264_vaapi'):
            return 'h264_vaapi'
        raise RuntimeError(
            'MEDIA_PUBLISH_ENCODER=h264_vaapi requires an ffmpeg build '
            'with h264_vaapi support',
        )
    if configured in {'auto', 'hardware'} and _ffmpeg_has_encoder(
        ffmpeg_binary,
        'h264_nvenc',
    ):
        return 'h264_nvenc'
    return 'libx264'


def _ffmpeg_has_encoder(ffmpeg_binary: str, encoder: str) -> bool:
    """Return whether ffmpeg advertises an encoder."""
    try:
        result = subprocess.run(
            [ffmpeg_binary, '-hide_banner', '-encoders'],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return False
    return encoder in result.stdout


def _is_nvenc_unavailable_error(line: str) -> bool:
    """Return whether an ffmpeg error means this publisher must use CPU."""
    return any(
        marker in line
        for marker in (
            'No capable devices found',
            'OpenEncodeSessionEx failed',
            'Cannot load libcuda',
        )
    )
