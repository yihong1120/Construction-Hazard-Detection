from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from typing import Final

import cv2
import numpy as np


_default_fps: Final[float] = 10.0
_default_crf: Final[int] = 28
_default_preset: Final[str] = 'veryfast'
_default_encoder: Final[str] = 'libx264'
_default_nvenc_preset: Final[str] = 'p1'
_default_nvenc_cq: Final[int] = 30


class MediaStreamPublisher:
    """Publish processed frames to a media server as H.264 over RTSP."""

    def __init__(
        self,
        publish_url: str,
        fps: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """Initialise a frame publisher.

        Args:
            publish_url: RTSP URL receiving encoded frames.
            fps: Target output frame rate.
            width: Optional fixed output width.
            height: Optional fixed output height.
        """
        self.publish_url = publish_url
        self.fps = max(
            1.0,
            fps or float(os.getenv('MEDIA_PUBLISH_FPS', str(_default_fps))),
        )
        self.width = width
        self.height = height
        self._process: asyncio.subprocess.Process | None = None
        self._writer_task: asyncio.Task[None] | None = None
        self._latest_frame: np.ndarray | None = None
        self._stream_size: tuple[int, int] | None = None
        self._started = False
        self._state_lock = asyncio.Lock()

    async def publish(self, frame: np.ndarray) -> None:
        """Update the frame being continuously published to RTSP."""
        prepared = self._prepare_frame(frame)
        async with self._state_lock:
            if self._started and not self._is_process_alive():
                await self._reset_after_process_exit()

            if not self._started:
                await self._start(prepared.shape[1], prepared.shape[0])
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
        if process is None:
            return
        if process.stdin is not None and not process.stdin.is_closing():
            try:
                process.stdin.close()
            except (BrokenPipeError, ConnectionResetError):
                pass
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

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

        command = self._build_ffmpeg_command(ffmpeg_binary, width, height)
        self._process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        self.width = width
        self.height = height
        self._stream_size = (width, height)
        self._started = True

    async def _writer_loop(self) -> None:
        """Continuously feed ffmpeg so MediaMTX keeps the live path online."""
        frame_interval = 1.0 / self.fps
        next_write = asyncio.get_running_loop().time()
        try:
            while True:
                process = self._process
                latest_frame = self._latest_frame
                if process is None or process.stdin is None:
                    self._started = False
                    return
                if process.returncode is not None:
                    self._started = False
                    return
                if latest_frame is not None:
                    try:
                        payload = memoryview(latest_frame).cast('B')
                        transport = getattr(process.stdin, 'transport', None)
                        if transport is not None:
                            transport.set_write_buffer_limits(
                                high=len(payload) * 2,
                                low=len(payload),
                            )
                            if (
                                transport.get_write_buffer_size()
                                > len(payload)
                            ):
                                await asyncio.sleep(frame_interval)
                                continue
                        process.stdin.write(payload)
                        await asyncio.wait_for(
                            process.stdin.drain(),
                            timeout=max(1.0, frame_interval * 3),
                        )
                    except (
                        asyncio.TimeoutError,
                        BrokenPipeError,
                        ConnectionResetError,
                        RuntimeError,
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

    def _build_ffmpeg_command(
        self,
        ffmpeg_binary: str,
        width: int,
        height: int,
    ) -> list[str]:
        """Build an ffmpeg command that publishes H.264 to MediaMTX."""
        gop_size = max(1, round(self.fps * 2))
        encoder = _select_encoder(ffmpeg_binary)
        command = [
            ffmpeg_binary,
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
        ]
        if encoder == 'h264_nvenc':
            command.extend(_build_nvenc_options())
        else:
            command.extend(_build_x264_options())
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


def _build_x264_options() -> list[str]:
    """Return ffmpeg options for CPU H.264 encoding."""
    return [
        '-c:v',
        'libx264',
        '-preset',
        os.getenv('MEDIA_PUBLISH_PRESET', _default_preset),
        '-tune',
        'zerolatency',
        '-pix_fmt',
        'yuv420p',
        '-crf',
        str(int(os.getenv('MEDIA_PUBLISH_CRF', str(_default_crf)))),
    ]


def _build_nvenc_options() -> list[str]:
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
        os.getenv('MEDIA_PUBLISH_NVENC_BITRATE', '0'),
        '-maxrate',
        os.getenv('MEDIA_PUBLISH_NVENC_MAXRATE', '8M'),
        '-bufsize',
        os.getenv('MEDIA_PUBLISH_NVENC_BUFSIZE', '16M'),
        '-pix_fmt',
        'yuv420p',
    ]


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
