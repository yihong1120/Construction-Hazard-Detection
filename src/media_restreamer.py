from __future__ import annotations

import asyncio
import os
import shutil
from typing import Final


_restart_delay_seconds: Final[float] = 2.0
_default_fps: Final[float] = 15.0


class MediaSourceRestreamer:
    """Restream a source URL to MediaMTX without waiting for detection."""

    def __init__(
        self,
        source_url: str,
        publish_url: str,
    ) -> None:
        """Initialise a restreamer.

        Args:
            source_url: Input stream URL.
            publish_url: RTSP URL published to the media server.
        """
        self.source_url = source_url
        self.publish_url = publish_url
        self._process: asyncio.subprocess.Process | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._closed = False

    async def start(self) -> None:
        """Start a background ffmpeg restream process."""
        if self._monitor_task is not None:
            return
        self._closed = False
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def close(self) -> None:
        """Stop the restream process and its restart loop."""
        self._closed = True
        monitor_task = self._monitor_task
        self._monitor_task = None
        if (
            monitor_task is not None
            and monitor_task is not asyncio.current_task()
        ):
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
        await self._stop_process()

    async def _monitor_loop(self) -> None:
        """Keep the clean stream alive if the source briefly disconnects."""
        while not self._closed:
            ffmpeg_binary = _find_ffmpeg()
            encoder = _get_encoder()
            command = _build_command(
                ffmpeg_binary,
                self.source_url,
                self.publish_url,
                encoder,
            )
            self._process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )

            process = self._process
            if process is not None:
                await process.wait()
            if not self._closed:
                await asyncio.sleep(_restart_delay_seconds)

    async def _stop_process(self) -> None:
        """Terminate the current ffmpeg process if it is still running."""
        process = self._process
        self._process = None
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()


def _find_ffmpeg() -> str:
    """Resolve the ffmpeg executable path.

    Returns:
        Path or executable name used to launch ffmpeg.

    Raises:
        RuntimeError: If ffmpeg cannot be found.
    """
    ffmpeg_binary = os.getenv('MEDIA_FFMPEG_PATH', '').strip()
    if ffmpeg_binary:
        return ffmpeg_binary
    ffmpeg_binary = shutil.which('ffmpeg') or ''
    if not ffmpeg_binary:
        raise RuntimeError('Media restreaming requires ffmpeg')
    return ffmpeg_binary


def _build_command(
    ffmpeg_binary: str,
    source_url: str,
    publish_url: str,
    encoder: str,
) -> list[str]:
    """Build the ffmpeg command for clean stream restreaming.

    Args:
        ffmpeg_binary: ffmpeg executable path.
        source_url: Input stream URL.
        publish_url: RTSP output URL.
        encoder: Encoder selection.

    Returns:
        Command argument list.
    """
    command = [
        ffmpeg_binary,
        '-hide_banner',
        '-loglevel',
        os.getenv('MEDIA_FFMPEG_LOGLEVEL', 'error'),
        '-fflags',
        '+genpts+nobuffer',
        '-flags',
        'low_delay',
        '-use_wallclock_as_timestamps',
        '1',
    ]
    if source_url.lower().startswith('rtsp://'):
        command.extend(['-rtsp_transport', 'tcp'])
    command.extend(['-i', source_url, '-an'])
    if encoder == 'copy':
        command.extend(['-c:v', 'copy'])
    elif encoder in {'nvenc', 'h264_nvenc'}:
        command.extend(_build_nvenc_options())
    else:
        command.extend(_build_x264_options())
    if encoder != 'copy':
        command.extend(_build_timing_options())
    command.extend([
        '-f',
        'rtsp',
        '-rtsp_transport',
        'tcp',
        publish_url,
    ])
    return command


def _build_nvenc_options() -> list[str]:
    """Return ffmpeg options for NVIDIA H.264 encoding."""
    return [
        '-c:v',
        'h264_nvenc',
        '-preset',
        os.getenv('MEDIA_PUBLISH_NVENC_PRESET', 'p1'),
        '-tune',
        'ull',
        '-rc',
        os.getenv('MEDIA_PUBLISH_NVENC_RC', 'vbr'),
        '-cq',
        os.getenv('MEDIA_PUBLISH_NVENC_CQ', '30'),
        '-b:v',
        os.getenv('MEDIA_PUBLISH_NVENC_BITRATE', '0'),
        '-maxrate',
        os.getenv('MEDIA_PUBLISH_NVENC_MAXRATE', '8M'),
        '-bufsize',
        os.getenv('MEDIA_PUBLISH_NVENC_BUFSIZE', '16M'),
        '-pix_fmt',
        'yuv420p',
    ]


def _build_x264_options() -> list[str]:
    """Return ffmpeg options for CPU H.264 encoding."""
    return [
        '-c:v',
        'libx264',
        '-preset',
        os.getenv('MEDIA_PUBLISH_PRESET', 'veryfast'),
        '-tune',
        'zerolatency',
        '-pix_fmt',
        'yuv420p',
        '-crf',
        os.getenv('MEDIA_PUBLISH_CRF', '28'),
    ]


def _build_timing_options() -> list[str]:
    """Return frame rate and GOP options for encoded streams."""
    fps = max(
        1.0,
        float(os.getenv('MEDIA_PUBLISH_CLEAN_FPS', str(_default_fps))),
    )
    gop_size = max(1, round(fps * 2))
    return [
        '-r',
        f'{fps:g}',
        '-g',
        str(gop_size),
        '-keyint_min',
        str(gop_size),
        '-sc_threshold',
        '0',
    ]


def _get_encoder() -> str:
    """Return the configured clean stream encoder."""
    return (
        os.getenv('MEDIA_PUBLISH_CLEAN_ENCODER')
        or os.getenv('MEDIA_RESTREAM_ENCODER')
        or 'h264_nvenc'
    ).strip().lower()
