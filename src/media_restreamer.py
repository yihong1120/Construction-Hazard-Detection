from __future__ import annotations

import asyncio
import logging
import shutil
from collections.abc import Awaitable
from typing import Final
from typing import Protocol

from src.media_encoding import EncoderOptions
from src.nvenc_session import release_nvenc_session
from src.nvenc_session import try_acquire_nvenc_session

_restart_delay_seconds: Final[float] = 2.0


logger = logging.getLogger(__name__)


class _FfmpegProcess(Protocol):
    """Subprocess operations used by the clean-stream restreamer."""

    @property
    def returncode(self) -> int | None:
        """Return the process exit status when available."""

    def terminate(self) -> None:
        """Request graceful process termination."""

    def kill(self) -> None:
        """Force process termination."""

    def wait(self) -> Awaitable[int | None]:
        """Wait for process completion."""


class MediaSourceRestreamer:
    """Restream a source URL to MediaMTX without waiting for detection."""

    def __init__(
        self,
        source_url: str,
        publish_url: str,
        encoding: EncoderOptions | None = None,
    ) -> None:
        """Initialise a restreamer.

        Args:
            source_url: Input stream URL.
            publish_url: RTSP URL published to the media server.
        """
        self.commands = SourceCommandBuilder(
            encoding if encoding is not None else EncoderOptions(),
        )
        self.source_url = source_url
        self.publish_url = publish_url
        self._process: _FfmpegProcess | None = None
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

    async def restart(self) -> None:
        """Reconnect the current source without ending the monitor loop."""
        if self._closed:
            return
        await self._stop_process()

    async def _monitor_loop(self) -> None:
        """Keep the clean stream alive if the source briefly disconnects."""
        while not self._closed:
            ffmpeg_binary = self.commands.find_ffmpeg()
            encoder = self.commands.get_encoder()
            uses_nvenc = False
            if encoder in {'nvenc', 'h264_nvenc'}:
                if try_acquire_nvenc_session():
                    encoder = 'h264_nvenc'
                    uses_nvenc = True
                else:
                    encoder = 'libx264'
                    logger.info(
                        f"[media:{self.publish_url}] NVENC session budget "
                        'reached; using libx264',
                    )
            command = self.commands.build_command(
                ffmpeg_binary,
                self.source_url,
                self.publish_url,
                encoder,
            )
            try:
                self._process = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )

                process = self._process
                if process is not None:
                    await process.wait()
            finally:
                if uses_nvenc:
                    release_nvenc_session()
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
        except TimeoutError:
            process.kill()
            await process.wait()


class SourceCommandBuilder:
    def __init__(self, encoding: EncoderOptions) -> None:
        self.encoding = encoding

    def find_ffmpeg(self) -> str:
        """Resolve the ffmpeg executable path.

        Returns:
            Path or executable name used to launch ffmpeg.

        Raises:
            RuntimeError: If ffmpeg cannot be found.
        """
        ffmpeg_binary = self.encoding.settings.ffmpeg_path
        if ffmpeg_binary:
            return ffmpeg_binary
        ffmpeg_binary = shutil.which('ffmpeg') or ''
        if not ffmpeg_binary:
            raise RuntimeError('Media restreaming requires ffmpeg')
        return ffmpeg_binary

    def build_command(
        self,
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
            self.encoding.settings.ffmpeg_loglevel,
            '-fflags',
            '+genpts+nobuffer',
            '-flags',
            'low_delay',
        ]
        if source_url.lower().startswith('rtsp://'):
            command.extend(['-rtsp_transport', 'tcp'])
        command.extend(['-i', source_url, '-an'])
        if encoder == 'copy':
            command.extend(['-c:v', 'copy'])
        elif encoder in {'nvenc', 'h264_nvenc'}:
            command.extend(self.encoding.nvenc_options())
        else:
            command.extend(self.encoding.x264_options())
        if encoder != 'copy':
            command.extend(self.build_timing_options())
        command.extend(
            [
                '-f',
                'rtsp',
                '-rtsp_transport',
                'tcp',
                publish_url,
            ],
        )
        return command

    def build_timing_options(self) -> list[str]:
        """Preserve source timing unless an output frame rate is explicit."""
        settings = self.encoding.settings
        fps = settings.clean_fps
        if fps is None:
            return [
                '-fps_mode', 'passthrough',
                '-force_key_frames',
                f'expr:gte(t,n_forced*{settings.keyframe_interval_seconds:g})',
                '-sc_threshold', '0',
            ]
        gop_size = max(1, round(fps * settings.keyframe_interval_seconds))
        return [
            '-r',
            f"{fps:g}",
            '-g',
            str(gop_size),
            '-keyint_min',
            str(gop_size),
            '-sc_threshold',
            '0',
        ]

    def get_encoder(self) -> str:
        return self.encoding.settings.clean_encoder.lower()
