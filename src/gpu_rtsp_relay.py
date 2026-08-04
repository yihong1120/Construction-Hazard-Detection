from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
from collections.abc import Awaitable
from typing import Protocol


_default_timeout_us = 5_000_000


class _FfmpegProcess(Protocol):
    """Minimal ffmpeg process interface used by the relay."""

    @property
    def returncode(self) -> int | None:
        """Return the process exit status when available."""

    def terminate(self) -> None:
        """Request graceful process termination."""

    def kill(self) -> None:
        """Force process termination."""

    def wait(self) -> Awaitable[int | None]:
        """Wait for process completion."""


class GpuRtspRelay:
    """Publish a TCP RTSP source locally without transcoding its video."""

    def __init__(self, source_url: str) -> None:
        """Create a private MediaMTX path for one camera source."""
        self.source_url = source_url
        self.publish_url = _build_publish_url(source_url)
        self._process: _FfmpegProcess | None = None

    @property
    def is_running(self) -> bool:
        """Return whether the copy relay process is still alive."""
        return self._process is not None and self._process.returncode is None

    async def start(self) -> None:
        """Start ffmpeg once; the caller owns retries and lifecycle."""
        if self.is_running:
            return
        self._process = await asyncio.create_subprocess_exec(
            *_build_command(
                _find_ffmpeg(),
                self.source_url,
                self.publish_url,
            ),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

    async def close(self) -> None:
        """Stop the local ffmpeg relay promptly."""
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


def _build_publish_url(source_url: str) -> str:
    """Build a stable local path without exposing source credentials."""
    base_url = (
        os.getenv('GPU_DECODE_RELAY_RTSP_BASE_URL')
        or os.getenv('MEDIA_PUBLISH_RTSP_BASE_URL')
        or 'rtsp://127.0.0.1:8554'
    ).rstrip('/')
    source_digest = hashlib.blake2b(
        source_url.encode(),
        digest_size=10,
    ).hexdigest()
    return f'{base_url}/gpu-decode-{source_digest}'


def _find_ffmpeg() -> str:
    """Resolve the local ffmpeg executable used for RTSP relaying."""
    ffmpeg_binary = os.getenv('MEDIA_FFMPEG_PATH', '').strip()
    if ffmpeg_binary:
        return ffmpeg_binary
    ffmpeg_binary = shutil.which('ffmpeg') or ''
    if not ffmpeg_binary:
        raise RuntimeError('GPU RTSP relay requires ffmpeg')
    return ffmpeg_binary


def _build_command(
    ffmpeg_binary: str,
    source_url: str,
    publish_url: str,
) -> list[str]:
    """Build a low-latency TCP-to-local-RTSP copy command."""
    command = [
        ffmpeg_binary,
        '-nostdin',
        '-hide_banner',
        '-loglevel',
        os.getenv('GPU_DECODE_RELAY_LOGLEVEL', 'error'),
        '-fflags',
        '+genpts+nobuffer',
        '-flags',
        'low_delay',
    ]
    if source_url.lower().startswith('rtsp://'):
        command.extend([
            '-rtsp_transport',
            'tcp',
            '-timeout',
            str(_input_timeout_us()),
        ])
    command.extend([
        '-i',
        source_url,
        '-map',
        '0:v:0',
        '-an',
        '-c:v',
        'copy',
        '-f',
        'rtsp',
        '-rtsp_transport',
        'tcp',
        publish_url,
    ])
    return command


def _input_timeout_us() -> int:
    """Return a positive RTSP connection timeout in microseconds."""
    try:
        return max(
            1,
            int(
                os.getenv(
                    'GPU_DECODE_RELAY_TIMEOUT_US',
                    str(_default_timeout_us),
                ),
            ),
        )
    except ValueError:
        return _default_timeout_us
