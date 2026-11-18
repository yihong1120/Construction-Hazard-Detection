"""A command-construction failure must not leak a host-wide NVENC lease."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from src.gpu_resources import try_acquire_resource
from src.media_restreamer import MediaSourceRestreamer
from src.media_stream_publisher import MediaStreamPublisher


def test_publisher_command_failure_returns_reservation(monkeypatch):
    monkeypatch.setenv('MEDIA_NVENC_MAX_SESSIONS', '1')
    with patch(
        'src.media_stream_publisher._select_encoder',
        return_value='h264_nvenc',
    ), patch(
        'src.media_stream_publisher.shutil.which', return_value='/bin/ffmpeg',
    ):
        publisher = MediaStreamPublisher('rtsp://127.0.0.1:1/fixture')
        with patch.object(
            publisher, '_build_ffmpeg_command',
            side_effect=RuntimeError('invalid command'),
        ):
            with pytest.raises(RuntimeError, match='invalid command'):
                asyncio.run(publisher._start(640, 352))
        lease = try_acquire_resource('nvenc', 1)
        assert lease is not None
        lease.close()


def test_restreamer_command_failure_returns_reservation(monkeypatch):
    monkeypatch.setenv('MEDIA_NVENC_MAX_SESSIONS', '1')
    restreamer = MediaSourceRestreamer(
        'rtsp://127.0.0.1:1/source', 'rtsp://127.0.0.1:1/output',
    )
    restreamer.commands = MagicMock()
    restreamer.commands.get_encoder.return_value = 'h264_nvenc'
    restreamer.commands.build_command.side_effect = RuntimeError(
        'invalid command',
    )
    with pytest.raises(RuntimeError, match='invalid command'):
        asyncio.run(restreamer._monitor_loop())
    lease = try_acquire_resource('nvenc', 1)
    assert lease is not None
    lease.close()
