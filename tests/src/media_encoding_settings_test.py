from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from src.media_publish_config import MediaPublisherFactory
from src.media_restreamer import SourceCommandBuilder
from src.settings.media import MediaPublishSettings

pytestmark = pytest.mark.ml


def test_encoders_share_captured_settings_across_fps_restarts(monkeypatch):
    for key, value in {
        'MEDIA_PUBLISH_FPS': 'auto', 'MEDIA_PREVIEW_FPS': 'auto',
        'MEDIA_PUBLISH_CLEAN_FPS': 'auto',
        'MEDIA_PUBLISH_PRESET': 'veryfast', 'MEDIA_PUBLISH_CRF': '28',
        'MEDIA_PUBLISH_KEYFRAME_INTERVAL_SECONDS': '2',
        'MEDIA_PUBLISH_RTP_PACKET_SIZE': '1200',
    }.items():
        monkeypatch.setenv(key, value)
    source = SimpleNamespace(source_fps=25)
    factory = MediaPublisherFactory(MediaPublishSettings(), source)
    detail = factory.create('rtsp://media/detail', rendition='detail')
    preview = factory.create('rtsp://media/preview', rendition='preview')
    restream = SourceCommandBuilder(factory.encoding)
    assert detail.encoding is preview.encoding is restream.encoding
    monkeypatch.setenv('MEDIA_PUBLISH_PRESET', 'slow')
    monkeypatch.setenv('MEDIA_PUBLISH_CRF', '1')
    monkeypatch.setenv('MEDIA_PUBLISH_CLEAN_FPS', '3')
    with patch(
        'src.media_encoding.EncoderSettings',
        side_effect=AssertionError('settings reloaded'),
    ):
        command = detail._build_ffmpeg_command(
            '/bin/ffmpeg', 640, 360, encoder='libx264',
        )
        assert command[command.index('-preset') + 1] == 'veryfast'
        assert command[command.index('-crf') + 1] == '28'
        assert command[command.index('-g') + 1] == '50'
        assert '-r' not in restream.build_timing_options()
        detail.close = AsyncMock()
        source.source_fps = 30
        replacement = asyncio.run(
            factory.ensure(
                'rtsp://media/detail', rendition='detail', current=detail,
            ),
        )
        assert replacement.fps == 30
        assert replacement.encoding is factory.encoding
        detail.close.assert_awaited_once()
