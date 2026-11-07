from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from src.settings.media import MediaPublishSettings
from src.stream_runtime_state import DetectionClock
from src.stream_runtime_state import LatestDetectionState
from src.stream_runtime_state import LatestFrameState
from src.stream_runtime_state import StreamMediaSettings
from src.stream_runtime_state import StreamRuntime


class DeliveryDouble:
    def __init__(self) -> None:
        self.submit = AsyncMock(return_value='queued-event')

    async def run(self, stop: asyncio.Event) -> None:
        await stop.wait()


def make_stream_runtime(**values) -> StreamRuntime:
    return StreamRuntime(
        cfg={
            'site': values.get('site', 'Site'),
            'stream_name': values.get('stream_name', 'Cam'),
            'video_url': values.get('video_url', 'rtsp://source'),
            'work_start_hour': values.get('work_start_hour', 0),
            'work_end_hour': values.get('work_end_hour', 24),
        },
        capture=values.get('streaming_capture', AsyncMock()),
        detector=values.get('yolo_detector', AsyncMock()),
        danger=values.get('danger_detector', MagicMock()),
        redis=values.get('redis_manager', MagicMock()),
        delivery=values.get('delivery', DeliveryDouble()),
        metadata_key=values.get('metadata_key', 'metadata'),
        media=StreamMediaSettings(
            values.get('media_path', 'media'),
            MediaPublishSettings(
                publish_base=values.get(
                    'media_publish_base', 'rtsp://media-server:8554',
                ),
                clean=values.get('publish_clean_stream', False),
                overlay=values.get('publish_overlay_streams', False),
                restream_clean=values.get('restream_clean_source', False),
            ),
        ),
        latest_frame=values.get('latest_frame', LatestFrameState()),
        latest_detection=values.get(
            'latest_detection',
            LatestDetectionState(),
        ),
        stop=values.get('stop_event', asyncio.Event()),
        clock=DetectionClock(
            values.get('last_notification_time', 0),
            values.get('last_warning_event_time'),
            values.get('warning_event_throttle_seconds', 30),
        ),
    )


def make_media_group(**values):
    from src.stream_media import StreamPublisherGroup
    from src.stream_runtime_state import OverlayPublisherVariant

    values['video_url'] = values.get(
        'source_url',
        values.get('video_url', 'rtsp://source'),
    )
    values['restream_clean_source'] = values.get(
        'use_source_restreamer',
        False,
    )
    if values.get('rendition') == 'preview':
        values['media_path'] = values['media_path'].removesuffix('_preview')
    group = StreamPublisherGroup(make_stream_runtime(**values))
    if values.get('demand_cache') is not None:
        group.demand_cache = values['demand_cache']
    if 'variants' in values:
        group.variants = values['variants']
    if values.get('source_reconnect_event') is not None:
        group.clean_reconnect_event = values['source_reconnect_event']
    group.test_variant = OverlayPublisherVariant(
        values.get('media_path', 'media'),
        values.get('rendition', 'detail'),
        values.get('overlay_media_publishers', {}),
        values.get(
            'overlay_ready_started_at',
            {
                values.get('label_language', 'en'): -float('inf'),
            },
        ),
    )
    return group


def make_overlay_cache(frames):
    from src.stream_overlay_frames import OverlayFrameCache
    from tests.overlay_fixtures_test import make_overlay_renderer

    cache = OverlayFrameCache(make_overlay_renderer())
    cache.frames = frames
    return cache
