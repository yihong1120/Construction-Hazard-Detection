from __future__ import annotations

import os
from typing import Any
from typing import TypedDict

from src.media_stream_publisher import MediaStreamPublisher


class PreviewPublisherOptions(TypedDict):
    """Bounded settings for a low-bandwidth media rendition."""

    fps: float
    width: int
    height: int
    bitrate: str
    maxrate: str
    bufsize: str


def env_enabled(name: str, default: bool) -> bool:
    """Read a boolean environment setting with a predictable default."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {'1', 'true', 'yes', 'on'}


def preview_publisher_options() -> PreviewPublisherOptions:
    """Return the lower-bandwidth encoder budget used by camera walls."""
    return {
        'fps': max(
            1.0,
            float(
                os.getenv(
                    'MEDIA_PREVIEW_FPS',
                    os.getenv('MEDIA_PUBLISH_FPS', '15'),
                ),
            ),
        ),
        'width': max(2, int(os.getenv('MEDIA_PREVIEW_WIDTH', '640'))),
        'height': max(2, int(os.getenv('MEDIA_PREVIEW_HEIGHT', '360'))),
        'bitrate': os.getenv('MEDIA_PREVIEW_BITRATE', '500k'),
        'maxrate': os.getenv('MEDIA_PREVIEW_MAXRATE', '700k'),
        'bufsize': os.getenv('MEDIA_PREVIEW_BUFSIZE', '1400k'),
    }


def create_media_publisher(
    publish_url: str,
    *,
    rendition: str,
    publisher_type: Any = MediaStreamPublisher,
) -> MediaStreamPublisher:
    """Create a detail or preview publisher with the correct encoder budget."""
    if rendition == 'preview':
        return publisher_type(
            publish_url=publish_url,
            **preview_publisher_options(),
        )
    if rendition == 'detail':
        return publisher_type(publish_url=publish_url)
    raise ValueError(f"unsupported media rendition: {rendition}")
