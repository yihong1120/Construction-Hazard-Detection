from __future__ import annotations

import base64
import os
from urllib.parse import quote


CLEAN_DEMAND_PREFIX = 'media_clean_demand'
OVERLAY_DEMAND_PREFIX = 'media_overlay_demand'
OVERLAY_READY_PREFIX = 'media_overlay_ready'
PREVIEW_MEDIA_PATH_SUFFIX = '_preview'


def encode_media_segment(value: str) -> str:
    """Encode one path segment for media-server-safe stream names."""
    encoded = base64.urlsafe_b64encode(value.encode('utf-8')).decode('ascii')
    return encoded.rstrip('=')


def decode_media_segment(value: str) -> str:
    """Decode one media-server-safe path segment."""
    padded = value + ('=' * (-len(value) % 4))
    return base64.urlsafe_b64decode(padded.encode('ascii')).decode('utf-8')


def build_media_path(site: str, stream_name: str) -> str:
    """Build a stable MediaMTX path for a site camera stream."""
    return (
        'hazard_'
        f'{encode_media_segment(site)}_'
        f'{encode_media_segment(stream_name)}'
    )


def build_annotated_media_path(
    path: str,
    label_language: str = 'zh-TW',
) -> str:
    """Build the MediaMTX path used for a pre-rendered overlay stream."""
    return f'{path}_annotated_{encode_media_segment(label_language)}'


def build_preview_media_path(path: str) -> str:
    """Build the MediaMTX path for the low-bitrate preview rendition."""
    return f'{path}{PREVIEW_MEDIA_PATH_SUFFIX}'


def parse_annotated_media_path(path: str) -> tuple[str, str] | None:
    """Return base media path and label language for an overlay stream path."""
    marker = '_annotated_'
    if marker not in path:
        return None
    base_path, encoded_language = path.rsplit(marker, 1)
    if not base_path.startswith('hazard_') or not encoded_language:
        return None
    try:
        return base_path, decode_media_segment(encoded_language)
    except Exception:
        return None


def build_overlay_demand_key(media_path: str, label_language: str) -> str:
    """Build the Redis key used to keep a shared overlay stream requested."""
    return (
        f'{OVERLAY_DEMAND_PREFIX}:'
        f'{media_path}:'
        f'{encode_media_segment(label_language)}'
    )


def build_clean_demand_key(media_path: str) -> str:
    """Build the Redis key used to keep a clean stream requested."""
    return f'{CLEAN_DEMAND_PREFIX}:{media_path}'


def build_overlay_ready_key(overlay_media_path: str) -> str:
    """Build the Redis key used to mark an overlay path as ready."""
    return f'{OVERLAY_READY_PREFIX}:{overlay_media_path}'


def build_media_hls_url(path: str, base_url: str | None = None) -> str:
    """Build the public HLS playback URL for a media-server path."""
    base = (
        base_url
        or os.getenv('MEDIA_PUBLIC_HLS_BASE_URL', '')
        or '/hazard/media'
    ).rstrip('/')
    return f'{base}/{quote(path, safe="")}/index.m3u8'


def build_media_webrtc_url(path: str, base_url: str | None = None) -> str:
    """Build the public WebRTC/WHEP playback URL for a media-server path."""
    base = (
        base_url
        or os.getenv('MEDIA_PUBLIC_WEBRTC_BASE_URL', '')
        or '/hazard/media/webrtc'
    ).rstrip('/')
    return f'{base}/{quote(path, safe="")}/whep'
