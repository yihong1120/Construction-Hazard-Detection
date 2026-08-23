from __future__ import annotations

import base64
import os
from urllib.parse import quote

CLEAN_DEMAND_PREFIX = 'media_clean_demand'
OVERLAY_DEMAND_PREFIX = 'media_overlay_demand'
OVERLAY_READY_PREFIX = 'media_overlay_ready'
PREVIEW_MEDIA_PATH_SUFFIX = '_preview'


def encode_media_segment(value: str) -> str:
    """Encode a logical name as one MediaMTX-safe path segment.

    Args:
        value: Unencoded site, stream, or language name.

    Returns:
        URL-safe Base64 text without optional padding.
    """
    # Padding is redundant inside a delimiter-separated path and is restored
    # by the decoder when the segment is read back.
    encoded = base64.urlsafe_b64encode(value.encode('utf-8')).decode('ascii')
    return encoded.rstrip('=')


def decode_media_segment(value: str) -> str:
    """Decode a MediaMTX-safe path segment into its original name.

    Args:
        value: URL-safe Base64 text with or without trailing padding.

    Returns:
        The decoded UTF-8 logical name.
    """
    padded = value + ('=' * (-len(value) % 4))
    return base64.urlsafe_b64decode(padded.encode('ascii')).decode('utf-8')


def build_media_path(site: str, stream_name: str) -> str:
    """Build the stable MediaMTX path for one site camera stream.

    Args:
        site: Site label that owns the stream.
        stream_name: Configured camera stream name.

    Returns:
        Delimiter-safe MediaMTX path for the detail rendition.
    """
    return (
        'hazard_'
        f"{encode_media_segment(site)}_"
        f"{encode_media_segment(stream_name)}"
    )


def build_annotated_media_path(
    path: str,
    label_language: str = 'zh-TW',
) -> str:
    """Build the MediaMTX path for a pre-rendered overlay stream.

    Args:
        path: Base clean-stream media path.
        label_language: Canonical label language for the overlay.

    Returns:
        Language-specific annotated media path.
    """
    return f"{path}_annotated_{encode_media_segment(label_language)}"


def build_preview_media_path(path: str) -> str:
    """Build the MediaMTX path for the preview rendition.

    Args:
        path: Detail-rendition media path.

    Returns:
        Media path for the low-bitrate preview rendition.
    """
    return f"{path}{PREVIEW_MEDIA_PATH_SUFFIX}"


def parse_annotated_media_path(path: str) -> tuple[str, str] | None:
    """Split a valid annotated path into its base path and language.

    Args:
        path: Candidate MediaMTX stream path.

    Returns:
        The base path and decoded label language, or ``None`` when the path
        does not use the annotated-stream contract.
    """
    marker = '_annotated_'
    if marker not in path:
        return None
    base_path, encoded_language = path.rsplit(marker, 1)
    if not base_path.startswith('hazard_') or not encoded_language:
        return None
    return base_path, decode_media_segment(encoded_language)


def build_overlay_demand_key(media_path: str, label_language: str) -> str:
    """Build the Redis key that leases a shared overlay stream.

    Args:
        media_path: Base clean-stream media path.
        label_language: Canonical overlay label language.

    Returns:
        Redis demand key scoped to a stream and language.
    """
    return (
        f"{OVERLAY_DEMAND_PREFIX}:"
        f"{media_path}:"
        f"{encode_media_segment(label_language)}"
    )


def build_clean_demand_key(media_path: str) -> str:
    """Build the Redis key that leases a shared clean stream.

    Args:
        media_path: Clean-stream MediaMTX path.

    Returns:
        Redis demand key for the stream.
    """
    return f"{CLEAN_DEMAND_PREFIX}:{media_path}"


def build_overlay_ready_key(overlay_media_path: str) -> str:
    """Build the Redis key recording an available overlay publisher.

    Args:
        overlay_media_path: Language-specific annotated media path.

    Returns:
        Redis ready-state key for the overlay path.
    """
    return f"{OVERLAY_READY_PREFIX}:{overlay_media_path}"


def build_media_hls_url(path: str, base_url: str | None = None) -> str:
    """Build the public HLS URL for a media-server path.

    Args:
        path: MediaMTX path to expose.
        base_url: Optional public proxy base that overrides configuration.

    Returns:
        Escaped HLS playlist URL.
    """
    base = (
        base_url
        or os.getenv('MEDIA_PUBLIC_HLS_BASE_URL', '')
        or '/hazard/media'
    ).rstrip('/')
    return f"{base}/{quote(path, safe='')}/index.m3u8"


def build_media_webrtc_url(path: str, base_url: str | None = None) -> str:
    """Build the public WebRTC WHEP URL for a media-server path.

    Args:
        path: MediaMTX path to expose.
        base_url: Optional public proxy base that overrides configuration.

    Returns:
        Escaped WHEP endpoint URL.
    """
    base = (
        base_url
        or os.getenv('MEDIA_PUBLIC_WEBRTC_BASE_URL', '')
        or '/hazard/media/webrtc'
    ).rstrip('/')
    return f"{base}/{quote(path, safe='')}/whep"
