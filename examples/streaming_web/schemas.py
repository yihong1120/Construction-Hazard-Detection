from __future__ import annotations

from typing import Literal
from typing import TypedDict

from pydantic import BaseModel

MAX_STREAM_PLAYBACK_BATCH_STREAMS = 24


class FrameOutData(TypedDict, total=False):
    """A compact Redis metadata record for MediaMTX live viewers."""

    key: str
    id: str
    stream_id: str
    redis_key: str
    has_warning: bool | str


class LabelListResponse(BaseModel):
    """Response model encapsulating a set of available labels."""

    labels: list[str]


class StreamPlaybackRequest(BaseModel):
    """Requested playback profile for one camera stream."""

    label: str | None = None
    stream_id: str | None = None
    key: str | None = None
    session_id: str | None = None
    profile: str = 'clean'
    rendition: Literal['detail', 'preview'] = 'detail'
    language: str | None = None
    transport: str = 'hls'


class StreamPlaybackBatchRequest(BaseModel):
    """Requested playback sessions for a site overview or explicit streams."""

    label: str | None = None
    streams: list[StreamPlaybackRequest] | None = None
    profile: str = 'overlay'
    rendition: Literal['detail', 'preview'] = 'detail'
    language: str | None = None
    transport: str = 'hls'


class OverlayLanguageInfo(BaseModel):
    """One backend-supported overlay language contract for clients."""

    code: str
    notification_code: str
    display_name: str
    native_name: str
    is_default: bool
    class_labels: dict[str, str]
    warning_labels: dict[str, str]
    notification_templates: dict[str, str]


class OverlayLanguageListResponse(BaseModel):
    """Supported overlay language codes and translation dictionaries."""

    default_language: str
    allowed_language_codes: list[str]
    supported_languages: list[str]
    aliases: dict[str, str]
    languages: list[OverlayLanguageInfo]
    stream_playback_endpoint: str
    playback_endpoint: str
    max_active_languages_per_stream: int
    demand_ttl_seconds: int
    ready_ttl_seconds: int


class FramePostResponse(BaseModel):
    """Response model representing a simple status message."""

    status: str
    message: str


class WebRTCOfferRequest(BaseModel):
    """Client SDP offer schema retained for API compatibility."""

    sdp: str
    type: str = 'offer'
    overlay: str | None = None
    lang: str | None = None
    language: str | None = None
    min_confidence: float | None = None


class WebRTCAnswerResponse(BaseModel):
    """Server SDP answer schema retained for API compatibility."""

    sdp: str
    type: str


__all__ = [
    'FrameOutData',
    'LabelListResponse',
    'MAX_STREAM_PLAYBACK_BATCH_STREAMS',
    'StreamPlaybackRequest',
    'StreamPlaybackBatchRequest',
    'OverlayLanguageInfo',
    'OverlayLanguageListResponse',
    'FramePostResponse',
    'WebRTCOfferRequest',
    'WebRTCAnswerResponse',
]
