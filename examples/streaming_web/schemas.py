from __future__ import annotations

from typing import Literal
from typing import NotRequired
from typing import TypeAlias
from typing import TypedDict

from pydantic import BaseModel

MAX_STREAM_PLAYBACK_BATCH_STREAMS = 24


class FrameOutData(TypedDict):
    """Represent a compact Redis metadata record for live viewers.

    Attributes:
        has_warning: Whether the associated frame contains a safety warning.
        key: Decoded configured stream name.
        id: Redis Stream message identifier.
        stream_id: Encoded configured stream identifier.
        redis_key: Canonical Redis metadata key.
    """

    has_warning: bool
    key: str
    id: str
    stream_id: str
    redis_key: str


PlaybackProfile: TypeAlias = Literal['clean', 'overlay']
PlaybackRendition: TypeAlias = Literal['detail', 'preview']


class CleanPlaybackSession(TypedDict):
    """Represent Redis-backed state for a clean-stream playback session.

    Attributes:
        session_id: Opaque browser-facing session identifier.
        username: Authenticated owner of the session.
        label: Site label containing the stream.
        stream_name: Decoded configured stream name.
        stream_id: Encoded configured stream identifier.
        profile: Constant clean-stream profile discriminator.
        rendition: Detail or preview rendition selection.
        language: Always ``None`` for clean playback.
        base_media_path: Selected clean MediaMTX path.
        overlay_media_path: Always ``None`` for clean playback.
        created_at: UTC ISO-8601 creation timestamp.
        expires_at: UTC ISO-8601 expiry timestamp.
    """

    session_id: str
    username: str
    label: str
    stream_name: str
    stream_id: str
    profile: Literal['clean']
    rendition: PlaybackRendition
    language: None
    base_media_path: str
    overlay_media_path: None
    created_at: str
    expires_at: str


class OverlayPlaybackSession(TypedDict):
    """Represent Redis-backed state for an annotated-stream session.

    Attributes:
        session_id: Opaque browser-facing session identifier.
        username: Authenticated owner of the session.
        label: Site label containing the stream.
        stream_name: Decoded configured stream name.
        stream_id: Encoded configured stream identifier.
        profile: Constant overlay profile discriminator.
        rendition: Detail or preview rendition selection.
        language: Canonical language rendered into labels.
        base_media_path: Selected clean MediaMTX path.
        overlay_media_path: Language-specific annotated media path.
        created_at: UTC ISO-8601 creation timestamp.
        expires_at: UTC ISO-8601 expiry timestamp.
    """

    session_id: str
    username: str
    label: str
    stream_name: str
    stream_id: str
    profile: Literal['overlay']
    rendition: PlaybackRendition
    language: str
    base_media_path: str
    overlay_media_path: str
    created_at: str
    expires_at: str


# The profile discriminator lets services access required media fields without
# runtime shape checks after trusted Redis session payloads are decoded.
PlaybackSession: TypeAlias = CleanPlaybackSession | OverlayPlaybackSession


class PlaybackSessionState(TypedDict):
    """Represent resolved HLS availability for a playback session.

    Attributes:
        status: Client-visible readiness state.
        state: Backwards-compatible readiness state.
        overlay_ready: Whether an overlay publisher is available.
        media_path: Selected MediaMTX path.
        hls_url: Direct internal HLS playlist URL.
    """

    status: Literal['ready', 'starting']
    state: Literal['ready', 'starting']
    overlay_ready: bool
    media_path: str
    hls_url: str


class PlaybackSessionResponse(TypedDict):
    """Represent the public playback-session response sent to clients.

    Attributes:
        session_id: Opaque browser-facing session identifier.
        stream_id: Encoded configured stream identifier.
        key: Decoded configured stream name.
        label: Site label containing the stream.
        transport: Fixed HLS transport identifier.
        status: Client-visible playback readiness.
        state: Backwards-compatible playback readiness.
        profile: Selected clean or overlay profile.
        rendition: Selected detail or preview rendition.
        playback_ready: Whether a stable playback endpoint is available.
        playback_url: Authorised stable playlist URL.
        media_hls_url: Current direct media playlist URL.
        language: Overlay language, if applicable.
        overlay_ready: Whether the overlay producer is ready.
        media_path: Current MediaMTX path.
        expires_at: UTC ISO-8601 session expiry timestamp.
        expires_in: Remaining configured session lifetime in seconds.
        demand_ttl_seconds: Redis demand-lease duration in seconds.
        webrtc_url: Optional direct WebRTC WHEP endpoint.
    """

    session_id: str
    stream_id: str
    key: str
    label: str
    transport: Literal['hls']
    status: Literal['ready', 'starting']
    state: Literal['ready', 'starting']
    profile: PlaybackProfile
    rendition: PlaybackRendition
    playback_ready: bool
    playback_url: str
    media_hls_url: str
    language: str | None
    overlay_ready: bool
    media_path: str
    expires_at: str
    expires_in: int
    demand_ttl_seconds: int
    webrtc_url: NotRequired[str]


class LabelListResponse(BaseModel):
    """Validate the site labels available to the authenticated user.

    Attributes:
        labels: Sorted site labels visible to the caller.
    """

    labels: list[str]


class StreamPlaybackRequest(BaseModel):
    """Validate a request for one stream playback session.

    Attributes:
        label: Optional site label containing the stream.
        stream_id: Optional encoded configured stream identifier.
        key: Optional decoded configured stream name.
        session_id: Optional existing session to update.
        profile: Requested clean or overlay profile.
        rendition: Requested detail or preview rendition.
        language: Optional overlay label language.
        transport: Requested playback transport.
    """

    label: str | None = None
    stream_id: str | None = None
    key: str | None = None
    session_id: str | None = None
    profile: str = 'clean'
    rendition: Literal['detail', 'preview'] = 'detail'
    language: str | None = None
    transport: str = 'hls'


class StreamPlaybackBatchRequest(BaseModel):
    """Validate playback sessions for a site overview or stream list.

    Attributes:
        label: Optional site label applying to the whole request.
        streams: Optional explicit per-stream requests.
        profile: Default clean or overlay profile for the batch.
        rendition: Default detail or preview rendition for the batch.
        language: Default overlay label language for the batch.
        transport: Default playback transport for the batch.
    """

    label: str | None = None
    streams: list[StreamPlaybackRequest] | None = None
    profile: str = 'overlay'
    rendition: Literal['detail', 'preview'] = 'detail'
    language: str | None = None
    transport: str = 'hls'


class OverlayLanguageInfo(BaseModel):
    """Describe one backend-supported overlay language for clients.

    Attributes:
        code: Canonical language code used by the backend.
        notification_code: Language code used by notifications.
        display_name: English display name.
        native_name: Language name in its own writing system.
        is_default: Whether the language is selected by default.
        class_labels: Localised detection-class labels.
        warning_labels: Localised warning labels.
        notification_templates: Localised notification templates.
    """

    code: str
    notification_code: str
    display_name: str
    native_name: str
    is_default: bool
    class_labels: dict[str, str]
    warning_labels: dict[str, str]
    notification_templates: dict[str, str]


class OverlayLanguageListResponse(BaseModel):
    """Validate the full overlay-language capability response.

    Attributes:
        default_language: Canonical default language code.
        allowed_language_codes: Canonical codes callers may request.
        supported_languages: Alias of canonical supported codes.
        aliases: Mapping of recognised aliases to canonical codes.
        languages: Per-language display and translation contracts.
        stream_playback_endpoint: Relative endpoint for playback requests.
        playback_endpoint: Backwards-compatible playback endpoint alias.
        max_active_languages_per_stream: Maximum concurrent overlay languages.
        demand_ttl_seconds: Overlay demand-lease duration.
        ready_ttl_seconds: Overlay readiness-marker duration.
    """

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
    """Validate a simple status response from a frame operation.

    Attributes:
        status: Machine-readable operation status.
        message: Human-readable operation result.
    """

    status: str
    message: str


__all__ = [
    'FrameOutData',
    'CleanPlaybackSession',
    'LabelListResponse',
    'MAX_STREAM_PLAYBACK_BATCH_STREAMS',
    'StreamPlaybackRequest',
    'StreamPlaybackBatchRequest',
    'OverlayLanguageInfo',
    'OverlayLanguageListResponse',
    'OverlayPlaybackSession',
    'PlaybackProfile',
    'PlaybackRendition',
    'PlaybackSession',
    'PlaybackSessionResponse',
    'PlaybackSessionState',
    'FramePostResponse',
]
