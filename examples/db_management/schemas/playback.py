from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

# Profile selects whether backend detection overlays are baked into the
# rendition. The endpoint selects rendition because a camera session is detail
# while a wall is preview.
PlaybackProfile = Literal['clean', 'overlay']
# Bound wall requests before they reach the upstream streaming service.
MAX_PLAYBACK_WALL_CAMERAS = 16


class PlaybackSessionRequest(BaseModel):
    """Define a cross-platform request for one camera's detail playback.

    Attributes:
        site: Site identifier or name containing the camera.
        camera: Camera identifier or name to play.
        profile: Whether the rendition includes detection overlays.
        language: Optional language preference for playback metadata.
        transport: Streaming transport supported by this endpoint.
        session_id: Optional existing media session to renew or reuse.
    """

    site: str = Field(min_length=1)
    camera: str = Field(min_length=1)
    profile: PlaybackProfile = 'overlay'
    language: str | None = None
    transport: Literal['hls'] = 'hls'
    session_id: str | None = None


class PlaybackWallRequest(BaseModel):
    """Define a responsive multi-camera preview-wall request.

    Attributes:
        site: Site identifier or name containing the cameras.
        cameras: Optional unique camera selection; omitting it uses the site's
            default wall selection.
        profile: Whether renditions include detection overlays.
        language: Optional language preference for playback metadata.
        transport: Streaming transport supported by this endpoint.
    """

    site: str = Field(min_length=1)
    cameras: list[str] | None = Field(
        default=None,
        max_length=MAX_PLAYBACK_WALL_CAMERAS,
    )
    profile: PlaybackProfile = 'overlay'
    language: str | None = None
    transport: Literal['hls'] = 'hls'

    @field_validator('cameras')
    @classmethod
    def validate_cameras(
        cls,
        cameras: list[str] | None,
    ) -> list[str] | None:
        """Validate an optional unique, non-blank camera selection.

        Args:
            cameras: Requested camera names, if the wall is explicitly scoped.

        Returns:
            The validated camera names, or ``None`` for the site default.

        Raises:
            ValueError: If a name is blank or a camera appears more than once.
        """
        if cameras is None:
            return None
        if any(not camera.strip() for camera in cameras):
            raise ValueError('camera names must not be blank')
        if len(set(cameras)) != len(cameras):
            raise ValueError('camera names must be unique')
        return cameras


class PlaybackRenewRequest(BaseModel):
    """Define a request to renew signed playback URLs.

    Attributes:
        id: Public identifier of the media session to renew.
    """

    id: str = Field(min_length=1)


class StreamingPlaybackItem(BaseModel):
    """Represent one validated playback item from the streaming service.

    Attributes:
        session_id: Upstream session identifier for the media stream.
        key: Stable item key used in the client response.
        label: Human-readable camera label.
        profile: Whether the rendition includes detection overlays.
        rendition: Detail or preview rendition selected for the item.
        language: Optional language used for item metadata.
    """

    model_config = ConfigDict(extra='allow', strict=True)

    session_id: str = Field(min_length=1)
    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    profile: PlaybackProfile
    rendition: Literal['detail', 'preview']
    language: str | None


class StreamingPlaybackBatchResponse(BaseModel):
    """Represent a validated multi-item streaming-service response.

    Attributes:
        items: Playback items returned by the upstream service.
        max_streams: Upstream concurrency limit for the response.
    """

    model_config = ConfigDict(extra='allow', strict=True)

    items: list[StreamingPlaybackItem]
    max_streams: int


class StreamingPlaybackErrorResponse(BaseModel):
    """Represent structured failure detail from the streaming service.

    Attributes:
        detail: Upstream error payload preserved for error translation.
    """

    model_config = ConfigDict(extra='allow', strict=True)

    detail: object


__all__ = [
    'MAX_PLAYBACK_WALL_CAMERAS',
    'PlaybackProfile',
    'PlaybackRenewRequest',
    'PlaybackSessionRequest',
    'PlaybackWallRequest',
    'StreamingPlaybackBatchResponse',
    'StreamingPlaybackErrorResponse',
    'StreamingPlaybackItem',
]
