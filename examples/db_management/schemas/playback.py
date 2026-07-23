from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator

# Profile selects whether backend detection overlays are baked into the
# rendition.  Rendition selection is deliberately handled by the endpoint:
# a single-camera session is detail, while a wall is preview.
PlaybackProfile = Literal['clean', 'overlay']
MAX_PLAYBACK_WALL_CAMERAS = 16


class PlaybackSessionRequest(BaseModel):
    """Cross-platform single-camera detail playback request."""

    site: str = Field(min_length=1)
    camera: str = Field(min_length=1)
    profile: PlaybackProfile = 'overlay'
    language: str | None = None
    transport: Literal['hls'] = 'hls'
    session_id: str | None = None


class PlaybackWallRequest(BaseModel):
    """Cross-platform responsive preview wall with optional overlays."""

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
        if cameras is None:
            return None
        if any(not camera.strip() for camera in cameras):
            raise ValueError('camera names must not be blank')
        if len(set(cameras)) != len(cameras):
            raise ValueError('camera names must be unique')
        return cameras


class PlaybackRenewRequest(BaseModel):
    """Renew signed playback URLs by public media session id."""

    id: str = Field(min_length=1)


__all__ = [
    'MAX_PLAYBACK_WALL_CAMERAS',
    'PlaybackProfile',
    'PlaybackRenewRequest',
    'PlaybackSessionRequest',
    'PlaybackWallRequest',
]
