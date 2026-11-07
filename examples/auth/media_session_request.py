from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field


@dataclass(frozen=True)
class MediaSessionOwner:
    username: str
    user_id: int
    parent: str
    platform: str
    generation: int | None = field(default=None, kw_only=True)


@dataclass(frozen=True, kw_only=True)
class MediaSessionRequest:
    owner: MediaSessionOwner
    site: str
    cameras: Sequence[str]
    profile: str
    batch: bool = False
    language: str | None = None
    quality: str | None = None
    purpose: str | None = None
    demand_keys: Sequence[str] | None = None
    playback_sessions: Mapping[str, Mapping[str, object]] | None = None
