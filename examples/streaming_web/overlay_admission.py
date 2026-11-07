from __future__ import annotations

from fastapi import HTTPException
from redis.asyncio import Redis
from redis.asyncio.client import Pipeline

from examples.auth.redis_transactions import optimistic
from examples.streaming_web import playback_config as config
from examples.streaming_web.media_paths import build_overlay_demand_key
from examples.streaming_web.media_paths import build_preview_media_path
from examples.streaming_web.media_paths import decode_media_segment
from examples.streaming_web.media_paths import OVERLAY_DEMAND_PREFIX
from examples.streaming_web.media_paths import PREVIEW_MEDIA_PATH_SUFFIX
from examples.streaming_web.playback_languages import (
    _allowed_overlay_languages,
)


class OverlayAdmission:
    """Admit language leases inside their owner's Redis transaction."""

    def __init__(self) -> None:
        self.allowed: tuple[str, ...] | None = None
        self.limit = config.OVERLAY_MAX_ACTIVE_LANGUAGES
        self.requests: dict[tuple[str, str], int] = {}

    def add(self, path: str, language: str, ttl: int) -> None:
        if self.allowed is None:
            self.allowed = _allowed_overlay_languages()
        if language not in self.allowed:
            raise HTTPException(422, 'unsupported_language')
        self.requests[path, language] = max(
            ttl,
            self.requests.get((path, language), 0),
        )

    def add_key(self, key: str, ttl: int) -> bool:
        prefix = OVERLAY_DEMAND_PREFIX + ':'
        if not key.startswith(prefix):
            return False
        path, separator, encoded = key[len(prefix):].rpartition(':')
        if not separator or not path:
            raise ValueError('Invalid overlay demand key')
        self.add(path, decode_media_segment(encoded), ttl)
        return True

    @property
    def requested_keys(self) -> set[str]:
        return {build_overlay_demand_key(*pair) for pair in self.requests}

    async def watch(self, pipe: Pipeline) -> None:

        cameras = {_camera_path(path) for path, _ in self.requests}
        pairs = [
            (path, lang)
            for base in sorted(cameras)
            for path in (base, build_preview_media_path(base))
            for lang in self.allowed or ()
        ]
        if not pairs:
            return
        keys = [build_overlay_demand_key(*pair) for pair in pairs]
        await pipe.watch(*keys)
        values = await pipe.mget(keys)
        active: dict[str, set[str]] = {base: set() for base in cameras}
        for (path, language), value in zip(pairs, values, strict=True):
            if value is not None:
                active[_camera_path(path)].add(language)
        for base in cameras:
            requested = {
                lang
                for path, lang in self.requests
                if _camera_path(path) == base
            }
            if (
                requested - active[base]
                and len(active[base] | requested) > self.limit
            ):
                raise HTTPException(429, 'overlay_language_limit_reached')

    def queue(self, pipe: Pipeline) -> None:
        for (path, language), ttl in self.requests.items():
            pipe.set(build_overlay_demand_key(path, language), b'1', ex=ttl)

    async def commit(self, redis: Redis) -> None:

        await optimistic(redis, self._reserve)

    async def _reserve(self, pipe: Pipeline) -> None:
        await self.watch(pipe)
        pipe.multi()
        self.queue(pipe)
        await pipe.execute()


def _camera_path(path: str) -> str:
    return path.removesuffix(PREVIEW_MEDIA_PATH_SUFFIX)
