from __future__ import annotations

import time
from collections.abc import Collection

import redis.asyncio as redis

from examples.streaming_web.media_paths import build_clean_demand_key
from examples.streaming_web.media_paths import build_overlay_demand_key
from examples.streaming_web.media_paths import build_overlay_ready_key


async def active_overlay_languages(
    rds: redis.Redis,
    media_path: str,
    allowed_languages: Collection[str],
) -> set[str]:
    """Return active overlay languages with one bounded Redis ``MGET``.

    The language set is server configuration, rather than unbounded Redis key
    space.  Looking up its explicit leases avoids a ``SCAN`` on every playback
    negotiation and matches the worker's producer-demand check.
    """
    languages = tuple(allowed_languages)
    if not languages:
        return set()
    values = await rds.mget([
        build_overlay_demand_key(media_path, language)
        for language in languages
    ])
    return {
        language
        for language, value in zip(languages, values)
        if value is not None
    }


async def touch_overlay_demand(
    rds: redis.Redis,
    media_path: str,
    label_language: str,
    ttl_seconds: int,
) -> None:
    """Renew one shared overlay producer lease."""
    await rds.set(
        build_overlay_demand_key(media_path, label_language),
        str(int(time.time())).encode('ascii'),
        ex=ttl_seconds,
    )


async def touch_clean_demand(
    rds: redis.Redis,
    media_path: str,
    ttl_seconds: int,
) -> None:
    """Renew one shared clean-stream producer lease."""
    await rds.set(
        build_clean_demand_key(media_path),
        str(int(time.time())).encode('ascii'),
        ex=ttl_seconds,
    )


async def overlay_is_ready(
    rds: redis.Redis,
    overlay_media_path: str,
) -> bool:
    """Return whether a producer has an unexpired ready marker."""
    return bool(await rds.exists(build_overlay_ready_key(overlay_media_path)))
