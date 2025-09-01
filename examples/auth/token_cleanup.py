from __future__ import annotations

import time

import jwt
from redis.asyncio import Redis

from examples.auth.cache import get_user_data
from examples.auth.cache import set_user_data
from examples.auth.config import Settings


settings = Settings()


async def prune_user_cache(
    redis_pool: Redis,
    username: str,
) -> dict[str, object] | None:
    """
    Prune a user's cached authentication data in Redis.

    Args:
        redis_pool: Asynchronous Redis client/connection.
        username: Username whose cache entry should be pruned.

    Returns:
        The updated cache dictionary if present, otherwise ``None`` when no
        cache entry exists.
    """
    cache: dict[str, object] | None = await get_user_data(redis_pool, username)
    if not cache:
        return None

    changed: bool = False
    now: int = int(time.time())

    # Refresh tokens pruning
    # The cache may contain a list of refresh tokens; keep those that are
    # valid according to ``jwt.decode`` and drop expired/invalid ones.
    refresh_raw = cache.get('refresh_tokens', [])
    refresh_tokens: list[str]
    if isinstance(refresh_raw, list):
        # Ensure a typed list of strings only
        refresh_tokens = [t for t in refresh_raw if isinstance(t, str)]
    else:
        refresh_tokens = []
    new_refresh_tokens: list[str] = []
    for tok in refresh_tokens:
        try:
            # Decode to verify validity and expiry
            jwt.decode(
                tok,
                settings.authjwt_secret_key,
                algorithms=[settings.ALGORITHM],
            )
            # Keep token only if decode succeeds
            new_refresh_tokens.append(tok)
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
            changed = True
            # Drop expired/invalid tokens silently
            continue
    if new_refresh_tokens != refresh_tokens:
        cache['refresh_tokens'] = new_refresh_tokens

    # JTI metadata pruning
    # jti_list holds active JWT IDs; jti_meta maps JTI -> expiry timestamp.
    jti_list_raw = cache.get('jti_list', [])
    jti_list: list[str]
    if isinstance(jti_list_raw, list):
        jti_list = [j for j in jti_list_raw if isinstance(j, str)]
    else:
        jti_list = []

    jti_meta_raw = cache.get('jti_meta', {})
    jti_meta: dict[str, int]
    if isinstance(jti_meta_raw, dict):
        # Build a strictly typed mapping of str -> int
        jti_meta = {
            k: int(v)
            for k, v in jti_meta_raw.items()
            if isinstance(k, str) and isinstance(v, int)
        }
    else:
        jti_meta = {}
    if jti_meta:
        new_jti_list: list[str] = []
        for j in jti_list:
            exp_ts: int = int(jti_meta.get(j, 0))
            # Keep if no expiry is tracked (0) or it is still in the future
            if exp_ts == 0 or exp_ts > now:
                new_jti_list.append(j)
            else:
                changed = True

        # Remove stale jti_meta entries not in list or already expired
        new_jti_meta: dict[str, int] = {}
        for j, exp in jti_meta.items():
            if j in new_jti_list and exp > now:
                new_jti_meta[j] = int(exp)
            else:
                changed = True

        if new_jti_list != jti_list:
            cache['jti_list'] = new_jti_list
        cache['jti_meta'] = new_jti_meta

    # Persist only if an actual change occurred
    if changed:
        await set_user_data(redis_pool, username, cache)

    return cache
