from __future__ import annotations

import time
from typing import cast

from jwt.exceptions import InvalidTokenError
from redis.asyncio import Redis

from examples.auth.cache import get_user_data
from examples.auth.cache import set_user_data
from examples.auth.jwt_config import jwt_refresh
from examples.db_management.schemas.auth import UserCache


def _refresh_tokens(cache: UserCache) -> list[str]:
    """Return the cache's canonical refresh-token list."""
    return cache['refresh_tokens']


def _prune_refresh_tokens(cache: UserCache) -> tuple[list[str], bool]:
    """
    Return valid refresh tokens and whether the list changed.


    Args:
        cache: The user cache dictionary.

    Returns:
    A tuple of (new_tokens, changed) where new_tokens is the list of
    valid refresh tokens and changed is a boolean indicating if the
    list was modified.
    """
    tokens = _refresh_tokens(cache)
    if not tokens:
        # Nothing to validate
        return tokens, False

    new_tokens: list[str] = []
    changed = False
    for tok in tokens:
        try:
            # Refresh tokens carry a refresh-only audience.  Decoding with
            # bare PyJWT would reject that audience and incorrectly delete a
            # valid session from Redis.
            jwt_refresh.decode_token(tok)
            new_tokens.append(tok)
        except InvalidTokenError:
            changed = True
            continue
    return new_tokens, changed or (new_tokens != tokens)


def _prune_jti(
    cache: UserCache,
    now: int,
) -> tuple[list[str], dict[str, int], bool]:
    """
    Return filtered (jti_list, jti_meta, changed).

    Args:
        cache: The user cache dictionary.
        now: Current timestamp as an integer.

    Returns:
        A tuple of (new_jti_list, new_jti_meta, changed) where
        new_jti_list is the filtered list of JTI strings, new_jti_meta is
        the filtered JTI metadata dictionary, and changed is a boolean
        indicating if any changes were made.
    """
    jti_list = cache['jti_list']
    jti_meta = cache['jti_meta']
    if not jti_meta and not jti_list:
        return jti_list, jti_meta, False

    new_jti_list: list[str] = []
    for j in jti_list:
        exp_ts: int = int(jti_meta.get(j, 0))
        if exp_ts == 0 or exp_ts > now:
            new_jti_list.append(j)

    new_jti_meta: dict[str, int] = {}
    for j, exp in jti_meta.items():
        if j in new_jti_list and exp > now:
            new_jti_meta[j] = int(exp)

    changed = (new_jti_list != jti_list) or (new_jti_meta != jti_meta)
    return new_jti_list, new_jti_meta, changed


async def prune_user_cache(
    redis_pool: Redis,
    username: str,
) -> UserCache | None:
    """
    Prune a user's cached authentication data in Redis.

    Args:
        redis_pool: Asynchronous Redis client/connection.
        username: Username whose cache entry should be pruned.

    Returns:
        The updated cache dictionary if present, otherwise ``None`` when no
        cache entry exists.
    """
    raw_cache = await get_user_data(redis_pool, username)
    cache = cast(UserCache, raw_cache) if raw_cache is not None else None
    if not cache:
        return None

    now: int = int(time.time())
    changed: bool = False

    # Refresh tokens pruning
    new_refresh_tokens, changed_refresh = _prune_refresh_tokens(cache)
    if new_refresh_tokens != _refresh_tokens(cache):
        cache['refresh_tokens'] = new_refresh_tokens
    changed = changed or changed_refresh

    # JTI metadata pruning
    new_jti_list, new_jti_meta, changed_jti = _prune_jti(cache, now)
    cache['jti_list'] = new_jti_list
    cache['jti_meta'] = new_jti_meta
    changed = changed or changed_jti

    if changed:
        await set_user_data(redis_pool, username, cast(dict[str, object], cache))

    return cache
