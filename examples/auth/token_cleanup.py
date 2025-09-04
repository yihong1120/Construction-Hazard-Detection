from __future__ import annotations

import time

import jwt
from redis.asyncio import Redis

from examples.auth.cache import get_user_data
from examples.auth.cache import set_user_data
from examples.auth.config import Settings


settings = Settings()


def _typed_refresh_tokens(cache: dict[str, object]) -> list[str]:
    """
    Return list of refresh tokens if present and validly typed.

    Args:
        cache: The user cache dictionary.

    Returns:
        A list of refresh tokens as strings.
    """
    raw = cache.get('refresh_tokens', [])
    if isinstance(raw, list):
        return [t for t in raw if isinstance(t, str)]
    return []


def _prune_refresh_tokens(cache: dict[str, object]) -> tuple[list[str], bool]:
    """
    Return valid refresh tokens and whether the list changed.


    Args:
        cache: The user cache dictionary.

    Returns:
    A tuple of (new_tokens, changed) where new_tokens is the list of
    valid refresh tokens and changed is a boolean indicating if the
    list was modified.
    """
    tokens = _typed_refresh_tokens(cache)
    if not tokens:
        # Nothing to validate
        return tokens, False

    new_tokens: list[str] = []
    changed = False
    for tok in tokens:
        try:
            jwt.decode(
                tok,
                settings.authjwt_secret_key,
                algorithms=[settings.ALGORITHM],
            )
            new_tokens.append(tok)
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
            changed = True
            continue
    return new_tokens, changed or (new_tokens != tokens)


def _typed_jti(cache: dict[str, object]) -> tuple[list[str], dict[str, int]]:
    """
    Return (jti_list, jti_meta) if present and validly typed.

    Args:
        cache: The user cache dictionary.

    Returns:
        A tuple of (jti_list, jti_meta) where jti_list is a list of JTI
        strings and jti_meta is a dictionary mapping JTI strings to their
        expiration timestamps.
    """
    jti_list_raw = cache.get('jti_list', [])
    if isinstance(jti_list_raw, list):
        jti_list = [j for j in jti_list_raw if isinstance(j, str)]
    else:
        jti_list = []

    jti_meta_raw = cache.get('jti_meta', {})
    if isinstance(jti_meta_raw, dict):
        jti_meta = {
            k: int(v)
            for k, v in jti_meta_raw.items()
            if isinstance(k, str) and isinstance(v, int)
        }
    else:
        jti_meta = {}
    return jti_list, jti_meta


def _prune_jti(
    cache: dict[str, object],
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
    jti_list, jti_meta = _typed_jti(cache)
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

    now: int = int(time.time())
    changed: bool = False

    # Refresh tokens pruning
    new_refresh_tokens, changed_refresh = _prune_refresh_tokens(cache)
    if new_refresh_tokens != _typed_refresh_tokens(cache):
        cache['refresh_tokens'] = new_refresh_tokens
    changed = changed or changed_refresh

    # JTI metadata pruning
    new_jti_list, new_jti_meta, changed_jti = _prune_jti(cache, now)
    if new_jti_list is not None:
        cache['jti_list'] = new_jti_list
    if new_jti_meta is not None:
        cache['jti_meta'] = new_jti_meta
    changed = changed or changed_jti

    if changed:
        await set_user_data(redis_pool, username, cache)

    return cache
