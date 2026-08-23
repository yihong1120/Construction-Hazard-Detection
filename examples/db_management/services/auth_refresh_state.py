from __future__ import annotations

import datetime
import hashlib
import json
from collections.abc import Awaitable
from collections.abc import Callable
from typing import cast

from fastapi import HTTPException
from redis.asyncio import Redis

from examples.db_management.schemas.auth import UserCache


def _hash_refresh_token(refresh_token: str) -> str:
    """Hash a refresh token before storing web-cookie session state.

    Args:
        refresh_token: Raw refresh token.

    Returns:
        Stable non-reversible token hash.
    """
    return hashlib.sha256(refresh_token.encode('utf-8')).hexdigest()


def _refresh_state_key(refresh_token: str) -> str:
    """Build the Redis key tracking one refresh token's state.

    Args:
        refresh_token: Raw refresh token.

    Returns:
        Refresh-token state key.
    """
    return f'oauth:refresh-state:{_hash_refresh_token(refresh_token)}'


def _refresh_family_revoked_key(family_id: str) -> str:
    """Build the Redis key marking a refresh-token family revoked.

    Args:
        family_id: Refresh-token rotation-family identifier.

    Returns:
        Family-revocation marker key.
    """
    return f'oauth:refresh-family-revoked:{family_id}'


async def register_refresh_token_state(
    redis_pool: Redis,
    refresh_token: str,
    username: str,
    family_id: str,
    *,
    refresh_ttl: datetime.timedelta,
    enforce_family_active: bool = False,
) -> None:
    """Register an active refresh token without storing its bearer value.

    Args:
        redis_pool: Redis connection holding token state.
        refresh_token: Raw refresh token to register.
        username: Username that owns the token.
        family_id: Rotation-family identifier.
        enforce_family_active: Whether a revoked family must be rejected.

    Raises:
        HTTPException: If the refresh-token family is revoked.
    """
    if enforce_family_active and await redis_pool.get(
        _refresh_family_revoked_key(family_id),
    ):
        raise HTTPException(status_code=401, detail='Refresh token reused')
    await redis_pool.set(
        _refresh_state_key(refresh_token),
        json.dumps(
            {
                'status': 'active',
                'username': username,
                'family_id': family_id,
            },
            separators=(',', ':'),
        ).encode('utf-8'),
        ex=int(refresh_ttl.total_seconds()),
    )


async def revoke_refresh_family(
    redis_pool: Redis,
    family_id: str,
    *,
    refresh_ttl: datetime.timedelta,
) -> None:
    """Mark a refresh-token family revoked for its remaining lifetime.

    Args:
        redis_pool: Redis connection holding token state.
        family_id: Rotation-family identifier to revoke.
    """
    await redis_pool.set(
        _refresh_family_revoked_key(family_id),
        b'1',
        ex=int(refresh_ttl.total_seconds()),
    )


async def revoke_user_access_tokens(
    redis_pool: Redis,
    username: str,
    *,
    get_user_data_fn: Callable[[Redis, str], Awaitable[object]],
    revoke_access_token_jtis_fn: Callable[[Redis, dict[str, int]], Awaitable[int]],
) -> int:
    """Immediately revoke every unexpired access token for a user.

    Args:
        redis_pool: Redis connection holding user token state.
        username: Username whose tokens are revoked.

    Returns:
        Number of access-token identifiers revoked.
    """
    cache = cast(UserCache | None, await get_user_data_fn(redis_pool, username))
    if not cache:
        return 0
    return await revoke_access_token_jtis_fn(redis_pool, cache['jti_meta'])


async def consume_refresh_token_state(
    redis_pool: Redis,
    refresh_token: str,
    family_id: str,
    username: str,
    *,
    refresh_ttl: datetime.timedelta,
    revoke_refresh_family_fn: Callable[..., Awaitable[None]],
    revoke_user_access_tokens_fn: Callable[..., Awaitable[int]],
) -> None:
    """Make a rotating refresh token single-use across all workers.

    Args:
        redis_pool: Redis connection holding token state.
        refresh_token: Raw refresh token to consume.
        family_id: Rotation-family identifier from token claims.
        username: Username from token claims.

    Raises:
        HTTPException: If the token is unknown, reused, or revoked.
    """
    if await redis_pool.get(_refresh_family_revoked_key(family_id)):
        await revoke_user_access_tokens_fn(redis_pool, username)
        raise HTTPException(status_code=401, detail='Refresh token reused')
    lock_key = f'{_refresh_state_key(refresh_token)}:consume'
    acquired = await redis_pool.set(lock_key, b'1', ex=30, nx=True)
    if not acquired:
        await revoke_refresh_family_fn(redis_pool, family_id)
        await revoke_user_access_tokens_fn(redis_pool, username)
        raise HTTPException(status_code=401, detail='Refresh token reused')
    raw = await redis_pool.get(_refresh_state_key(refresh_token))
    if raw is None:
        await revoke_refresh_family_fn(redis_pool, family_id)
        await revoke_user_access_tokens_fn(redis_pool, username)
        raise HTTPException(status_code=401, detail='Refresh token reused')
    state = json.loads(raw)
    if (
        state.get('status') != 'active'
        or state.get('family_id') != family_id
    ):
        await revoke_refresh_family_fn(redis_pool, family_id)
        await revoke_user_access_tokens_fn(redis_pool, username)
        raise HTTPException(status_code=401, detail='Refresh token reused')
    state['status'] = 'used'
    await redis_pool.set(
        _refresh_state_key(refresh_token),
        json.dumps(state, separators=(',', ':')).encode('utf-8'),
        ex=int(refresh_ttl.total_seconds()),
    )


def _cache_contains_refresh_token(
    cache: UserCache,
    refresh_token: str,
) -> bool:
    """Return whether a raw or hashed refresh token is recognised.

    Args:
        cache: Cached user token state.
        refresh_token: Raw refresh token to look up.

    Returns:
        ``True`` when the cache recognises the token.
    """
    if refresh_token in cache['refresh_tokens']:
        return True
    token_hash = _hash_refresh_token(refresh_token)
    return token_hash in cache['refresh_token_hashes']


def _remove_refresh_token_from_cache(
    cache: UserCache,
    refresh_token: str,
) -> None:
    """Remove a refresh token and its hash from a cache payload.

    Args:
        cache: Mutable cached user token state.
        refresh_token: Raw refresh token to remove.
    """
    token_hash = _hash_refresh_token(refresh_token)
    cache['refresh_tokens'] = [
        token
        for token in cache['refresh_tokens']
        if token != refresh_token
    ]
    cache['refresh_token_hashes'] = [
        value
        for value in cache['refresh_token_hashes']
        if value != token_hash
    ]


def _store_refresh_token_in_cache(
    cache: UserCache,
    refresh_token: str,
    *,
    hash_refresh_token: bool = False,
) -> None:
    """Store a refresh token raw for mobile or hashed for web.

    Args:
        cache: Mutable cached user token state.
        refresh_token: Raw refresh token to store.
        hash_refresh_token: Whether to retain only a token hash.
    """
    if hash_refresh_token:
        cache['refresh_token_hashes'].append(
            _hash_refresh_token(refresh_token),
        )
    else:
        cache['refresh_tokens'].append(refresh_token)
