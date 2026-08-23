from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from typing import cast
from typing import TypedDict

from redis.asyncio import Redis

_ACCESS_REVOCATION_PREFIX = 'auth:access-revoked'


class AccessTokenRevocationPayload(TypedDict):
    """Access-token claims required by the revocation store."""

    jti: str
    exp: int


def access_token_jti(payload: Mapping[str, object]) -> str:
    """Return the signed access-token identifier from a JWT payload."""
    return cast(str, payload['jti'])


def _revocation_key(jti: str) -> str:
    """Return a Redis key without exposing a token identifier."""
    digest = hashlib.sha256(jti.encode('utf-8')).hexdigest()
    return f"{_ACCESS_REVOCATION_PREFIX}:{digest}"


def _remaining_lifetime(payload: AccessTokenRevocationPayload) -> int:
    """Return the positive remaining lifetime of a token in seconds."""
    return max(0, payload['exp'] - int(time.time()))


async def revoke_access_token(
    redis: Redis,
    payload: AccessTokenRevocationPayload,
) -> bool:
    """Mark one currently valid access token as revoked until it expires."""
    jti = access_token_jti(payload)
    ttl = _remaining_lifetime(payload)
    if ttl <= 0:
        return False
    await redis.set(_revocation_key(jti), '1', ex=ttl)
    return True


async def revoke_access_token_jtis(
    redis: Redis,
    jti_expirations: Mapping[str, int],
) -> int:
    """Revoke every unexpired access-token identifier in a user cache."""
    revoked = 0
    for jti, expires_at in jti_expirations.items():
        if await revoke_access_token(redis, {'jti': jti, 'exp': expires_at}):
            revoked += 1
    return revoked


async def is_access_token_revoked(
    redis: Redis,
    payload: Mapping[str, object],
) -> bool:
    """Return whether a signed access token was explicitly revoked."""
    jti = access_token_jti(payload)
    return bool(await redis.exists(_revocation_key(jti)))
