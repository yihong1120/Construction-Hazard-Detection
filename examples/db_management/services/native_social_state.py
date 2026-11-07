from __future__ import annotations

import hashlib
import json
import re
import secrets
from collections.abc import Mapping
from typing import Literal

from fastapi import HTTPException
from fastapi import Request
from redis.asyncio import Redis

from examples.auth.config import Settings


NativeSocialProvider = Literal['google', 'apple']


settings = Settings()


_TRANSACTION_RE = re.compile(r'^[A-Za-z0-9_-]{43,128}$')


_KEYCLOAK_SUBJECT_RE = re.compile(r'^[A-Za-z0-9._:-]{1,255}$')


_REDEEM_HMAC_MAX_AGE_SECONDS = 30


_BEGIN_RATE_LIMIT_MAX = 20


_BEGIN_RATE_LIMIT_WINDOW_SECONDS = 300


def _not_enabled() -> HTTPException:
    """Return the consistent disabled-feature response."""
    return HTTPException(
        status_code=404,
        detail='native_social_exchange_not_enabled',
    )


def _require_enabled() -> None:
    """Fail closed unless the server has every native-exchange secret."""
    if not settings.native_social_exchange_enabled:
        raise _not_enabled()


def _opaque_identifier() -> str:
    """Create an unguessable URL-safe transaction identifier."""
    return secrets.token_urlsafe(32)


def _transaction_key(transaction_id: str) -> str:
    """Hash public transaction IDs before using them as Redis keys."""
    digest = hashlib.sha256(transaction_id.encode('ascii')).hexdigest()
    return f'native-social:transaction:{digest}'


def _rate_limit_key(request: Request) -> str:
    """Build a privacy-preserving anonymous begin rate-limit key."""
    host = request.client.host if request.client else 'unknown'
    digest = hashlib.sha256(host.encode('utf-8')).hexdigest()
    return f'native-social:begin-rate:{digest}'


async def _enforce_begin_rate_limit(
    request: Request,
    redis: Redis,
) -> None:
    """Bound unauthenticated transaction allocation per trusted peer."""
    key = _rate_limit_key(request)
    count = await redis.incr(key)
    if count == 1:
        await redis.expire(key, _BEGIN_RATE_LIMIT_WINDOW_SECONDS)
    if count > _BEGIN_RATE_LIMIT_MAX:
        retry_after = await redis.ttl(key)
        raise HTTPException(
            status_code=429,
            detail='native_social_exchange_rate_limited',
            headers={
                'Retry-After': str(
                    (
                        max(1, retry_after)
                        if isinstance(retry_after, int)
                        else _BEGIN_RATE_LIMIT_WINDOW_SECONDS
                    ),
                ),
            },
        )


def _validate_client_binding(
    client_id: str,
    redirect_uri: str,
) -> None:
    """Require one of the server-configured Keycloak public clients."""
    allowed = settings.native_social_allowed_clients
    if redirect_uri not in allowed.get(client_id, ()):
        raise HTTPException(status_code=400, detail='invalid_oidc_client')


def _decode_record(raw: bytes | str | None) -> dict[str, object]:
    """Decode a Redis transaction record without exposing malformed state."""
    if raw is None:
        raise HTTPException(
            status_code=401,
            detail='native_social_exchange_expired',
        )
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=401,
            detail='native_social_exchange_expired',
        ) from exc
    if not isinstance(value, dict):
        raise HTTPException(
            status_code=401,
            detail='native_social_exchange_expired',
        )
    return value


async def _consume_transaction(
    redis: Redis,
    transaction_id: str,
) -> dict[str, object]:
    """Atomically consume a one-use transaction before validation work."""
    if not _TRANSACTION_RE.fullmatch(transaction_id):
        raise HTTPException(
            status_code=401,
            detail='native_social_exchange_expired',
        )
    return _decode_record(await redis.getdel(_transaction_key(transaction_id)))


async def _store_transaction(
    redis: Redis,
    transaction_id: str,
    payload: Mapping[str, object],
    ttl_seconds: int,
) -> None:
    """Store compact, non-secret transaction state with an absolute TTL."""
    await redis.set(
        _transaction_key(transaction_id),
        json.dumps(payload, separators=(',', ':')).encode('utf-8'),
        ex=ttl_seconds,
    )


def _required_string(record: Mapping[str, object], name: str) -> str:
    """Return a bounded string field from trusted transaction state."""
    value = record.get(name)
    if not isinstance(value, str) or not value:
        raise HTTPException(
            status_code=401,
            detail='native_social_exchange_expired',
        )
    return value


def _record_provider(record: Mapping[str, object]) -> NativeSocialProvider:
    """Return the fixed provider stored in a transaction."""
    provider = record.get('provider')
    if provider not in {'google', 'apple'}:
        raise HTTPException(
            status_code=401,
            detail='native_social_exchange_expired',
        )
    return provider
