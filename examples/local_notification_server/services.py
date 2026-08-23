from __future__ import annotations

import logging
from collections.abc import Iterable
from collections.abc import Mapping
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import cast

import redis.asyncio as redis
from cryptography.fernet import InvalidToken
from redis.asyncio.client import Pipeline
from sqlalchemy import select
from sqlalchemy import update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.config import Settings
from examples.auth.models import FcmDeviceToken
from examples.local_notification_server.fcm_token_crypto import (
    decrypt_token,
)
from examples.local_notification_server.fcm_token_crypto import (
    disable_undecryptable_token as _disable_undecryptable_fcm_token,
)
from examples.local_notification_server.fcm_token_crypto import encrypt_token
from examples.local_notification_server.fcm_token_crypto import fcm_token_hash
from examples.local_notification_server.schemas import (
    DeviceRegistrationRequest,
)

settings = Settings()
logger = logging.getLogger(__name__)
_token_cache_ttl_seconds = 86400 * 30


def encrypt_fcm_token(device_token: str) -> str:
    """Encrypt an FCM token with the current application key."""
    return encrypt_token(device_token, settings.fcm_token_encryption_key)


def decrypt_fcm_token(encrypted_token: str) -> str:
    """Decrypt an FCM token with the current application key."""
    return decrypt_token(encrypted_token, settings.fcm_token_encryption_key)


def _token_index_key(user_id: int) -> str:
    """Build the Redis set key for a user's token hashes.

    Args:
        user_id: Token owner identifier.

    Returns:
        Redis key indexing metadata hashes for the user's tokens.
    """
    return f"fcm_token_index:{user_id}"


def _token_meta_key(user_id: int, token_hash: str) -> str:
    """Build the Redis hash key for one token's non-sensitive metadata.

    Args:
        user_id: Token owner identifier.
        token_hash: SHA-256 digest of the raw FCM token.

    Returns:
        Redis key containing metadata for the token registration.
    """
    return f"fcm_token_meta:{user_id}:{token_hash}"


def _token_cache_ready_key(user_id: int) -> str:
    """Build the marker that distinguishes a cold token cache from no
    tokens."""
    return f"fcm_tokens_ready:{user_id}"


def _decode_redis_string(value: bytes) -> str:
    """Decode a Redis hash key or value into text.

    Args:
        value: UTF-8 bytes returned by Redis.

    Returns:
        Decoded text value.
    """
    return value.decode()


def _datetime_to_api(value: datetime) -> str:
    """Serialise UTC datetimes for API responses and Redis metadata.

    Args:
        value: Timezone-aware or naïve timestamp to normalise to UTC.

    Returns:
        ISO 8601 timestamp with a ``Z`` UTC suffix and no microseconds.
    """
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace('+00:00', 'Z')
    )


def _queue_token_cache_write(
    pipe: Pipeline[Any],
    row: FcmDeviceToken,
    device_token: str,
) -> None:
    """Queue Redis writes for one active database-backed FCM token.

    Args:
        pipe: Redis pipeline receiving cache write operations.
        row: Persisted token entity supplying durable metadata.
        device_token: Decrypted FCM token used only as the send-cache key.
    """
    user_key = f"fcm_tokens:{row.user_id}"
    meta_key = _token_meta_key(row.user_id, row.device_token_hash)
    mapping: dict[str, str] = {
        'token_hash': row.device_token_hash,
        'device_lang': row.device_lang,
        'registered_at': _datetime_to_api(row.created_at),
        'last_seen_at': _datetime_to_api(row.last_seen_at),
        'is_active': 'true' if row.disabled_at is None else 'false',
    }
    if row.last_success_at is not None:
        mapping['last_success_at'] = _datetime_to_api(row.last_success_at)
    if row.last_failure_at is not None:
        mapping['last_failure_at'] = _datetime_to_api(row.last_failure_at)
    if row.failure_reason is not None:
        mapping['failure_reason'] = row.failure_reason
    # Redis is a rebuildable delivery cache, so every entry uses the same TTL.
    pipe.hset(user_key, device_token, row.device_lang)
    pipe.expire(user_key, _token_cache_ttl_seconds)
    pipe.sadd(_token_index_key(row.user_id), row.device_token_hash)
    pipe.hset(
        meta_key,
        mapping=cast(
            Mapping[str | bytes, bytes | float | int | str],
            mapping,
        ),
    )
    pipe.expire(meta_key, _token_cache_ttl_seconds)
    pipe.expire(_token_index_key(row.user_id), _token_cache_ttl_seconds)
    pipe.set(
        _token_cache_ready_key(row.user_id),
        '1',
        ex=_token_cache_ttl_seconds,
    )


async def record_fcm_token_registration(
    user_id: int,
    req: DeviceRegistrationRequest,
    db: AsyncSession,
    rds: redis.Redis,
) -> dict[str, str]:
    """Persist an FCM token and refresh its Redis send-cache entry.

    Args:
        user_id: Authenticated identifier that owns the token.
        req: Validated token registration request.
        db: Database session used for the durable token record.
        rds: Redis connection used for the rebuildable send cache.

    Returns:
        Non-sensitive registration timestamps and token hash metadata.
    """
    now = datetime.now(timezone.utc).replace(microsecond=0)
    token_hash = fcm_token_hash(req.device_token)
    row = cast(
        FcmDeviceToken | None,
        await db.scalar(
            select(FcmDeviceToken).where(
                FcmDeviceToken.device_token_hash == token_hash,
            ),
        ),
    )
    if row is None:
        # A token hash is globally unique, preventing duplicate registrations.
        row = FcmDeviceToken(
            user_id=user_id,
            device_token_encrypted=encrypt_fcm_token(req.device_token),
            device_token_hash=token_hash,
            platform=req.platform,
            device_lang=req.device_lang,
            permission_status='unknown',
            last_seen_at=now,
            created_at=now,
            updated_at=now,
        )
        db.add(row)
        registered_at = now
    else:
        registered_at = row.created_at
        row.user_id = user_id
        row.device_token_encrypted = encrypt_fcm_token(req.device_token)
        row.platform = req.platform
        row.device_lang = req.device_lang
        row.permission_status = 'unknown'
        row.app_version = None
        row.web_vapid_key_available = None
        row.web_service_worker_registered = None
        row.last_seen_at = now
        row.failure_reason = None
        row.disabled_at = None
        row.updated_at = now

    await db.commit()

    pipe = rds.pipeline()
    _queue_token_cache_write(pipe, row, req.device_token)
    await pipe.execute()
    return {
        'token_hash': token_hash,
        'registered_at': _datetime_to_api(registered_at),
        'last_seen_at': _datetime_to_api(now),
    }


async def delete_fcm_token_metadata(
    user_id: int,
    device_token: str,
    db: AsyncSession,
    rds: redis.Redis,
) -> bool:
    """Disable metadata for an explicitly deleted token.

    Args:
        user_id: Identifier of the token owner.
        device_token: Raw token being explicitly removed.
        db: Database session used to disable the durable record.
        rds: Redis connection used to remove cached metadata.

    Returns:
        True when a database token row was disabled, otherwise False.
    """
    token_hash = fcm_token_hash(device_token)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    result = cast(
        CursorResult[Any],
        await db.execute(
            update(FcmDeviceToken)
            .where(
                FcmDeviceToken.user_id == user_id,
                FcmDeviceToken.device_token_hash == token_hash,
            )
            .values(
                disabled_at=now,
                last_failure_at=None,
                failure_reason=None,
                updated_at=now,
            ),
        ),
    )
    await db.commit()
    await rds.delete(_token_meta_key(user_id, token_hash))
    await rds.srem(_token_index_key(user_id), token_hash)
    return cast(int, result.rowcount) > 0


async def list_fcm_device_status(
    user_id: int,
    db: AsyncSession,
) -> list[dict[str, object]]:
    """Return current and historical FCM token status rows from the database.

    Args:
        user_id: Identifier of the token owner.
        db: Database session used to load token records.

    Returns:
        Non-sensitive token diagnostic mappings, newest active tokens first.
    """
    result = await db.execute(
        select(FcmDeviceToken)
        .where(FcmDeviceToken.user_id == user_id)
        .order_by(
            FcmDeviceToken.disabled_at.asc().nullsfirst(),
            FcmDeviceToken.last_seen_at.desc(),
            FcmDeviceToken.id.desc(),
        ),
    )
    rows = cast(list[FcmDeviceToken], result.scalars().all())
    return [
        {
            'token_hash': row.device_token_hash,
            'platform': row.platform,
            'device_lang': row.device_lang,
            'permission_status': row.permission_status,
            'registered_at': _datetime_to_api(row.created_at),
            'last_seen_at': _datetime_to_api(row.last_seen_at),
            'last_success_at': (
                _datetime_to_api(row.last_success_at)
                if row.last_success_at is not None
                else None
            ),
            'last_failure_at': (
                _datetime_to_api(row.last_failure_at)
                if row.last_failure_at is not None
                else None
            ),
            'failure_reason': row.failure_reason,
            'is_active': row.disabled_at is None,
        }
        for row in rows
    ]


async def load_active_fcm_device_tokens(
    user_id: int,
    db: AsyncSession,
) -> list[str]:
    """Load decrypted active FCM tokens for one user from the database.

    Args:
        user_id: Identifier of the token owner.
        db: Database session used to load active records.

    Returns:
        Raw tokens for immediate Firebase delivery only.
    """
    result = await db.execute(
        select(FcmDeviceToken)
        .where(
            FcmDeviceToken.user_id == user_id,
            FcmDeviceToken.disabled_at.is_(None),
        )
        .order_by(FcmDeviceToken.last_seen_at.desc()),
    )
    now = datetime.now(timezone.utc).replace(microsecond=0)
    tokens: list[str] = []
    disabled_tokens = False
    rows = cast(list[FcmDeviceToken], result.scalars().all())
    for row in rows:
        try:
            tokens.append(decrypt_fcm_token(row.device_token_encrypted))
        except InvalidToken:
            # Ciphertext encrypted with a retired key cannot be recovered.
            _disable_undecryptable_fcm_token(row, now)
            disabled_tokens = True
            logger.warning(
                'Disabled undecryptable FCM token: user_id=%s token_hash=%s',
                row.user_id,
                row.device_token_hash,
            )
    if disabled_tokens:
        await db.commit()
    return tokens


async def refresh_fcm_token_cache_for_users(
    user_ids: list[int],
    db: AsyncSession,
    rds: redis.Redis,
) -> int:
    """Rebuild the Redis sendable-token cache from the database source of
    truth.

    Args:
        user_ids: Token owner identifiers whose cache entries are rebuilt.
        db: Database session used to load active token records.
        rds: Redis connection used to replace send-cache entries.

    Returns:
        Number of active token entries written to Redis.
    """
    unique_user_ids = list(dict.fromkeys(user_ids))
    result = await db.execute(
        select(FcmDeviceToken)
        .where(
            FcmDeviceToken.user_id.in_(unique_user_ids),
            FcmDeviceToken.disabled_at.is_(None),
        )
        .order_by(FcmDeviceToken.user_id, FcmDeviceToken.last_seen_at.desc()),
    )
    pipe = rds.pipeline()
    for user_id in unique_user_ids:
        # Atomically replace the entire rebuildable cache for each recipient.
        pipe.delete(f"fcm_tokens:{user_id}")
        pipe.delete(_token_index_key(user_id))
        # A ready marker also records a deliberate empty token set.  Without
        # it, every notification would query the database again for users who
        # have never registered a device.
        pipe.set(
            _token_cache_ready_key(user_id),
            '1',
            ex=_token_cache_ttl_seconds,
        )

    now = datetime.now(timezone.utc).replace(microsecond=0)
    cached = 0
    disabled_tokens = False
    rows = cast(list[FcmDeviceToken], result.scalars().all())
    for row in rows:
        try:
            token = decrypt_fcm_token(row.device_token_encrypted)
        except InvalidToken:
            # A retired key or corrupt ciphertext must not abort delivery.
            _disable_undecryptable_fcm_token(row, now)
            pipe.delete(_token_meta_key(row.user_id, row.device_token_hash))
            disabled_tokens = True
            logger.warning(
                'Disabled undecryptable FCM token: user_id=%s token_hash=%s',
                row.user_id,
                row.device_token_hash,
            )
            continue
        _queue_token_cache_write(pipe, row, token)
        cached += 1
    if disabled_tokens:
        await db.commit()
    await pipe.execute()
    return cached


async def ensure_fcm_token_cache_for_users(
    user_ids: list[int],
    db: AsyncSession,
    rds: redis.Redis,
) -> int:
    """Hydrate only cold recipient token caches from the durable store.

    Device registration and invalid-token handling keep active caches current.
    This function is therefore a cache-through miss path, rather than the
    former per-notification full rebuild.
    """
    unique_user_ids = list(dict.fromkeys(user_ids))
    if not unique_user_ids:
        return 0
    ready_values = await rds.mget(
        [_token_cache_ready_key(user_id) for user_id in unique_user_ids],
    )
    missing_ids = [
        user_id
        for user_id, ready in zip(unique_user_ids, ready_values)
        if ready is None
    ]
    if not missing_ids:
        return 0
    return await refresh_fcm_token_cache_for_users(missing_ids, db, rds)


async def mark_fcm_tokens_success(
    user_id: int,
    device_tokens: Iterable[str],
    rds: redis.Redis,
    db: AsyncSession,
) -> None:
    """Mark successful test sends for token diagnostics.

    Args:
        user_id: Identifier of the token owner.
        device_tokens: Tokens confirmed as successfully delivered.
        rds: Redis connection used to update cached diagnostics.
        db: Database session used to persist diagnostics.
    """
    now_dt = datetime.now(timezone.utc).replace(microsecond=0)
    now = _datetime_to_api(now_dt)
    token_hashes = [fcm_token_hash(token) for token in device_tokens]
    await db.execute(
        update(FcmDeviceToken)
        .where(
            FcmDeviceToken.user_id == user_id,
            FcmDeviceToken.device_token_hash.in_(token_hashes),
        )
        .values(
            last_success_at=now_dt,
            last_failure_at=None,
            failure_reason=None,
            disabled_at=None,
            updated_at=now_dt,
        ),
    )
    await db.commit()

    pipe = rds.pipeline()
    for token_hash in token_hashes:
        pipe.hset(
            _token_meta_key(user_id, token_hash),
            mapping={
                'last_success_at': now,
                'last_failure_at': '',
                'failure_reason': '',
                'is_active': 'true',
            },
        )
    await pipe.execute()


async def mark_fcm_tokens_failure(
    user_id: int,
    device_tokens: Iterable[str],
    rds: redis.Redis,
    reason: str,
    db: AsyncSession,
) -> None:
    """Mark failed sends for token diagnostics.

    Args:
        user_id: Identifier of the token owner.
        device_tokens: Tokens for which delivery failed.
        rds: Redis connection used to update cached diagnostics.
        reason: Stable diagnostic code describing the failure.
        db: Database session used to persist diagnostics.
    """
    now_dt = datetime.now(timezone.utc).replace(microsecond=0)
    now = _datetime_to_api(now_dt)
    token_hashes = [fcm_token_hash(token) for token in device_tokens]
    await db.execute(
        update(FcmDeviceToken)
        .where(
            FcmDeviceToken.user_id == user_id,
            FcmDeviceToken.device_token_hash.in_(token_hashes),
        )
        .values(
            last_failure_at=now_dt,
            failure_reason=reason,
            updated_at=now_dt,
        ),
    )
    await db.commit()

    pipe = rds.pipeline()
    for token_hash in token_hashes:
        pipe.hset(
            _token_meta_key(user_id, token_hash),
            mapping={
                'last_failure_at': now,
                'failure_reason': reason,
            },
        )
    await pipe.execute()


async def mark_invalid_fcm_tokens_for_users(
    user_ids: list[int],
    invalid_tokens: Iterable[str],
    rds: redis.Redis,
    db: AsyncSession,
    reason: str = 'invalid_token',
) -> None:
    """Disable invalid tokens and retain their failure diagnostics.

    Args:
        user_ids: Candidate owners of the invalid tokens.
        invalid_tokens: Tokens Firebase identified as unusable.
        rds: Redis connection used to remove send-cache entries.
        reason: Stable diagnostic code stored for disabled tokens.
        db: Database session used to persist disabled state.
    """
    invalid_set = set(invalid_tokens)
    now_dt = datetime.now(timezone.utc).replace(microsecond=0)
    now = _datetime_to_api(now_dt)
    invalid_hashes = [fcm_token_hash(token) for token in invalid_set]
    await db.execute(
        update(FcmDeviceToken)
        .where(
            FcmDeviceToken.user_id.in_(user_ids),
            FcmDeviceToken.device_token_hash.in_(invalid_hashes),
        )
        .values(
            disabled_at=now_dt,
            last_failure_at=now_dt,
            failure_reason=reason,
            updated_at=now_dt,
        ),
    )
    await db.commit()

    for user_id in user_ids:
        # Ownership is resolved from each user's cached token map before
        # removal.
        raw_map = cast(
            Mapping[bytes, bytes],
            await rds.hgetall(f"fcm_tokens:{user_id}"),
        )
        owned_invalid_tokens = [
            token
            for raw_token in raw_map
            if (token := _decode_redis_string(raw_token)) in invalid_set
        ]
        if not owned_invalid_tokens:
            continue
        pipe = rds.pipeline()
        for token in owned_invalid_tokens:
            token_hash = fcm_token_hash(token)
            pipe.hdel(f"fcm_tokens:{user_id}", token)
            pipe.sadd(_token_index_key(user_id), token_hash)
            pipe.hset(
                _token_meta_key(user_id, token_hash),
                mapping={
                    'last_failure_at': now,
                    'failure_reason': reason,
                    'is_active': 'false',
                },
            )
        await pipe.execute()
