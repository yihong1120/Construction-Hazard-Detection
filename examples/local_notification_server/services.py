from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any
from typing import cast

import redis.asyncio as redis
from cryptography.fernet import InvalidToken
from fastapi import HTTPException
from redis.asyncio.client import Pipeline
from redis.exceptions import RedisError
from sqlalchemy import select
from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.config import Settings
from examples.auth.models import FcmDeviceToken
from examples.auth.redis_transactions import optimistic
from examples.local_notification_server.fcm_token_crypto import (
    decrypt_token,
)
from examples.local_notification_server.fcm_token_crypto import (
    disable_undecryptable_token as _disable_undecryptable_fcm_token,
)
from examples.local_notification_server.fcm_token_crypto import encrypt_token
from examples.local_notification_server.fcm_token_crypto import fcm_token_hash
from examples.local_notification_server.fcm_token_keys import (
    _token_cache_ready_key,
)
from examples.local_notification_server.fcm_token_keys import _token_index_key
from examples.local_notification_server.fcm_token_keys import _token_meta_key
from examples.local_notification_server.schemas import (
    DeviceRegistrationRequest,
)

settings = Settings()
logger = logging.getLogger(__name__)
_token_cache_ttl_seconds = 86400 * 30

_REMOVE_REGISTRATION_CACHE = """
local revision = redis.call('HGET', KEYS[3], 'registration_revision')
if revision and revision ~= ARGV[3] then return 0 end
redis.call('DEL', KEYS[4])
if not revision or ARGV[3] == '' then return 0 end
redis.call('HDEL', KEYS[1], ARGV[1])
redis.call('SREM', KEYS[2], ARGV[2])
redis.call('DEL', KEYS[3])
return 1
"""


def encrypt_fcm_token(device_token: str) -> str:
    """Encrypt an FCM token with the current application key."""
    return encrypt_token(device_token, settings.fcm_token_encryption_key)


def decrypt_fcm_token(encrypted_token: str) -> str:
    """Decrypt an FCM token with the current application key."""
    return decrypt_token(encrypted_token, settings.fcm_token_encryption_key)


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
        'registration_revision': row.last_seen_at.replace(
            tzinfo=timezone.utc,
        ).isoformat(),
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
    token_hash = fcm_token_hash(req.device_token)
    now = datetime.now(timezone.utc)
    encrypted = encrypt_fcm_token(req.device_token)
    inserted = await db.execute(
        insert(FcmDeviceToken).values(
            user_id=user_id, device_token_encrypted=encrypted,
            device_token_hash=token_hash, platform=req.platform,
            device_lang=req.device_lang, permission_status='unknown',
            last_seen_at=now, created_at=now, updated_at=now,
        ).on_conflict_do_nothing(
            index_elements=['device_token_hash'],
        ).returning(FcmDeviceToken),
    )
    row = inserted.scalar_one_or_none()
    created = row is not None
    if row is None:
        row = await db.scalar(
            select(FcmDeviceToken).where(
                FcmDeviceToken.device_token_hash == token_hash,
            ).with_for_update().execution_options(populate_existing=True),
        )
    if row is None:
        raise HTTPException(503, 'device_registration_changed')
    previous_owner = None if created else row.user_id
    registered_at = row.created_at
    if not created:
        now = max(
            datetime.now(timezone.utc),
            row.last_seen_at.replace(tzinfo=timezone.utc)
            + timedelta(microseconds=1),
        )
        row.user_id = user_id
        row.device_token_encrypted = encrypted
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

    try:
        pipe = rds.pipeline(transaction=True)
        if previous_owner is not None and previous_owner != user_id:
            pipe.hdel(f'fcm_tokens:{previous_owner}', req.device_token)
            pipe.srem(_token_index_key(previous_owner), token_hash)
            pipe.delete(
                _token_meta_key(previous_owner, token_hash),
                _token_cache_ready_key(previous_owner),
            )
        _queue_token_cache_write(pipe, row, req.device_token)
        pipe.delete(_token_cache_ready_key(user_id))
        await pipe.execute()
    except Exception:
        # Durable ownership is authoritative. The next send reconciles cache
        # revisions against the database before using any cached device.
        logger.warning('FCM cache update deferred for user_id=%s', user_id)
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
    result = await db.execute(
        update(FcmDeviceToken)
        .where(
            FcmDeviceToken.user_id == user_id,
            FcmDeviceToken.device_token_hash == token_hash,
            FcmDeviceToken.disabled_at.is_(None),
        )
        .values(
            disabled_at=now,
            last_failure_at=None,
            failure_reason=None,
            updated_at=now,
        )
        .returning(FcmDeviceToken.last_seen_at)
        .execution_options(synchronize_session=False),
    )
    revision = result.scalar_one_or_none()
    await db.commit()
    try:
        await rds.eval(
            _REMOVE_REGISTRATION_CACHE,
            4,
            f'fcm_tokens:{user_id}',
            _token_index_key(user_id),
            _token_meta_key(user_id, token_hash),
            _token_cache_ready_key(user_id),
            device_token,
            token_hash,
            (
                revision.replace(tzinfo=timezone.utc).isoformat()
                if revision is not None
                else ''
            ),
        )
    except RedisError:
        logger.warning(
            'FCM unregister cache cleanup deferred user_id=%s',
            user_id,
        )
    return revision is not None


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
            'registration_revision': row.last_seen_at.replace(
                tzinfo=timezone.utc,
            ).isoformat(),
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
    snapshot = await _decrypt_active_fcm_tokens([user_id], db)
    return [token for _, token in snapshot.active]


@dataclass(repr=False)
class FcmTokenSnapshot:
    active: list[tuple[FcmDeviceToken, str]]
    disabled: list[tuple[int, str]]


async def _decrypt_active_fcm_tokens(
    user_ids: list[int], db: AsyncSession,
) -> FcmTokenSnapshot:
    disabled: list[tuple[int, str]] = []
    for _ in range(5):
        result = await db.execute(
            select(FcmDeviceToken).where(
                FcmDeviceToken.user_id.in_(user_ids),
                FcmDeviceToken.disabled_at.is_(None),
            ).order_by(
                FcmDeviceToken.user_id, FcmDeviceToken.last_seen_at.desc(),
            ).execution_options(populate_existing=True),
        )
        active: list[tuple[FcmDeviceToken, str]] = []
        changed = False
        failed = False
        for row in result.scalars().all():
            try:
                active.append(
                    (row, decrypt_fcm_token(row.device_token_encrypted)),
                )
            except InvalidToken:
                failed = True
                if await _disable_undecryptable_fcm_token(
                    db, row, datetime.now(timezone.utc),
                ):
                    disabled.append((row.user_id, row.device_token_hash))
                    logger.warning(
                        'Disabled undecryptable FCM token: user_id=%s',
                        row.user_id,
                    )
                else:
                    changed = True
        if failed:
            await db.commit()
        if not changed:
            return FcmTokenSnapshot(active, disabled)
    raise HTTPException(503, 'device_registration_changed')


def _cache_revision(records: Iterable[tuple[str, datetime]]) -> str:
    source = '\n'.join(
        sorted(
            f'{token_hash}:{updated.isoformat()}'
            for token_hash, updated in records
        ),
    )
    return hashlib.sha256(source.encode()).hexdigest()


@dataclass
class _FcmTokenCacheRebuild:
    user_ids: list[int]
    db: AsyncSession

    async def apply(self, pipe: Pipeline) -> int:
        await pipe.watch(
            *[_token_cache_ready_key(user) for user in self.user_ids],
        )
        previous = {
            user: await pipe.smembers(_token_index_key(user))
            for user in self.user_ids
        }
        snapshot = await _decrypt_active_fcm_tokens(self.user_ids, self.db)
        decoded = snapshot.active
        revisions: dict[int, list[tuple[str, datetime]]] = {
            user: [] for user in self.user_ids
        }
        for row, _ in decoded:
            revisions[row.user_id].append(
                (row.device_token_hash, row.updated_at),
            )
        pipe.multi()
        for user_id in self.user_ids:
            pipe.delete(f'fcm_tokens:{user_id}', _token_index_key(user_id))
            for token_hash in previous[user_id]:
                text = (
                    token_hash.decode()
                    if isinstance(token_hash, bytes)
                    else str(token_hash)
                )
                pipe.delete(_token_meta_key(user_id, text))
            pipe.set(
                _token_cache_ready_key(user_id),
                _cache_revision(revisions[user_id]),
                ex=_token_cache_ttl_seconds,
            )
        for user, token_hash in snapshot.disabled:
            pipe.delete(_token_meta_key(user, token_hash))
        for row, token in decoded:
            _queue_token_cache_write(pipe, row, token)
        await pipe.execute()
        return len(decoded)


async def refresh_fcm_token_cache_for_users(
    user_ids: list[int],
    db: AsyncSession,
    rds: redis.Redis,
) -> int:
    users = list(dict.fromkeys(user_ids))
    if not users:
        return 0
    return await optimistic(rds, _FcmTokenCacheRebuild(users, db).apply)


async def ensure_fcm_token_cache_for_users(
    user_ids: list[int],
    db: AsyncSession,
    rds: redis.Redis,
) -> int:
    """Validate durable ownership revisions before trusting cached devices."""
    users = list(dict.fromkeys(user_ids))
    if not users:
        return 0
    result = await db.execute(
        select(
            FcmDeviceToken.user_id,
            FcmDeviceToken.device_token_hash,
            FcmDeviceToken.updated_at,
        ).where(
            FcmDeviceToken.user_id.in_(users),
            FcmDeviceToken.disabled_at.is_(None),
        ),
    )
    records: dict[int, list[tuple[str, datetime]]] = {
        user: [] for user in users
    }
    for user, token_hash, updated in result.all():
        records[user].append((token_hash, updated))
    ready = await rds.mget([_token_cache_ready_key(user) for user in users])
    stale = [
        user
        for user, value in zip(users, ready, strict=True)
        if value != _cache_revision(records[user]).encode()
    ]
    return (
        await refresh_fcm_token_cache_for_users(stale, db, rds) if stale else 0
    )
