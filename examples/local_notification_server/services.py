from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import defaultdict
from collections.abc import AsyncIterable
from collections.abc import AsyncIterator
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Coroutine
from collections.abc import Iterable
from collections.abc import Mapping
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import cast
from typing import DefaultDict
from typing import Final

import redis.asyncio as redis
from cryptography.fernet import Fernet
from cryptography.fernet import InvalidToken
from redis.asyncio.client import Pipeline
from sqlalchemy import select
from sqlalchemy import update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.config import Settings
from examples.auth.models import FcmDeviceToken
from examples.auth.models import Notification
from examples.auth.models import Site
from examples.auth.models import SiteNotificationPreference
from examples.auth.models import User
from examples.auth.models import USER_STATUS_ACTIVE
from examples.local_notification_server.fcm_service import (
    FcmSendResult,
)
from examples.local_notification_server.fcm_service import (
    send_fcm_notification_service,
)
from examples.local_notification_server.lang_config import LANGUAGES
from examples.local_notification_server.lang_config import NotificationLanguage
from examples.local_notification_server.lang_config import Translator
from examples.local_notification_server.schemas import DeviceRegistrationRequest
from examples.local_notification_server.schemas import SiteNotifyRequest
from src.warning_types import Warnings

# Bound cache rebuilding and FCM fan-out so one busy site cannot monopolise a worker.
_recipient_index_ready_value: Final[str] = '1'
_token_fetch_chunk_size: Final[int] = 500
_fcm_batch_size: Final[int] = 100
_fcm_max_concurrency: Final[int] = 8
_notification_record_language: Final[NotificationLanguage] = 'zh-TW'
settings = Settings()
logger = logging.getLogger(__name__)

PushTaskResult = FcmSendResult


def fcm_token_hash(device_token: str) -> str:
    """Hash an FCM token for metadata keys and API responses.

    Args:
        device_token: Raw Firebase Cloud Messaging registration token.

    Returns:
        Stable SHA-256 hexadecimal digest that is safe to retain in metadata.
    """
    return hashlib.sha256(device_token.encode('utf-8')).hexdigest()


def _fcm_token_fernet() -> Fernet:
    """Build the configured Fernet encryptor for FCM tokens at rest.

    Returns:
        Fernet instance created from ``FCM_TOKEN_ENCRYPTION_KEY``.

    Raises:
        ValueError: If the required key is absent or not a Fernet key.
    """
    return Fernet(settings.fcm_token_encryption_key.encode('utf-8'))


def encrypt_fcm_token(device_token: str) -> str:
    """Encrypt an FCM token before storing it in the database.

    Args:
        device_token: Raw Firebase Cloud Messaging registration token.

    Returns:
        URL-safe Fernet ciphertext encoded as UTF-8 text.
    """
    return _fcm_token_fernet().encrypt(
        device_token.encode('utf-8'),
    ).decode('utf-8')


def decrypt_fcm_token(encrypted_token: str) -> str:
    """Decrypt an FCM token loaded from the database.

    Args:
        encrypted_token: Fernet-encrypted device token from the database.

    Returns:
        Decrypted UTF-8 device token.

    Raises:
        cryptography.fernet.InvalidToken: If stored encrypted data cannot be
            verified with the configured Fernet key.
    """
    return _fcm_token_fernet().decrypt(
        encrypted_token.encode('utf-8'),
    ).decode('utf-8')


def _disable_undecryptable_fcm_token(
    row: FcmDeviceToken,
    occurred_at: datetime,
) -> None:
    """Mark an FCM registration unusable when its encrypted token is unreadable.

    Args:
        row: Persisted FCM registration that could not be decrypted.
        occurred_at: UTC timestamp used for the failure and disabled fields.
    """
    row.disabled_at = occurred_at
    row.last_failure_at = occurred_at
    row.failure_reason = 'token_decryption_failed'
    row.updated_at = occurred_at


def _token_index_key(user_id: int) -> str:
    """Build the Redis set key for a user's token hashes.

    Args:
        user_id: Token owner identifier.

    Returns:
        Redis key indexing metadata hashes for the user's tokens.
    """
    return f'fcm_token_index:{user_id}'


def _token_meta_key(user_id: int, token_hash: str) -> str:
    """Build the Redis hash key for one token's non-sensitive metadata.

    Args:
        user_id: Token owner identifier.
        token_hash: SHA-256 digest of the raw FCM token.

    Returns:
        Redis key containing metadata for the token registration.
    """
    return f'fcm_token_meta:{user_id}:{token_hash}'


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
    user_key = f'fcm_tokens:{row.user_id}'
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
    pipe.expire(user_key, 86400 * 30)
    pipe.sadd(_token_index_key(row.user_id), row.device_token_hash)
    pipe.hset(
        meta_key,
        mapping=cast(
            Mapping[str | bytes, bytes | float | int | str],
            mapping,
        ),
    )
    pipe.expire(meta_key, 86400 * 30)
    pipe.expire(_token_index_key(row.user_id), 86400 * 30)


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
        FcmDeviceToken | None, await db.scalar(
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
        CursorResult[Any], await db.execute(
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
                if row.last_success_at is not None else None
            ),
            'last_failure_at': (
                _datetime_to_api(row.last_failure_at)
                if row.last_failure_at is not None else None
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
    """Rebuild the Redis sendable-token cache from the database source of truth.

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
        pipe.delete(f'fcm_tokens:{user_id}')
        pipe.delete(_token_index_key(user_id))

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
        # Ownership is resolved from each user's cached token map before removal.
        raw_map = cast(
            Mapping[bytes, bytes],
            await rds.hgetall(f'fcm_tokens:{user_id}'),
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
            pipe.hdel(f'fcm_tokens:{user_id}', token)
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


def _site_user_cache_key(site_name: str) -> str:
    """Build the Redis set key for site notification recipients.

    Args:
        site_name: Site name used by notification requests.

    Returns:
        Redis set key containing recipient user IDs.
    """
    return f'site_notification_users:{site_name}'


def _site_user_cache_ready_key(site_name: str) -> str:
    """Build the Redis readiness key for a site recipient index.

    Args:
        site_name: Site name used by notification requests.

    Returns:
        Redis key indicating that the recipient index is ready.
    """
    return f'site_notification_users_ready:{site_name}'


async def _fetch_site_notification_user_ids_from_db(
    site_name: str,
    db: AsyncSession,
) -> list[int] | None:
    """Load current recipient user IDs for a site from the database.

    Args:
        site_name: Site name to look up.
        db: Async database session dependency.

    Returns:
        Active recipient user IDs, or None when the site does not exist.
    """
    stmt = select(Site.id).where(Site.name == site_name)
    site_id_row = (await db.execute(stmt)).first()
    if site_id_row is None:
        return None
    site_id = site_id_row[0]

    users_stmt = (
        select(SiteNotificationPreference.user_id)
        .join(User, User.id == SiteNotificationPreference.user_id)
        .where(
            SiteNotificationPreference.site_id == site_id,
            SiteNotificationPreference.is_enabled.is_(True),
            User.status == USER_STATUS_ACTIVE,
        )
    )
    # Only explicit opt-ins from active accounts may enter the delivery index.
    return list((await db.execute(users_stmt)).scalars().all())


async def refresh_site_notification_user_cache(
    site_name: str,
    db: AsyncSession,
    rds: redis.Redis,
) -> list[int] | None:
    """Rebuild the Redis recipient index for a site from the database.

    Args:
        site_name: Site name to rebuild.
        db: Async database session dependency.
        rds: Redis connection used to write the recipient index.

    Returns:
        Active recipient user IDs, or None when the site does not exist.
    """
    user_ids = await _fetch_site_notification_user_ids_from_db(site_name, db)
    if user_ids is None:
        await invalidate_site_notification_user_cache([site_name], rds)
        return None

    pipe = rds.pipeline()
    cache_key = _site_user_cache_key(site_name)
    ready_key = _site_user_cache_ready_key(site_name)
    # A ready marker distinguishes an empty subscription from a cold cache.
    pipe.delete(cache_key)
    if user_ids:
        pipe.sadd(cache_key, *user_ids)
    pipe.set(ready_key, _recipient_index_ready_value)
    await pipe.execute()
    return user_ids


async def _get_site_user_index_members(
    site_name: str,
    rds: redis.Redis,
) -> list[int]:
    """Read recipient IDs from the Redis set for a site.

    Args:
        site_name: Site name to read.
        rds: Redis connection used to read the recipient index.

    Returns:
        Recipient user IDs from Redis.
    """
    members = cast(
        Awaitable[set[bytes]],
        rds.smembers(_site_user_cache_key(site_name)),
    )
    return [int(member) for member in await members]


async def invalidate_site_notification_user_cache(
    site_names: list[str],
    rds: redis.Redis,
) -> None:
    """Delete Redis recipient indexes for the given sites.

    Args:
        site_names: Site names whose indexes should be removed.
        rds: Redis connection used to delete cache keys.
    """
    keys: list[str] = []
    for site_name in site_names:
        keys.extend([
            _site_user_cache_key(site_name),
            _site_user_cache_ready_key(site_name),
        ])
    if keys:
        await rds.delete(*keys)


async def get_site_notification_user_ids_cached(
    site_name: str,
    db: AsyncSession,
    rds: redis.Redis,
) -> list[int] | None:
    """Get notification recipient IDs using the Redis site index.

    Args:
        site_name: Site name to look up.
        db: Database session used for cold-cache rebuilds.
        rds: Redis connection used as the live recipient index.

    Returns:
        Recipient user IDs if the site exists; otherwise ``None``.
    """
    ready_key = _site_user_cache_ready_key(site_name)
    if await rds.exists(ready_key):
        return await _get_site_user_index_members(site_name, rds)
    return await refresh_site_notification_user_cache(site_name, db, rds)


def _decode_lang_token_map(
    raw_maps: Iterable[Mapping[bytes, bytes]],
) -> DefaultDict[NotificationLanguage, list[str]]:
    """Decode Redis hash results into a language-to-tokens map.

    Args:
        raw_maps: Byte mappings representing users' token-to-language entries.

    Returns:
        Tokens grouped by canonical BCP 47 language code.
    """
    lang_to_tokens: DefaultDict[NotificationLanguage, list[str]] = (
        defaultdict(list)
    )
    for raw_map in raw_maps:
        for token_b, lang_b in raw_map.items():
            token: str = _decode_redis_string(token_b)
            lang = cast(NotificationLanguage, _decode_redis_string(lang_b))
            lang_to_tokens[lang].append(token)
    return lang_to_tokens


async def diagnose_push_preflight(
    req: SiteNotifyRequest,
    user_ids: list[int],
    rds: redis.Redis,
) -> dict[str, object]:
    """Return diagnostics for why notification recipients did not send.

    Args:
        req: Validated notification request.
        user_ids: Recipient user IDs to inspect.
        rds: Redis connection used to read token hashes.

    Returns:
        JSON-serialisable diagnostics for log and API responses.
    """
    diagnostics = await _collect_push_token_diagnostics(user_ids, rds)
    tokens_by_language = cast(
        Mapping[NotificationLanguage, int],
        diagnostics['tokens_by_language'],
    )
    translated_languages, sendable_tokens = _sendable_push_languages(
        req.body,
        tokens_by_language,
    )

    return {
        'recipient_users': len(user_ids),
        'users_with_tokens': diagnostics['users_with_tokens'],
        'token_entries': diagnostics['token_entries'],
        'unique_tokens': diagnostics['unique_tokens'],
        'sendable_tokens': sendable_tokens,
        'tokens_by_language': dict(
            sorted(tokens_by_language.items()),
        ),
        'body_keys': list(req.body.keys()),
        'translated_languages': sorted(translated_languages),
    }


async def _collect_push_token_diagnostics(
    user_ids: list[int],
    rds: redis.Redis,
) -> dict[str, object]:
    """Collect recipient-token counts in Redis-sized chunks.

    Args:
        user_ids: Recipient identifiers whose cached tokens are inspected.
        rds: Redis connection used to read token-to-language hashes.

    Returns:
        Aggregate counts and per-language token totals.
    """
    users_with_tokens = 0
    token_entries = 0
    tokens_by_language: DefaultDict[NotificationLanguage, int] = defaultdict(
        int,
    )
    for start in range(0, len(user_ids), _token_fetch_chunk_size):
        pipe = rds.pipeline()
        for user_id in user_ids[start:start + _token_fetch_chunk_size]:
            pipe.hgetall(f'fcm_tokens:{user_id}')
        results: list[dict[bytes, bytes]] = await pipe.execute()
        for raw_map in results:
            if raw_map:
                users_with_tokens += 1
            for _raw_token, raw_language in raw_map.items():
                token_entries += 1
                language = cast(
                    NotificationLanguage,
                    _decode_redis_string(raw_language),
                )
                tokens_by_language[language] += 1
    return {
        'users_with_tokens': users_with_tokens,
        'token_entries': token_entries,
        'unique_tokens': token_entries,
        'tokens_by_language': tokens_by_language,
    }


def _sendable_push_languages(
    body: Warnings,
    tokens_by_language: Mapping[NotificationLanguage, int],
) -> tuple[list[NotificationLanguage], int]:
    """Return languages with complete translations and their token count.

    Args:
        body: Validated warning payload to translate.
        tokens_by_language: Token count for every recipient language.

    Returns:
        Languages that can render the payload and their total token count.
    """
    translated_languages: list[NotificationLanguage] = []
    sendable_tokens = 0
    for language, token_count in tokens_by_language.items():
        Translator.translate_from_dict(body, language)
        translated_languages.append(language)
        sendable_tokens += token_count
    return translated_languages, sendable_tokens


async def create_notification_records_for_users(
    req: SiteNotifyRequest,
    user_ids: list[int],
    db: AsyncSession,
) -> int:
    """Persist one notification-centre record for each recipient user.

    Args:
        req: Validated site-notification request.
        user_ids: Potential recipient user identifiers.
        db: Database session used to create notification records.

    Returns:
        Number of recipient records persisted.
    """
    record_body = f'{req.site} - {req.stream_name}\n' + '\n'.join(
        Translator.translate_from_dict(
            req.body, _notification_record_language,
        ),
    )
    records = [
        Notification(
            user_id=user_id,
            type=req.notification_type,
            title=req.title,
            body=record_body,
            deep_link=req.deep_link,
            metadata_json=req.metadata,
        )
        for user_id in user_ids
    ]
    db.add_all(records)
    await db.commit()
    return len(user_ids)


def _build_push_task(
    req: SiteNotifyRequest,
    lang: NotificationLanguage,
    tokens: list[str],
) -> Awaitable[PushTaskResult]:
    """Build one FCM batch task for one canonical language.

    Args:
        req: Validated notification request.
        lang: Canonical language code for the target tokens.
        tokens: Device tokens in this batch.

    Returns:
        Awaitable send task.
    """
    title = LANGUAGES[lang]['warning_notification']
    translated_lines = Translator.translate_from_dict(req.body, lang)
    data = {
        'navigate': 'violation_list_page',
        'violation_id': (
            str(req.violation_id) if req.violation_id is not None else ''
        ),
        'deep_link': req.deep_link,
        'type': req.notification_type,
    }

    body: str = f"{req.site} - {req.stream_name}\n" + \
        '\n'.join(translated_lines)

    logger.info(
        'FCM notification batch prepared lang=%s tokens=%d body_lines=%d '
        'data_keys=%s',
        lang,
        len(tokens),
        len(translated_lines),
        sorted(data),
    )

    return send_fcm_notification_service(
        device_tokens=tokens,
        title=title,
        body=body,
        image_path=req.image_path,
        data=data,
    )


async def _iter_push_tasks_streaming(
    req: SiteNotifyRequest,
    user_ids: list[int],
    rds: redis.Redis,
) -> AsyncIterator[Awaitable[PushTaskResult]]:
    """Stream Redis token chunks into FCM batch tasks.

    This avoids materialising all device tokens in a single
    ``lang_to_tokens`` map. Memory is bounded by the Redis chunk, one partial
    FCM batch per active language, and the executor's active tasks.

    Args:
        req: Validated notification request.
        user_ids: Recipient user IDs to fetch tokens for.
        rds: Redis connection used to read token hashes.

    Yields:
        Awaitable FCM batch send tasks.
    """
    pending_batches: DefaultDict[NotificationLanguage, list[str]] = (
        defaultdict(list)
    )

    for start in range(0, len(user_ids), _token_fetch_chunk_size):
        # Read bounded Redis chunks before forming language-specific FCM batches.
        pipe = rds.pipeline()
        for user_id in user_ids[start:start + _token_fetch_chunk_size]:
            pipe.hgetall(f'fcm_tokens:{user_id}')

        redis_results: list[dict[bytes, bytes]] = await pipe.execute()
        chunk_tokens = _decode_lang_token_map(redis_results)

        for lang, tokens in chunk_tokens.items():
            batch = pending_batches[lang]
            for token in tokens:
                batch.append(token)
                if len(batch) >= _fcm_batch_size:
                    task = _build_push_task(req, lang, list(batch))
                    batch.clear()
                    yield task

    for lang, tokens in pending_batches.items():
        if not tokens:
            continue
        yield _build_push_task(req, lang, list(tokens))


async def _execute_push_tasks_bounded_streaming(
    push_tasks: AsyncIterable[Awaitable[PushTaskResult]],
    invalid_token_handler: Callable[[tuple[str, ...]], Awaitable[object]],
    timeout: float = 30.0,
    max_concurrency: int = _fcm_max_concurrency,
) -> tuple[bool, int | None, int | None, str | None]:
    """Execute streamed push tasks with bounded concurrency.

    The async iterable may fetch Redis chunks while producing tasks, so this
    executor pulls only enough batches to fill the concurrency window.

    Args:
        push_tasks: Async iterable that yields awaitable FCM batch send tasks.
        timeout: Maximum execution time in seconds.
        max_concurrency: Maximum number of active send tasks.
        invalid_token_handler: Callback receiving invalid tokens.

    Returns:
        Tuple of `(ok, total_batches, successful_batches, error_message)`.
    """
    window = _run_streaming_push_task_window(
        push_tasks,
        max_concurrency,
        invalid_token_handler,
    )
    return await _complete_push_task_window(window, timeout)


async def _fill_pending_streaming_push_tasks(
    task_iter: AsyncIterator[Awaitable[PushTaskResult]],
    pending: set[asyncio.Future[PushTaskResult]],
    max_workers: int,
) -> None:
    """Fill a streaming FCM task window until capacity or iterator exhaustion.

    Args:
        task_iter: Async iterator yielding unscheduled FCM batch tasks.
        pending: Mutable set of currently scheduled tasks.
        max_workers: Maximum number of concurrently scheduled tasks.
    """
    while len(pending) < max_workers:
        try:
            awaitable = await task_iter.__anext__()
        except StopAsyncIteration:
            return
        pending.add(asyncio.ensure_future(awaitable))


async def _collect_completed_push_tasks(
    pending: set[asyncio.Future[PushTaskResult]],
) -> tuple[int, int, set[str]]:
    """Await completed tasks and aggregate their FCM result details.

    Args:
        pending: Currently scheduled FCM batch tasks.

    Returns:
        Completed count, successful count, and invalid tokens from the window.
    """
    done, _ = await asyncio.wait(
        pending,
        return_when=asyncio.FIRST_COMPLETED,
    )
    pending.difference_update(done)
    successful_batches = 0
    invalid_tokens: set[str] = set()
    for task in done:
        result = await task
        successful_batches += int(
            result.success_count > 0 and result.failure_count == 0,
        )
        invalid_tokens.update(result.invalid_tokens)
    return len(done), successful_batches, invalid_tokens


async def _run_streaming_push_task_window(
    push_tasks: AsyncIterable[Awaitable[PushTaskResult]],
    max_workers: int,
    invalid_token_handler: Callable[[tuple[str, ...]], Awaitable[object]],
) -> tuple[int, int]:
    """Run lazily streamed push tasks with bounded concurrency.

    Args:
        push_tasks: Async iterable yielding FCM batch tasks on demand.
        max_workers: Maximum number of concurrently scheduled tasks.
        invalid_token_handler: Callback for invalid FCM tokens.

    Returns:
        Total and successful FCM batch counts.
    """
    pending: set[asyncio.Future[PushTaskResult]] = set()
    total_batches = 0
    successful_batches = 0
    invalid_tokens: set[str] = set()
    task_iter = push_tasks.__aiter__()
    try:
        await _fill_pending_streaming_push_tasks(
            task_iter,
            pending,
            max_workers,
        )
        while pending:
            total, successful, invalid = await _collect_completed_push_tasks(
                pending,
            )
            total_batches += total
            successful_batches += successful
            invalid_tokens.update(invalid)
            await _fill_pending_streaming_push_tasks(
                task_iter,
                pending,
                max_workers,
            )
    finally:
        # Cancel scheduled work and release the stream's Redis resources.
        for task in pending:
            task.cancel()
    if invalid_tokens:
        await invalid_token_handler(tuple(sorted(invalid_tokens)))
    return total_batches, successful_batches


async def _complete_push_task_window(
    window: Coroutine[Any, Any, tuple[int, int]],
    timeout: float,
) -> tuple[bool, int | None, int | None, str | None]:
    """Apply the public timeout and error contract to one task window.

    Args:
        window: FCM task-window coroutine aggregation operation.
        timeout: Maximum time allowed for all work in the window.

    Returns:
        Completion flag, total batches, successful batches, and error code.
    """
    try:
        total, successful = await asyncio.wait_for(window, timeout=timeout)
        return True, total, successful, None
    except asyncio.TimeoutError:
        window.close()
        return False, None, None, 'FCM notification sending timed out.'
    except Exception:
        window.close()
        return False, None, None, 'internal_error'
