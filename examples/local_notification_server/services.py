from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
from collections import defaultdict
from collections.abc import AsyncIterable
from collections.abc import AsyncIterator
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
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
from examples.local_notification_server.lang_config import normalize_language
from examples.local_notification_server.lang_config import Translator
from examples.local_notification_server.schemas import NotificationType
from examples.local_notification_server.schemas import SiteNotifyRequest
from examples.local_notification_server.schemas import TokenRequest
from src.warning_types import Warnings

# Redis recipient index settings.
_recipient_index_ready_value: Final[str] = '1'
_recipient_index_lock_seconds: Final[int] = 30
_recipient_index_wait_attempts: Final[int] = 5
_recipient_index_wait_seconds: Final[float] = 0.05
_token_fetch_chunk_size: Final[int] = 500
_fcm_batch_size: Final[int] = 100
_fcm_max_concurrency: Final[int] = 8
_notification_record_language: Final[str] = 'zh-TW'
settings = Settings()

PushTaskResult = bool | FcmSendResult


def _utc_now_iso() -> str:
    """Return a compact UTC timestamp for Redis metadata."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace('+00:00', 'Z')
    )


def fcm_token_hash(device_token: str) -> str:
    """Hash an FCM token for metadata keys and API responses."""
    return hashlib.sha256(device_token.encode('utf-8')).hexdigest()


def _fcm_token_fernet() -> Fernet:
    """Build a stable Fernet encryptor for FCM tokens at rest."""
    raw_key = settings.fcm_token_encryption_key.strip()
    if raw_key:
        try:
            return Fernet(raw_key.encode('utf-8'))
        except (ValueError, TypeError):
            pass

    key_source = raw_key or settings.authjwt_secret_key
    key = base64.urlsafe_b64encode(
        hashlib.sha256(key_source.encode('utf-8')).digest(),
    )
    return Fernet(key)


def encrypt_fcm_token(device_token: str) -> str:
    """Encrypt an FCM token before storing it in the database."""
    return _fcm_token_fernet().encrypt(
        device_token.encode('utf-8'),
    ).decode('utf-8')


def decrypt_fcm_token(encrypted_token: str) -> str:
    """Decrypt an FCM token loaded from the database."""
    try:
        return _fcm_token_fernet().decrypt(
            encrypted_token.encode('utf-8'),
        ).decode('utf-8')
    except InvalidToken:
        return ''


def _token_index_key(user_id: int) -> str:
    return f'fcm_token_index:{user_id}'


def _token_meta_key(user_id: int, token_hash: str) -> str:
    return f'fcm_token_meta:{user_id}:{token_hash}'


def _decode_redis_string(value: bytes | str | None) -> str:
    """Decode a Redis hash key or value into text."""
    if value is None:
        return ''
    if isinstance(value, bytes):
        return value.decode()
    return value


def _decode_redis_bool(value: object) -> bool | None:
    decoded = _decode_redis_string(cast(bytes | str | None, value))
    if decoded == '':
        return None
    return decoded.lower() == 'true'


def _encode_optional_bool(value: bool | None) -> str:
    if value is None:
        return ''
    return 'true' if value else 'false'


def _datetime_to_api(value: datetime | None) -> str:
    """Serialise UTC datetimes for API responses and Redis metadata."""
    if value is None:
        return ''
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace('+00:00', 'Z')
    )


def _platform_value(req: TokenRequest) -> str:
    return req.platform or 'unknown'


def _permission_status_value(req: TokenRequest) -> str:
    return req.permission_status or 'unknown'


def _fcm_token_status_row(row: FcmDeviceToken) -> dict[str, object]:
    """Convert an FCM token ORM row into a non-sensitive status payload."""
    return {
        'token_hash': row.device_token_hash,
        'platform': row.platform or 'unknown',
        'device_lang': row.device_lang,
        'permission_status': row.permission_status or 'unknown',
        'registered_at': _datetime_to_api(row.created_at) or None,
        'last_seen_at': _datetime_to_api(row.last_seen_at) or None,
        'last_success_at': _datetime_to_api(row.last_success_at) or None,
        'last_failure_at': _datetime_to_api(row.last_failure_at) or None,
        'failure_reason': row.failure_reason,
        'is_active': row.disabled_at is None,
        'web_vapid_key_available': row.web_vapid_key_available,
        'web_service_worker_registered': (
            row.web_service_worker_registered
        ),
    }


def _result_scalars_all(result: object) -> list[object]:
    """Return SQLAlchemy scalar rows, tolerating loose mocks in tests."""
    scalars = getattr(result, 'scalars', None)
    if not callable(scalars):
        return []
    if inspect.iscoroutinefunction(scalars):
        return []
    scalar_result = scalars()
    if inspect.isawaitable(scalar_result):
        return []
    all_method = getattr(scalar_result, 'all', None)
    if not callable(all_method):
        return []
    if inspect.iscoroutinefunction(all_method):
        return []
    rows = all_method()
    if inspect.isawaitable(rows):
        return []
    return list(rows)


def _queue_token_cache_write(
    pipe: Pipeline[Any],
    row: FcmDeviceToken,
    device_token: str,
) -> None:
    """Queue Redis writes for one active DB-backed FCM token."""
    user_key = f'fcm_tokens:{row.user_id}'
    meta_key = _token_meta_key(row.user_id, row.device_token_hash)
    mapping: dict[str | bytes, bytes | float | int | str] = {
        'token_hash': row.device_token_hash,
        'platform': row.platform or 'unknown',
        'device_lang': row.device_lang,
        'permission_status': row.permission_status or 'unknown',
        'registered_at': _datetime_to_api(row.created_at),
        'last_seen_at': _datetime_to_api(row.last_seen_at),
        'last_success_at': _datetime_to_api(row.last_success_at),
        'last_failure_at': _datetime_to_api(row.last_failure_at),
        'failure_reason': row.failure_reason or '',
        'is_active': 'true' if row.disabled_at is None else 'false',
        'web_vapid_key_available': _encode_optional_bool(
            row.web_vapid_key_available,
        ),
        'web_service_worker_registered': _encode_optional_bool(
            row.web_service_worker_registered,
        ),
    }
    pipe.hset(user_key, device_token, row.device_lang)
    pipe.expire(user_key, 86400 * 30)
    pipe.sadd(_token_index_key(row.user_id), row.device_token_hash)
    pipe.hset(meta_key, mapping=mapping)
    pipe.expire(meta_key, 86400 * 30)
    pipe.expire(_token_index_key(row.user_id), 86400 * 30)


async def record_fcm_token_registration(
    req: TokenRequest,
    device_lang: str,
    db: AsyncSession,
    rds: redis.Redis,
) -> dict[str, str]:
    """Persist an FCM token in DB and refresh the Redis send cache."""
    now = datetime.now(timezone.utc).replace(microsecond=0)
    token_hash = fcm_token_hash(req.device_token)
    row = await db.scalar(
        select(FcmDeviceToken).where(
            FcmDeviceToken.device_token_hash == token_hash,
        ),
    )
    if not isinstance(row, FcmDeviceToken):
        row = None
    if row is None:
        row = FcmDeviceToken(
            user_id=req.user_id,
            device_token_encrypted=encrypt_fcm_token(req.device_token),
            device_token_hash=token_hash,
            platform=_platform_value(req),
            device_lang=device_lang,
            permission_status=_permission_status_value(req),
            app_version=req.app_version,
            web_vapid_key_available=req.web_vapid_key_available,
            web_service_worker_registered=(
                req.web_service_worker_registered
            ),
            last_seen_at=now,
            created_at=now,
            updated_at=now,
        )
        db.add(row)
        registered_at = now
    else:
        registered_at = row.created_at or now
        row.user_id = req.user_id
        row.device_token_encrypted = encrypt_fcm_token(req.device_token)
        row.platform = _platform_value(req)
        row.device_lang = device_lang
        row.permission_status = _permission_status_value(req)
        row.app_version = req.app_version
        row.web_vapid_key_available = req.web_vapid_key_available
        row.web_service_worker_registered = (
            req.web_service_worker_registered
        )
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

    Returns:
        True when a DB token row was disabled, otherwise False.
    """
    token_hash = fcm_token_hash(device_token)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    result = await db.execute(
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
    )
    await db.commit()
    await rds.delete(_token_meta_key(user_id, token_hash))
    await rds.srem(_token_index_key(user_id), token_hash)
    rowcount = getattr(result, 'rowcount', 0)
    return isinstance(rowcount, int) and rowcount > 0


async def list_fcm_device_status(
    user_id: int,
    db: AsyncSession,
) -> list[dict[str, object]]:
    """Return current and historical FCM token status rows from DB."""
    result = await db.execute(
        select(FcmDeviceToken)
        .where(FcmDeviceToken.user_id == user_id)
        .order_by(
            FcmDeviceToken.disabled_at.asc().nullsfirst(),
            FcmDeviceToken.last_seen_at.desc(),
            FcmDeviceToken.id.desc(),
        ),
    )
    return [
        _fcm_token_status_row(cast(FcmDeviceToken, row))
        for row in _result_scalars_all(result)
        if isinstance(row, FcmDeviceToken)
    ]


async def load_active_fcm_device_tokens(
    user_id: int,
    db: AsyncSession,
) -> list[str]:
    """Load decrypted active FCM tokens for one user from DB."""
    result = await db.execute(
        select(FcmDeviceToken)
        .where(
            FcmDeviceToken.user_id == user_id,
            FcmDeviceToken.disabled_at.is_(None),
        )
        .order_by(FcmDeviceToken.last_seen_at.desc()),
    )
    tokens: list[str] = []
    for row in _result_scalars_all(result):
        if not isinstance(row, FcmDeviceToken):
            continue
        token = decrypt_fcm_token(row.device_token_encrypted)
        if token:
            tokens.append(token)
    return tokens


async def refresh_fcm_token_cache_for_users(
    user_ids: list[int],
    db: AsyncSession,
    rds: redis.Redis,
) -> int:
    """Rebuild Redis sendable-token cache from DB source of truth."""
    unique_user_ids = list(dict.fromkeys(user_ids))
    if not unique_user_ids:
        return 0
    result = await db.execute(
        select(FcmDeviceToken)
        .where(
            FcmDeviceToken.user_id.in_(unique_user_ids),
            FcmDeviceToken.disabled_at.is_(None),
        )
        .order_by(FcmDeviceToken.user_id, FcmDeviceToken.last_seen_at.desc()),
    )
    pipe = rds.pipeline()
    cached = 0
    for row in _result_scalars_all(result):
        if not isinstance(row, FcmDeviceToken):
            continue
        token = decrypt_fcm_token(row.device_token_encrypted)
        if not token:
            continue
        _queue_token_cache_write(pipe, row, token)
        cached += 1
    if cached:
        await pipe.execute()
    return cached


async def mark_fcm_tokens_success(
    user_id: int,
    device_tokens: Iterable[str],
    rds: redis.Redis,
    db: AsyncSession | None = None,
) -> None:
    """Mark successful test sends for token diagnostics."""
    now_dt = datetime.now(timezone.utc).replace(microsecond=0)
    now = _datetime_to_api(now_dt)
    token_hashes = [fcm_token_hash(token) for token in device_tokens]
    if db is not None and token_hashes:
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
    db: AsyncSession | None = None,
) -> None:
    """Mark failed sends for token diagnostics."""
    now_dt = datetime.now(timezone.utc).replace(microsecond=0)
    now = _datetime_to_api(now_dt)
    token_hashes = [fcm_token_hash(token) for token in device_tokens]
    if db is not None and token_hashes:
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
    reason: str = 'invalid_token',
    db: AsyncSession | None = None,
) -> None:
    """Remove invalid tokens from sendable hashes and keep failure metadata."""
    invalid_set = set(invalid_tokens)
    if not invalid_set:
        return

    now_dt = datetime.now(timezone.utc).replace(microsecond=0)
    now = _datetime_to_api(now_dt)
    invalid_hashes = [fcm_token_hash(token) for token in invalid_set]
    if db is not None:
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
        raw_map = cast(
            Mapping[bytes | str, bytes | str | None],
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


def _site_user_cache_lock_key(site_name: str) -> str:
    """Build the Redis lock key used while rebuilding a site index.

    Args:
        site_name: Site name used by notification requests.

    Returns:
        Redis lock key for recipient index rebuilds.
    """
    return f'site_notification_users_lock:{site_name}'


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
        Awaitable[set[bytes | str]],
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
            _site_user_cache_lock_key(site_name),
        ])
    if keys:
        await rds.delete(*keys)


async def get_site_notification_user_ids_cached(
    site_name: str,
    db: AsyncSession,
    rds: redis.Redis,
) -> list[int] | None:
    """
    Get notification recipient user IDs for a site using a Redis index.

    Args:
        site_name: The site name to look up.
        db: An async SQLAlchemy session used for cold rebuilds.
        rds: Redis connection used as the live recipient index.

    Returns:
        A list of user IDs if the site exists; otherwise ``None``.
    """
    ready_key = _site_user_cache_ready_key(site_name)
    lock_key = _site_user_cache_lock_key(site_name)

    if await rds.exists(ready_key):
        return await _get_site_user_index_members(site_name, rds)

    lock_acquired = await rds.set(
        lock_key,
        _recipient_index_ready_value,
        ex=_recipient_index_lock_seconds,
        nx=True,
    )
    if lock_acquired:
        try:
            return await refresh_site_notification_user_cache(
                site_name, db, rds,
            )
        finally:
            await rds.delete(lock_key)

    for _ in range(_recipient_index_wait_attempts):
        await asyncio.sleep(_recipient_index_wait_seconds)
        if await rds.exists(ready_key):
            return await _get_site_user_index_members(site_name, rds)

    return await refresh_site_notification_user_cache(site_name, db, rds)


def _decode_lang_token_map(
    raw_maps: Iterable[
        Mapping[bytes, bytes | None] | Mapping[str, str | None]
    ],
) -> DefaultDict[str, list[str]]:
    """
    Decode Redis HGETALL results into a language-to-tokens map.

    Args:
        raw_maps: A list of byte dictionaries from Redis, each representing a
            user's token-to-language mapping.

    Returns:
        A mapping from BCP 47 language code to a list of device tokens.
    """
    lang_to_tokens: DefaultDict[str, list[str]] = defaultdict(list)
    seen_tokens: set[str] = set()
    for raw_map in raw_maps:
        for token_b, lang_b in raw_map.items():
            token: str = _decode_redis_string(token_b)
            if token in seen_tokens:
                continue
            seen_tokens.add(token)
            lang = normalize_language(_decode_redis_string(lang_b))
            if lang is None:
                continue
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
    users_with_tokens = 0
    token_entries = 0
    duplicate_tokens = 0
    unsupported_language_tokens = 0
    sendable_tokens = 0
    seen_tokens: set[str] = set()
    tokens_by_language: DefaultDict[str, int] = defaultdict(int)
    unsupported_languages: DefaultDict[str, int] = defaultdict(int)

    for start in range(0, len(user_ids), _token_fetch_chunk_size):
        pipe = rds.pipeline()
        for user_id in user_ids[start:start + _token_fetch_chunk_size]:
            pipe.hgetall(f'fcm_tokens:{user_id}')

        redis_results: list[dict[bytes, bytes]] = await pipe.execute()
        for raw_map in redis_results:
            if raw_map:
                users_with_tokens += 1

            for raw_token, raw_lang in raw_map.items():
                token = _decode_redis_string(raw_token)
                if not token:
                    continue

                token_entries += 1
                if token in seen_tokens:
                    duplicate_tokens += 1
                    continue
                seen_tokens.add(token)

                raw_language = _decode_redis_string(raw_lang)
                language = normalize_language(raw_language)
                if language is None:
                    unsupported_language_tokens += 1
                    unsupported_languages[raw_language or '<empty>'] += 1
                    continue

                tokens_by_language[language] += 1

    translated_languages: list[str] = []
    for language, token_count in tokens_by_language.items():
        if _translate_title(language) and _translate_body_lines(
            req.body, language,
        ):
            sendable_tokens += token_count
            translated_languages.append(language)

    return {
        'recipient_users': len(user_ids),
        'users_with_tokens': users_with_tokens,
        'token_entries': token_entries,
        'unique_tokens': len(seen_tokens),
        'duplicate_tokens': duplicate_tokens,
        'sendable_tokens': sendable_tokens,
        'unsupported_language_tokens': unsupported_language_tokens,
        'tokens_by_language': dict(sorted(tokens_by_language.items())),
        'unsupported_languages': dict(sorted(unsupported_languages.items())),
        'body_keys': list(req.body.keys()),
        'translated_languages': sorted(translated_languages),
    }


async def _get_lang_to_tokens(
    user_ids: list[int], rds: redis.Redis,
) -> DefaultDict[str, list[str]]:
    """
    Fetch device tokens for users and group them by language.

    Args:
        user_ids: The user IDs to fetch tokens for.
        rds: An asyncio Redis client instance.

    Returns:
        A mapping from language code to the list of tokens in that language.
    """
    lang_to_tokens: DefaultDict[str, list[str]] = defaultdict(list)
    for start in range(0, len(user_ids), _token_fetch_chunk_size):
        pipe = rds.pipeline()
        for user_id in user_ids[start:start + _token_fetch_chunk_size]:
            pipe.hgetall(f'fcm_tokens:{user_id}')
        redis_results: list[dict[bytes, bytes]] = await pipe.execute()
        chunk_tokens = _decode_lang_token_map(redis_results)
        for lang, tokens in chunk_tokens.items():
            lang_to_tokens[lang].extend(tokens)
    return lang_to_tokens


def _translate_title(lang: str) -> str:
    """
    Translate notification title by language.

    Args:
        lang: A BCP 47 language tag.

    Returns:
        The translated title string, or an empty string for unsupported
        languages.
    """
    language = normalize_language(lang)
    if language is None:
        return ''
    return LANGUAGES[language].get('warning_notification', '')


def _translate_body_lines(
    body_dict: Warnings,
    lang: str,
) -> list[str]:
    """
    Translate body lines using the given language.

    Args:
        body_dict: Mapping from language to message spec dictionary.
        lang: Target language to translate into.

    Returns:
        A list of translated message lines.
    """
    return Translator.translate_from_dict(body_dict, lang)


def _notification_record_type(req: SiteNotifyRequest) -> NotificationType:
    """Resolve the notification-center category for a push request."""
    if req.notification_type is not None:
        return req.notification_type
    if req.violation_id is not None:
        return 'violation'
    return 'site_alert'


def _notification_deep_link(req: SiteNotifyRequest) -> str:
    """Return the app deep link shared by FCM data and stored records."""
    if req.deep_link:
        return req.deep_link
    if req.violation_id is not None:
        return f'/violations?violation_id={req.violation_id}'
    return '/violations'


def _notification_record_title(req: SiteNotifyRequest) -> str:
    """Build the stored notification title."""
    if req.title:
        return req.title
    return _translate_title(_notification_record_language) or 'Notification'


def _notification_record_body(req: SiteNotifyRequest) -> str:
    """Build the stored notification body in the default app language."""
    translated_lines = _translate_body_lines(
        req.body,
        _notification_record_language,
    )
    if not translated_lines:
        translated_lines = list(req.body.keys())

    summary = f'{req.site} - {req.stream_name}'
    if not translated_lines:
        return summary
    return summary + '\n' + '\n'.join(translated_lines)


def _notification_record_metadata(
    req: SiteNotifyRequest,
) -> dict[str, object]:
    """Build structured metadata for a stored site notification."""
    metadata = dict(req.metadata or {})
    metadata.setdefault('site', req.site)
    metadata.setdefault('stream_name', req.stream_name)
    metadata.setdefault('warnings', req.body)
    if req.image_path is not None:
        metadata.setdefault('image_path', req.image_path)
    if req.violation_id is not None:
        metadata.setdefault('violation_id', req.violation_id)
    return metadata


async def create_notification_records_for_users(
    req: SiteNotifyRequest,
    user_ids: list[int],
    db: AsyncSession,
) -> int:
    """Persist one notification-center record for each recipient user."""
    recipient_ids = list(dict.fromkeys(user_ids))
    if not recipient_ids:
        return 0

    records = [
        Notification(
            user_id=user_id,
            type=_notification_record_type(req),
            title=_notification_record_title(req),
            body=_notification_record_body(req),
            deep_link=_notification_deep_link(req),
            metadata_json=_notification_record_metadata(req),
        )
        for user_id in recipient_ids
    ]
    db.add_all(records)
    await db.commit()
    return len(records)


def _iter_push_tasks(
    req: SiteNotifyRequest,
    lang_to_tokens: DefaultDict[str, list[str]],
) -> Iterator[Awaitable[PushTaskResult]]:
    """
    Yield push tasks for sending notifications, batching tokens as needed.

    Args:
        req: Validated site notification request.
        lang_to_tokens: Mapping of language codes to device tokens.

    Returns:
        Awaitable tasks (each returns ``True`` on success, ``False``
        otherwise), yielded one batch at a time.
    """
    for lang, tokens in lang_to_tokens.items():
        for i in range(0, len(tokens), _fcm_batch_size):
            task = _build_push_task(req, lang, tokens[i:i + _fcm_batch_size])
            if task is not None:
                yield task


def _notification_data(req: SiteNotifyRequest) -> dict[str, str]:
    """Build stable FCM data fields for notification navigation.

    Args:
        req: Validated notification request.

    Returns:
        String-only FCM data payload.
    """
    return {
        'navigate': 'violation_list_page',
        'violation_id': str(req.violation_id or ''),
        'deep_link': _notification_deep_link(req),
        'type': _notification_record_type(req),
    }


def _build_push_task(
    req: SiteNotifyRequest,
    lang: str,
    tokens: list[str],
) -> Awaitable[PushTaskResult] | None:
    """Build one FCM batch task for a supported language.

    Args:
        req: Validated notification request.
        lang: Language code for the target tokens.
        tokens: Device tokens in this batch.

    Returns:
        Awaitable send task, or None when the language/body cannot be sent.
    """
    if not tokens:
        return None

    language = normalize_language(lang)
    if language is None:
        print(
            f"FCM notification skipped: unsupported language {lang!r}, "
            f"tokens: {len(tokens)}",
        )
        return None

    title: str = _translate_title(language)
    translated_lines: list[str] = _translate_body_lines(req.body, language)
    if not title or not translated_lines:
        print(
            'FCM notification skipped: no translated notification lines '
            f"for language {language}, body keys: {list(req.body.keys())}",
        )
        return None

    body: str = f"{req.site} - {req.stream_name}\n" + \
        '\n'.join(translated_lines)

    print(
        'FCM notification batch prepared: '
        f"lang={language}, tokens={len(tokens)}, "
        f"body_lines={len(translated_lines)}, "
        f"data_keys={sorted(_notification_data(req).keys())}",
    )

    return send_fcm_notification_service(
        device_tokens=tokens,
        title=title,
        body=body,
        image_path=req.image_path,
        data=_notification_data(req),
    )


def _push_task_succeeded(result: PushTaskResult) -> bool:
    """Return whether a push task result represents success.

    Args:
        result: Result returned by a push task.

    Returns:
        True when the push task succeeded.
    """
    return bool(result)


def _invalid_tokens_from_push_result(
    result: PushTaskResult,
) -> tuple[str, ...]:
    """Return invalid FCM tokens collected from a push result.

    Args:
        result: Result returned by a push task.

    Returns:
        Invalid tokens reported by Firebase.
    """
    if isinstance(result, FcmSendResult):
        return result.invalid_tokens
    return ()


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
    pending_batches: DefaultDict[str, list[str]] = defaultdict(list)

    for start in range(0, len(user_ids), _token_fetch_chunk_size):
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
                    if task is not None:
                        yield task

    for lang, tokens in pending_batches.items():
        if not tokens:
            continue
        task = _build_push_task(req, lang, list(tokens))
        if task is not None:
            yield task


def _build_push_tasks(
    req: SiteNotifyRequest,
    lang_to_tokens: DefaultDict[str, list[str]],
) -> list[Awaitable[PushTaskResult]]:
    """Build push tasks for compatibility with existing direct callers.

    Request handlers should prefer ``_iter_push_tasks_streaming`` with
    ``_execute_push_tasks_bounded_streaming`` so large recipient lists do not
    materialize every device token or FCM coroutine at once.

    Args:
        req: Validated notification request.
        lang_to_tokens: Tokens grouped by language.

    Returns:
        Awaitable FCM batch send tasks.
    """
    return list(_iter_push_tasks(req, lang_to_tokens))


async def _execute_push_tasks_bounded(
    push_tasks: Iterable[Awaitable[PushTaskResult]],
    timeout: float = 30.0,
    max_concurrency: int = _fcm_max_concurrency,
    invalid_token_handler: (
        Callable[[tuple[str, ...]], Awaitable[object]] | None
    ) = None,
) -> tuple[bool, int | None, int | None, str | None]:
    """Execute push tasks with bounded concurrency and aggregate counts.

    Args:
        push_tasks: Finite iterable of awaitable FCM batch send tasks.
        timeout: Maximum execution time in seconds.
        max_concurrency: Maximum number of active send tasks.
        invalid_token_handler: Optional callback receiving invalid tokens.

    Returns:
        Tuple of `(ok, total_batches, successful_batches, error_message)`.
    """
    pending: set[asyncio.Future[PushTaskResult]] = set()

    async def run_window() -> tuple[int, int]:
        """Run the bounded task window for a finite iterable."""
        total_batches = 0
        successful_batches = 0
        invalid_tokens: set[str] = set()
        task_iter = iter(push_tasks)

        def schedule_next() -> bool:
            """Schedule the next task when the window has capacity."""
            try:
                awaitable = next(task_iter)
            except StopIteration:
                return False
            pending.add(asyncio.ensure_future(awaitable))
            return True

        for _ in range(max(1, max_concurrency)):
            if not schedule_next():
                break

        try:
            while pending:
                done, _ = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                pending.difference_update(done)
                for task in done:
                    total_batches += 1
                    result = await task
                    successful_batches += int(_push_task_succeeded(result))
                    invalid_tokens.update(
                        _invalid_tokens_from_push_result(result),
                    )
                while len(pending) < max(1, max_concurrency):
                    if not schedule_next():
                        break
        finally:
            for task in pending:
                task.cancel()

        if invalid_tokens and invalid_token_handler is not None:
            await invalid_token_handler(tuple(sorted(invalid_tokens)))

        return total_batches, successful_batches

    window = run_window()
    try:
        total, successful = await asyncio.wait_for(window, timeout=timeout)
        return True, total, successful, None
    except asyncio.TimeoutError:
        window.close()
        return False, None, None, 'FCM notification sending timed out.'
    except Exception:
        window.close()
        return False, None, None, 'internal_error'


async def _execute_push_tasks_bounded_streaming(
    push_tasks: AsyncIterable[Awaitable[PushTaskResult]],
    timeout: float = 30.0,
    max_concurrency: int = _fcm_max_concurrency,
    invalid_token_handler: (
        Callable[[tuple[str, ...]], Awaitable[object]] | None
    ) = None,
) -> tuple[bool, int | None, int | None, str | None]:
    """Execute streamed push tasks with bounded concurrency.

    The async iterable may fetch Redis chunks while producing tasks, so this
    executor pulls only enough batches to fill the concurrency window.

    Args:
        push_tasks: Async iterable that yields awaitable FCM batch send tasks.
        timeout: Maximum execution time in seconds.
        max_concurrency: Maximum number of active send tasks.
        invalid_token_handler: Optional callback receiving invalid tokens.

    Returns:
        Tuple of `(ok, total_batches, successful_batches, error_message)`.
    """
    pending: set[asyncio.Future[PushTaskResult]] = set()
    max_workers = max(1, max_concurrency)

    async def run_window() -> tuple[int, int]:
        """Run the bounded task window for a streaming iterable."""
        total_batches = 0
        successful_batches = 0
        invalid_tokens: set[str] = set()
        task_iter = push_tasks.__aiter__()
        exhausted = False

        async def schedule_next() -> bool:
            """Schedule the next streamed task when capacity is available."""
            nonlocal exhausted
            if exhausted:
                return False
            try:
                awaitable = await task_iter.__anext__()
            except StopAsyncIteration:
                exhausted = True
                return False
            pending.add(asyncio.ensure_future(awaitable))
            return True

        try:
            for _ in range(max_workers):
                if not await schedule_next():
                    break

            while pending:
                done, _ = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                pending.difference_update(done)
                for task in done:
                    total_batches += 1
                    result = await task
                    successful_batches += int(_push_task_succeeded(result))
                    invalid_tokens.update(
                        _invalid_tokens_from_push_result(result),
                    )
                while len(pending) < max_workers:
                    if not await schedule_next():
                        break
        finally:
            for task in pending:
                task.cancel()
            aclose = getattr(task_iter, 'aclose', None)
            if aclose is not None:
                await aclose()

        if invalid_tokens and invalid_token_handler is not None:
            await invalid_token_handler(tuple(sorted(invalid_tokens)))

        return total_batches, successful_batches

    window = run_window()
    try:
        total, successful = await asyncio.wait_for(window, timeout=timeout)
        return True, total, successful, None
    except asyncio.TimeoutError:
        window.close()
        return False, None, None, 'FCM notification sending timed out.'
    except Exception:
        window.close()
        return False, None, None, 'internal_error'


async def _execute_push_tasks(
    push_tasks: list[Awaitable[PushTaskResult]], timeout: float = 30.0,
) -> tuple[bool, list[PushTaskResult] | None, str | None]:
    """
    Execute push tasks with a timeout and return results.

    Args:
        push_tasks: List of awaitable tasks created by ``_build_push_tasks``.
        timeout: Maximum time in seconds to wait for all tasks to complete.

    Returns:
        A tuple ``(ok, results, error_message)`` where:
        - ``ok`` is ``True`` when execution completes without timeout or
          unexpected exception.
        - ``results`` is a list of booleans for each batch when ``ok`` is
          ``True``; otherwise ``None``.
        - ``error_message`` contains a user-safe message when ``ok`` is
          ``False``; otherwise ``None``.
    """
    try:
        results = list(
            await asyncio.wait_for(
                asyncio.gather(*push_tasks, return_exceptions=False),
                timeout=timeout,
            ),
        )
        return True, results, None
    except asyncio.TimeoutError:
        # Return a generic timeout message (safe to surface to clients).
        return False, None, 'FCM notification sending timed out.'
    except Exception:
        # Do not surface internal exception details to clients.
        # Return a generic error indicator; log details at the call site.
        return False, None, 'internal_error'
