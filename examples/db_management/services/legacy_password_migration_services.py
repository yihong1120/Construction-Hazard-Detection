"""One-time Keycloak migration of existing Visionnaire password hashes.

Keycloak owns credentials after the OIDC cutover, so Visionnaire never issues
an application token from these checks.  A custom Keycloak form authenticator
uses this private, loopback-only protocol only after its normal password check
failed.  When the existing Argon2 password is proved, Keycloak stores that
same password using its own credential provider and then calls ``complete`` to
disable the legacy verifier.

Neither endpoint returns, logs, stores, or replays a plaintext password.  The
short-lived migration token is an opaque post-verification acknowledgement,
not a credential.
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import logging
import re
import secrets
import time
from collections.abc import Mapping

from fastapi import HTTPException
from fastapi import Request
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.config import Settings
from examples.auth.models import User
from examples.auth.models import USER_STATUS_ACTIVE
from examples.auth.models import UserIdentity

settings = Settings()
logger = logging.getLogger(__name__)

_MAX_BODY_BYTES = 16 * 1024
_MAX_PASSWORD_BYTES = 1024
_MIGRATION_TOKEN_RE = re.compile(r'^[A-Za-z0-9_-]{43,128}$')
_KEYCLOAK_SUBJECT_RE = re.compile(r'^[A-Za-z0-9._:-]{1,255}$')
_HMAC_MAX_AGE_SECONDS = 30
_HMAC_CONTEXT = b'visionnaire:legacy-password-migration:v1.'
_DISABLED_PASSWORD_HASH = 'oauth_disabled:legacy-password-migrated'


def _require_enabled() -> None:
    """Keep the transitional credential bridge disabled by default."""
    if not settings.legacy_password_migration_enabled:
        raise HTTPException(status_code=404, detail='not_found')


def _is_loopback_request(request: Request) -> bool:
    """Accept migration passwords only from the local Keycloak process."""
    if request.client is None:
        return False
    try:
        return ipaddress.ip_address(request.client.host).is_loopback
    except ValueError:
        return request.client.host.lower() == 'localhost'


def _opaque_token() -> str:
    """Create a non-credential acknowledgement for the second phase."""
    return secrets.token_urlsafe(32)


def _migration_key(token: str) -> str:
    """Avoid retaining public acknowledgement values in Redis key names."""
    digest = hashlib.sha256(token.encode('ascii')).hexdigest()
    return f'legacy-password-migration:{digest}'


def legacy_password_migration_signature(timestamp: str, body: bytes) -> str:
    """Return the domain-separated HMAC for the Keycloak-only protocol."""
    digest = hmac.digest(
        settings.native_social_exchange_shared_secret.encode('utf-8'),
        _HMAC_CONTEXT + timestamp.encode('ascii') + b'.' + body,
        'sha256',
    )
    return digest.hex()


async def _read_signed_payload(
    request: Request,
    *,
    fields: frozenset[str],
) -> dict[str, str]:
    """Verify transport origin, HMAC freshness, and a strict JSON body."""
    _require_enabled()
    if not _is_loopback_request(request):
        raise HTTPException(status_code=403, detail='not_found')
    timestamp = request.headers.get('X-Visionnaire-Timestamp', '')
    signature = request.headers.get('X-Visionnaire-Legacy-Signature', '')
    if not timestamp.isdecimal():
        raise HTTPException(status_code=401, detail='not_found')
    if abs(time.time() - int(timestamp)) > _HMAC_MAX_AGE_SECONDS:
        raise HTTPException(status_code=401, detail='not_found')
    body = await request.body()
    if not body or len(body) > _MAX_BODY_BYTES:
        raise HTTPException(status_code=401, detail='not_found')
    expected = legacy_password_migration_signature(timestamp, body)
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=401, detail='not_found')
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(status_code=401, detail='not_found') from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != fields
        or any(not isinstance(value, str) for value in payload.values())
    ):
        raise HTTPException(status_code=401, detail='not_found')
    return payload


def _valid_keycloak_subject(value: str) -> bool:
    """Require the immutable Keycloak identifier, never a client username."""
    return bool(_KEYCLOAK_SUBJECT_RE.fullmatch(value))


async def _legacy_user_for_subject(
    db: AsyncSession,
    keycloak_subject: str,
) -> User | None:
    """Resolve the local user solely through the immutable OIDC link."""
    return await db.scalar(
        select(User)
        .join(UserIdentity, UserIdentity.user_id == User.id)
        .where(
            User.status == USER_STATUS_ACTIVE,
            UserIdentity.provider == settings.oidc_identity_provider,
            UserIdentity.provider_user_id == keycloak_subject,
        ),
    )


async def verify_legacy_password(
    request: Request,
    db: AsyncSession,
    redis: Redis,
) -> dict[str, object]:
    """Prove a legacy Argon2 password and issue a one-use completion token."""
    payload = await _read_signed_payload(
        request,
        fields=frozenset({'keycloak_subject', 'password'}),
    )
    keycloak_subject = payload['keycloak_subject']
    password = payload['password']
    if (
        not _valid_keycloak_subject(keycloak_subject)
        or not password
        or len(password.encode('utf-8')) > _MAX_PASSWORD_BYTES
    ):
        raise HTTPException(status_code=401, detail='not_found')
    user = await _legacy_user_for_subject(db, keycloak_subject)
    if user is None or str(user.password_hash).startswith('oauth_disabled:'):
        raise HTTPException(status_code=401, detail='not_found')
    if not await user.check_password(password):
        raise HTTPException(status_code=401, detail='not_found')

    migration_token = _opaque_token()
    await redis.set(
        _migration_key(migration_token),
        json.dumps(
            {
                'kind': 'legacy-password-migration',
                'keycloak_subject': keycloak_subject,
                'local_user_id': user.id,
            },
            separators=(',', ':'),
        ).encode('utf-8'),
        ex=settings.legacy_password_migration_ttl_seconds,
    )
    return {
        'migration_token': migration_token,
        'expires_in': settings.legacy_password_migration_ttl_seconds,
    }


async def complete_legacy_password_migration(
    request: Request,
    db: AsyncSession,
    redis: Redis,
) -> dict[str, str]:
    """Disable a legacy verifier only after Keycloak stored its credential."""
    payload = await _read_signed_payload(
        request,
        fields=frozenset({'keycloak_subject', 'migration_token'}),
    )
    keycloak_subject = payload['keycloak_subject']
    migration_token = payload['migration_token']
    if (
        not _valid_keycloak_subject(keycloak_subject)
        or not _MIGRATION_TOKEN_RE.fullmatch(migration_token)
    ):
        raise HTTPException(status_code=401, detail='not_found')
    raw_record = await redis.getdel(_migration_key(migration_token))
    try:
        record = json.loads(raw_record) if raw_record is not None else None
    except (TypeError, ValueError):
        record = None
    stored_user_id = record.get('local_user_id') if isinstance(
        record, Mapping,
    ) else None
    stored_subject = (
        record.get('keycloak_subject') if isinstance(record, Mapping) else None
    )
    if (
        not isinstance(record, Mapping)
        or record.get('kind') != 'legacy-password-migration'
        or not isinstance(stored_user_id, int)
        or not isinstance(stored_subject, str)
        or not hmac.compare_digest(stored_subject, keycloak_subject)
    ):
        raise HTTPException(status_code=401, detail='not_found')
    user = await _legacy_user_for_subject(db, keycloak_subject)
    if user is None or user.id != stored_user_id:
        raise HTTPException(status_code=401, detail='not_found')
    if not str(user.password_hash).startswith('oauth_disabled:'):
        user.password_hash = _DISABLED_PASSWORD_HASH
        await db.commit()
        logger.info(
            'Legacy password migration completed for local user id=%s',
            user.id,
        )
    return {'status': 'completed'}
