from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import datetime
from datetime import timezone
from typing import Any

import httpx
from fastapi import HTTPException
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from examples.auth.config import Settings
from examples.auth.models import User
from examples.auth.models import USER_STATUS_ACTIVE
from examples.auth.models import USER_STATUS_EMAIL_UNVERIFIED
from examples.auth.models import USER_STATUS_PENDING_ADMIN_APPROVAL
from examples.auth.models import USER_STATUS_REJECTED
from examples.auth.models import USER_STATUS_SUSPENDED
from examples.auth.models import UserProfile
from src.http_client_pool import get_application_http_client

settings = Settings()
logger = logging.getLogger(__name__)

BREVO_SEND_EMAIL_URL = 'https://api.brevo.com/v3/smtp/email'
VERIFICATION_SENT_RESPONSE = (
    'If the account requires verification, a verification email has been sent.'
)
VERIFY_EMAIL_SUCCESS_RESPONSE = 'Email verified successfully.'


def _now() -> datetime:
    """Return the current timezone-aware UTC time.

    Returns:
        Current UTC time for verification expiry and audit data.
    """
    return datetime.now(timezone.utc)


def _hash_token(raw_token: str) -> str:
    """Hash a raw verification token before storing or comparing it.

    Args:
        raw_token: Raw one-time token sent to the user.

    Returns:
        Non-reversible token hash.
    """
    return hashlib.sha256(raw_token.encode('utf-8')).hexdigest()


def _hash_identifier(value: str) -> str:
    """Hash an identifier before using it in a Redis key.

    Args:
        value: Email address or other identifier.

    Returns:
        Non-reversible identifier hash.
    """
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _resend_rate_key(email: str) -> str:
    """Build the short-window verification-resend key for an email.

    Args:
        email: Email address requesting a verification resend.

    Returns:
        Redis rate-limit key.
    """
    return f'email_verification_rate:email:{_hash_identifier(email)}'


def _resend_daily_rate_key(email: str) -> str:
    """Build the daily verification-resend key for an email.

    Args:
        email: Email address requesting a verification resend.

    Returns:
        Redis daily rate-limit key.
    """
    return f'email_verification_daily_rate:email:{_hash_identifier(email)}'


def _email_verification_key(token_hash: str) -> str:
    """Build the Redis key for an email-verification token.

    Args:
        token_hash: Non-reversible verification-token hash.

    Returns:
        Redis token-record key.
    """
    return f'email_verification:{token_hash}'


def _email_verification_used_key(token_hash: str) -> str:
    """Build the Redis key recording a consumed verification token.

    Args:
        token_hash: Non-reversible verification-token hash.

    Returns:
        Redis consumed-token marker key.
    """
    return f'email_verification_used:{token_hash}'


def _email_verification_user_key(user_id: int) -> str:
    """Build the Redis key mapping a user to an active token.

    Args:
        user_id: Database identifier of the user.

    Returns:
        Redis user-to-token mapping key.
    """
    return f'email_verification_user:{user_id}'


def _build_verify_url(raw_token: str) -> str:
    """Build the public verification URL for a raw token.

    Args:
        raw_token: One-time verification token to place in the URL.

    Returns:
        Public verification URL.
    """
    public_url = settings.app_public_url.rstrip('/')
    return f'{public_url}/verify-email?token={raw_token}'


async def _enforce_resend_rate_limit(
    email: str,
    redis_pool: Redis,
) -> None:
    """Enforce short-window and daily verification-resend limits.

    Args:
        email: Email address requesting a resend.
        redis_pool: Redis connection holding rate-limit counters.

    Raises:
        HTTPException: If either resend limit is exceeded.
    """
    key = _resend_rate_key(email)
    current = int(await redis_pool.incr(key))
    if current == 1:
        await redis_pool.expire(
            key,
            settings.email_verification_resend_rate_limit_seconds,
        )
    if current > 1:
        ttl = int(await redis_pool.ttl(key))
        retry_after = (
            ttl
            if ttl > 0
            else settings.email_verification_resend_rate_limit_seconds
        )
        raise HTTPException(
            status_code=429,
            detail={
                'code': 'verification_resend_rate_limited',
                'retry_after_seconds': retry_after,
            },
            headers={'Retry-After': str(retry_after)},
        )

    daily_key = _resend_daily_rate_key(email)
    daily_count = int(await redis_pool.incr(daily_key))
    if daily_count == 1:
        await redis_pool.expire(
            daily_key,
            settings.email_verification_daily_limit_window_seconds,
        )
    if daily_count > settings.email_verification_daily_limit:
        ttl = int(await redis_pool.ttl(daily_key))
        retry_after = (
            ttl
            if ttl > 0
            else settings.email_verification_daily_limit_window_seconds
        )
        raise HTTPException(
            status_code=429,
            detail={
                'code': 'verification_daily_limit_exceeded',
                'retry_after_seconds': retry_after,
            },
            headers={'Retry-After': str(retry_after)},
        )


async def _find_user_by_email(
    email: str,
    db: AsyncSession,
) -> User | None:
    """Find the user whose profile matches an email address.

    Args:
        email: Email address to search for.
        db: Database session used to load the user.

    Returns:
        Matching user, or ``None`` when absent.
    """
    return await db.scalar(
        select(User)
        .options(selectinload(User.profile))
        .join(UserProfile, UserProfile.user_id == User.id)
        .where(UserProfile.email == email.strip().lower()),
    )


def _profile_email(user: User) -> str:
    """Return the normalised email address from a user's profile.

    Args:
        user: User with a required loaded profile.

    Returns:
        Normalised email address.
    """
    profile = user.profile
    email = str(profile.email if profile else '').strip().lower()
    if not email:
        raise HTTPException(
            status_code=400,
            detail='User email is required for verification.',
        )
    return email


async def _delete_existing_token_for_user(
    user_id: int,
    redis_pool: Redis,
) -> None:
    """Remove any previous verification token belonging to a user.

    Args:
        user_id: Database identifier of the user.
        redis_pool: Redis connection holding token state.
    """
    index_key = _email_verification_user_key(user_id)
    existing_hash = await redis_pool.get(index_key)
    if existing_hash is not None:
        await redis_pool.delete(
            _email_verification_key(existing_hash.decode('ascii')),
        )
    await redis_pool.delete(index_key)


async def _create_email_verification_token(
    user: User,
    redis_pool: Redis,
) -> str:
    """Create and persist a replacement verification token for a user.

    Args:
        user: User receiving the verification token.
        redis_pool: Redis connection used to store token state.

    Returns:
        Raw one-time token to send by email.
    """
    await _delete_existing_token_for_user(user.id, redis_pool)
    raw_token = secrets.token_urlsafe(48)
    token_hash = _hash_token(raw_token)
    ttl = settings.email_verification_token_ttl_seconds
    await redis_pool.set(
        _email_verification_key(token_hash),
        str(user.id).encode('ascii'),
        ex=ttl,
    )
    await redis_pool.set(
        _email_verification_user_key(user.id),
        token_hash.encode('ascii'),
        ex=ttl,
    )
    return raw_token


async def _delete_token_by_raw_token(
    raw_token: str,
    redis_pool: Redis,
) -> None:
    """Delete the verification record associated with a raw token.

    Args:
        raw_token: Raw verification token whose records are removed.
        redis_pool: Redis connection holding token state.
    """
    token_hash = _hash_token(raw_token)
    await redis_pool.delete(_email_verification_key(token_hash))


async def _send_email_verification_email(
    email: str,
    username: str,
    verify_url: str,
) -> None:
    """Send a transactional verification link through Brevo.

    Args:
        email: Recipient email address.
        username: Account username used in message content.
        verify_url: Public one-time verification URL.
    """
    if not settings.brevo_api_key or not settings.mail_from:
        raise HTTPException(
            status_code=500,
            detail='Email verification is not configured.',
        )

    expires_hours = max(
        settings.email_verification_token_ttl_seconds // 3600,
        1,
    )
    payload: dict[str, Any] = {
        'sender': {
            'email': settings.mail_from,
            'name': settings.mail_from_name,
        },
        'to': [{'email': email}],
        'params': {
            'username': username,
            'verify_url': verify_url,
            'VERIFY_URL': verify_url,
            'expires_hours': expires_hours,
        },
    }

    if settings.brevo_email_verification_template_id > 0:
        payload['templateId'] = settings.brevo_email_verification_template_id
    else:
        payload.update({
            'subject': 'Verify your Visionnaire account',
            'htmlContent': (
                '<p>請點擊下方連結完成信箱驗證。</p>'
                f'<p><a href="{verify_url}">驗證信箱</a></p>'
                f'<p>此連結將於 {expires_hours} 小時後失效。</p>'
            ),
            'textContent': (
                '請使用以下連結完成信箱驗證：\n'
                f'{verify_url}\n\n'
                f'此連結將於 {expires_hours} 小時後失效。'
            ),
        })

    headers = {
        'accept': 'application/json',
        'api-key': settings.brevo_api_key,
        'content-type': 'application/json',
    }
    try:
        client = await get_application_http_client(
            'brevo-email',
            timeout=10.0,
        )
        if client is not None:
            response = await client.post(
                BREVO_SEND_EMAIL_URL,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
        else:
            async with httpx.AsyncClient(timeout=10.0) as ephemeral_client:
                response = await ephemeral_client.post(
                    BREVO_SEND_EMAIL_URL,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:500] if exc.response else ''
        logger.warning(
            'Brevo email verification rejected: status=%s body=%s',
            exc.response.status_code if exc.response else 'unknown',
            body,
        )
        raise HTTPException(
            status_code=502,
            detail='Failed to send verification email.',
        ) from exc
    except httpx.HTTPError as exc:
        logger.warning(
            'Brevo email verification request failed: %s',
            exc.__class__.__name__,
        )
        raise HTTPException(
            status_code=502,
            detail='Failed to send verification email.',
        ) from exc


async def send_signup_verification_email(
    user: User,
    redis_pool: Redis,
) -> dict[str, str]:
    """Create a one-time verification token and email it to a new user.

    Args:
        user: Newly registered user requiring verification.
        redis_pool: Redis connection used to store token state.

    Returns:
        Generic success message for the registration flow.
    """
    email = _profile_email(user)
    raw_token = await _create_email_verification_token(
        user,
        redis_pool,
    )
    try:
        await _send_email_verification_email(
            email,
            user.username,
            _build_verify_url(raw_token),
        )
    except HTTPException:
        await _delete_token_by_raw_token(raw_token, redis_pool)
        raise
    return {
        'message': VERIFICATION_SENT_RESPONSE,
        'code': 'verification_email_sent',
        'status': user.status,
    }


async def resend_verification_email(
    email: str,
    db: AsyncSession,
    redis_pool: Redis,
) -> dict[str, str]:
    """Send another verification email for an unverified account.

    Args:
        email: Account email address requesting a resend.
        db: Database session used to locate the account.
        redis_pool: Redis connection used for rate limits and token state.

    Returns:
        Generic response that does not disclose account existence.
    """
    normalized_email = email.strip().lower()
    await _enforce_resend_rate_limit(normalized_email, redis_pool)

    user = await _find_user_by_email(normalized_email, db)
    if (
        user is None
        or user.status != USER_STATUS_EMAIL_UNVERIFIED
        or user.email_verified_at is not None
    ):
        return {
            'message': VERIFICATION_SENT_RESPONSE,
            'code': 'verification_email_sent',
        }

    return await send_signup_verification_email(user, redis_pool)


async def verify_email_token(
    raw_token: str | None,
    db: AsyncSession,
    redis_pool: Redis,
) -> dict[str, str]:
    """Verify a one-time email token and advance account state.

    Args:
        raw_token: Raw one-time verification token.
        db: Database session used to update the user.
        redis_pool: Redis connection used to consume token state.

    Returns:
        Verification message and resulting account status.

    Raises:
        HTTPException: If the token is invalid, expired, or already used.
    """
    token_value = _required_email_token(raw_token)
    token_hash = _hash_token(token_value)
    raw_payload = await _consume_email_verification_payload(
        redis_pool,
        token_hash,
    )
    user_id = _email_verification_user_id(raw_payload)

    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=400,
            detail={'code': 'invalid_token', 'message': 'Invalid token.'},
        )

    _apply_email_verification_status(user)
    if user.status in {
        USER_STATUS_REJECTED,
        USER_STATUS_SUSPENDED,
    }:
        await db.commit()
        await _mark_email_token_used(redis_pool, token_hash)
        raise HTTPException(
            status_code=403,
            detail={
                'code': 'account_not_verifiable',
                'status': user.status,
            },
        )
    if user.status not in {
        USER_STATUS_ACTIVE,
        USER_STATUS_PENDING_ADMIN_APPROVAL,
    }:
        await db.commit()
        await _mark_email_token_used(redis_pool, token_hash)
        raise HTTPException(
            status_code=403,
            detail={'code': 'account_not_active', 'status': user.status},
        )

    await db.commit()
    await redis_pool.delete(_email_verification_user_key(user.id))
    await _mark_email_token_used(redis_pool, token_hash)
    return {
        'message': VERIFY_EMAIL_SUCCESS_RESPONSE,
        'code': 'email_verified',
        'status': user.status,
    }


def _required_email_token(raw_token: str | None) -> str:
    """Return a non-empty verification token or raise the public error.

    Args:
        raw_token: Optional token supplied by the client.

    Returns:
        Non-empty verification token.

    Raises:
        HTTPException: If the token is absent.
    """
    token_value = (raw_token or '').strip()
    if not token_value:
        raise HTTPException(
            status_code=400,
            detail={'code': 'invalid_token', 'message': 'Invalid token.'},
        )
    return token_value


async def _consume_email_verification_payload(
    redis_pool: Redis,
    token_hash: str,
) -> bytes:
    """Consume an active token or raise its used/expired error.

    Args:
        redis_pool: Redis connection holding token state.
        token_hash: Non-reversible verification-token hash.

    Returns:
        Raw persisted token payload.

    Raises:
        HTTPException: If the token is used, expired, or invalid.
    """
    raw_payload = await redis_pool.getdel(_email_verification_key(token_hash))
    if raw_payload is not None:
        return raw_payload
    if await redis_pool.get(_email_verification_used_key(token_hash)):
        raise HTTPException(
            status_code=400,
            detail={'code': 'token_used', 'message': 'Token already used.'},
        )
    raise HTTPException(
        status_code=400,
        detail={
            'code': 'invalid_or_expired_token',
            'message': 'Token is invalid or expired.',
        },
    )


def _email_verification_user_id(raw_payload: bytes) -> int:
    """Extract the account identifier from a consumed token payload.

    Args:
        raw_payload: Redis payload stored for an active verification token.

    Returns:
        Verified user's database identifier.

    Raises:
        HTTPException: If the payload does not contain a valid identifier.
    """
    return int(raw_payload)


def _apply_email_verification_status(user: User) -> None:
    """Set verification time and advance an unverified user once.

    Args:
        user: User whose email-verification state is updated.
    """
    if user.email_verified_at is None:
        user.email_verified_at = _now()
    if user.status == USER_STATUS_EMAIL_UNVERIFIED:
        user.status = USER_STATUS_PENDING_ADMIN_APPROVAL


async def _mark_email_token_used(redis_pool: Redis, token_hash: str) -> None:
    """Record token use for the remainder of its verification TTL.

    Args:
        redis_pool: Redis connection holding token state.
        token_hash: Non-reversible verification-token hash.
    """
    await redis_pool.set(
        _email_verification_used_key(token_hash),
        b'1',
        ex=settings.email_verification_token_ttl_seconds,
    )
