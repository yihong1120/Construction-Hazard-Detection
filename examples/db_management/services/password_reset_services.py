from __future__ import annotations

import hashlib
import json
import logging
import secrets
from typing import Any

import httpx
from fastapi import HTTPException
from redis.asyncio import Redis
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from examples.auth.cache import PROJECT_PREFIX
from examples.auth.config import Settings
from examples.auth.models import User
from examples.auth.models import USER_STATUS_ACTIVE
from examples.auth.models import UserProfile
from examples.db_management.schemas.password_reset import (
    PasswordErrorResponse,
)
from examples.db_management.services.auth_services import (
    clear_login_guard_for_identifiers,
)
from examples.db_management.services.password_policy import (
    validate_password_minimum,
)

settings = Settings()
logger = logging.getLogger(__name__)

BREVO_SEND_EMAIL_URL = 'https://api.brevo.com/v3/smtp/email'
FORGOT_PASSWORD_RESPONSE = 'If the email exists, a reset link has been sent.'
PASSWORD_RESET_SUCCESS_RESPONSE = 'Password reset successfully.'
RESET_TOKEN_INVALID_RESPONSE = PasswordErrorResponse(
    code='reset_token_invalid',
    message='Reset token is invalid or expired.',
).model_dump(exclude_none=True)


def _hash_token(raw_token: str) -> str:
    """Hash a raw reset token before using it as a Redis key.

    Args:
        raw_token: Raw one-time password-reset token.

    Returns:
        Non-reversible token hash.
    """
    return hashlib.sha256(raw_token.encode('utf-8')).hexdigest()


def _hash_identifier(value: str) -> str:
    """Hash a rate-limit identifier so PII is absent from Redis keys.

    Args:
        value: Email address, IP address, or username.

    Returns:
        Non-reversible identifier hash.
    """
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _password_reset_key(token_hash: str) -> str:
    """Build the Redis key for a password-reset token.

    Args:
        token_hash: Non-reversible reset-token hash.

    Returns:
        Redis token-record key.
    """
    return f"password_reset:{token_hash}"


def _email_rate_key(email: str) -> str:
    """Build the password-reset rate-limit key for an email address.

    Args:
        email: Email address requesting a reset.

    Returns:
        Redis email rate-limit key.
    """
    return f"password_reset_rate:email:{_hash_identifier(email)}"


def _ip_rate_key(client_ip: str) -> str:
    """Build the password-reset rate-limit key for a client address.

    Args:
        client_ip: Requesting client address.

    Returns:
        Redis IP rate-limit key.
    """
    return f"password_reset_rate:ip:{_hash_identifier(client_ip)}"


def _user_cache_key(username: str) -> str:
    """Build the cached-user key for a username.

    Args:
        username: Account username.

    Returns:
        Redis user-cache key.
    """
    return f"{PROJECT_PREFIX}:user_cache:{username}"


def _build_reset_url(raw_token: str) -> str:
    """Build the public password-reset URL for a raw token.

    Args:
        raw_token: One-time reset token to place in the URL.

    Returns:
        Public password-reset URL.
    """
    public_url = settings.app_public_url.rstrip('/')
    return f"{public_url}/reset_password?token={raw_token}"


async def _increment_rate_limit(
    redis_pool: Redis,
    key: str,
    window_seconds: int,
) -> int:
    """Increment a Redis counter and ensure it has an expiry.

    Args:
        redis_pool: Redis connection holding the counter.
        key: Counter key to increment.
        window_seconds: Expiry window applied to a new counter.

    Returns:
        Counter value after incrementing.
    """
    current = int(await redis_pool.incr(key))
    if current == 1:
        await redis_pool.expire(key, window_seconds)
    return current


async def _enforce_forgot_password_rate_limits(
    email: str,
    client_ip: str | None,
    redis_pool: Redis,
) -> None:
    """Limit reset requests by email address and client IP.

    Args:
        email: Email address requesting a reset.
        client_ip: Optional requesting client address.
        redis_pool: Redis connection holding rate-limit state.

    Raises:
        HTTPException: If either rate limit is exceeded.
    """
    email_count = await _increment_rate_limit(
        redis_pool,
        _email_rate_key(email),
        settings.password_reset_email_rate_limit_seconds,
    )
    if email_count > 1:
        raise HTTPException(status_code=429, detail='Too many requests.')

    if not client_ip:
        return

    ip_count = await _increment_rate_limit(
        redis_pool,
        _ip_rate_key(client_ip),
        settings.password_reset_ip_rate_limit_window_seconds,
    )
    if ip_count > settings.password_reset_ip_rate_limit_max:
        raise HTTPException(status_code=429, detail='Too many requests.')


async def _find_user_by_email(
    email: str,
    db: AsyncSession,
) -> User | None:
    """Find an active user by profile email.

    Args:
        email: Email address to search for.
        db: Database session used to load the account.

    Returns:
        Active user, or ``None`` when unavailable.
    """
    return await db.scalar(
        select(User)
        .join(UserProfile, UserProfile.user_id == User.id)
        .where(
            func.lower(UserProfile.email) == email.lower(),
            User.status == USER_STATUS_ACTIVE,
        ),
    )


async def _send_password_reset_email(
    email: str,
    reset_url: str,
) -> None:
    """Send a password-reset email through Brevo.

    Args:
        email: Recipient email address.
        reset_url: Public one-time password-reset URL.
    """
    if not settings.brevo_api_key or not settings.mail_from:
        raise HTTPException(
            status_code=500,
            detail='Password reset email is not configured.',
        )

    payload: dict[str, Any] = {
        'sender': {
            'email': settings.mail_from,
            'name': settings.mail_from_name,
        },
        'to': [{'email': email}],
        'subject': 'Reset your password',
        'htmlContent': (
            '<p>Use the link below to reset your password.</p>'
            f'<p><a href="{reset_url}">Reset password</a></p>'
            '<p>This link expires soon. If you did not request this, '
            'you can ignore this email.</p>'
        ),
        'textContent': (
            'Use the following link to reset your password:\n'
            f"{reset_url}\n\n"
            'This link expires soon. If you did not request this, '
            'you can ignore this email.'
        ),
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                BREVO_SEND_EMAIL_URL,
                headers={
                    'accept': 'application/json',
                    'api-key': settings.brevo_api_key,
                    'content-type': 'application/json',
                },
                json=payload,
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:500] if exc.response else ''
        logger.warning(
            'Brevo password reset email rejected: status=%s body=%s',
            exc.response.status_code if exc.response else 'unknown',
            body,
        )
        raise HTTPException(
            status_code=502,
            detail='Failed to send password reset email.',
        ) from exc
    except httpx.HTTPError as exc:
        logger.warning(
            'Brevo password reset email request failed: %s',
            exc.__class__.__name__,
        )
        raise HTTPException(
            status_code=502,
            detail='Failed to send password reset email.',
        ) from exc


async def request_password_reset(
    email: str,
    db: AsyncSession,
    redis_pool: Redis,
    client_ip: str | None = None,
) -> dict[str, str]:
    """Create a one-time reset token and send it by e-mail if the user
    exists."""
    normalized_email = email.strip().lower()
    await _enforce_forgot_password_rate_limits(
        normalized_email,
        client_ip,
        redis_pool,
    )

    user = await _find_user_by_email(normalized_email, db)
    if user is None:
        return {'message': FORGOT_PASSWORD_RESPONSE}

    raw_token = secrets.token_urlsafe(48)
    token_hash = _hash_token(raw_token)
    await redis_pool.set(
        _password_reset_key(token_hash),
        json.dumps(
            {'user_id': user.id, 'email': normalized_email},
        ).encode('utf-8'),
        ex=settings.password_reset_token_ttl_seconds,
    )

    try:
        await _send_password_reset_email(
            normalized_email,
            _build_reset_url(raw_token),
        )
    except HTTPException:
        await redis_pool.delete(_password_reset_key(token_hash))
        raise
    return {'message': FORGOT_PASSWORD_RESPONSE}


async def reset_password(
    raw_token: str | None,
    new_password: str | None,
    db: AsyncSession,
    redis_pool: Redis,
) -> dict[str, str]:
    """Reset a user's password if the token is valid and unused.

    Args:
        raw_token: Raw one-time reset token.
        new_password: Requested replacement password.
        db: Database session used to update the account.
        redis_pool: Redis connection holding token and cache state.

    Returns:
        Confirmation message after the password is updated.

    Raises:
        HTTPException: If input, policy, token, or account validation fails.
    """
    if not raw_token:
        raise HTTPException(
            status_code=400,
            detail=RESET_TOKEN_INVALID_RESPONSE,
        )

    validate_password_minimum(new_password)

    token = raw_token.strip()
    if not token:
        raise HTTPException(
            status_code=400,
            detail=RESET_TOKEN_INVALID_RESPONSE,
        )

    redis_key = _password_reset_key(_hash_token(token))
    raw_payload = await redis_pool.getdel(redis_key)
    if raw_payload is None:
        raise HTTPException(
            status_code=400,
            detail=RESET_TOKEN_INVALID_RESPONSE,
        )

    payload = json.loads(raw_payload)
    user_id = int(payload['user_id'])
    email = payload['email']

    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=400,
            detail=RESET_TOKEN_INVALID_RESPONSE,
        )

    user.set_password(new_password)
    try:
        await db.commit()
    except Exception as exc:
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail=PasswordErrorResponse(
                code='database_error',
                message=f"Database error: {exc}",
            ).model_dump(exclude_none=True),
        ) from exc

    await redis_pool.delete(_user_cache_key(user.username))
    await clear_login_guard_for_identifiers(
        redis_pool,
        [user.username, email],
    )
    return {'message': PASSWORD_RESET_SUCCESS_RESPONSE}
