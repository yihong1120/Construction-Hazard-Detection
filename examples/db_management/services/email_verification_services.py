from __future__ import annotations

import hashlib
import json
import logging
import secrets
from datetime import datetime
from datetime import timezone
from typing import Any

import httpx
from fastapi import HTTPException
from redis.asyncio import Redis
from sqlalchemy import func
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

settings = Settings()
logger = logging.getLogger(__name__)

BREVO_SEND_EMAIL_URL = 'https://api.brevo.com/v3/smtp/email'
VERIFICATION_SENT_RESPONSE = (
    'If the account requires verification, a verification email has been sent.'
)
VERIFY_EMAIL_SUCCESS_RESPONSE = 'Email verified successfully.'


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _hash_token(raw_token: str) -> str:
    """Hash a raw email verification token before storing or comparing it."""
    return hashlib.sha256(raw_token.encode('utf-8')).hexdigest()


def _hash_identifier(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _resend_rate_key(email: str) -> str:
    return f'email_verification_rate:email:{_hash_identifier(email)}'


def _resend_daily_rate_key(email: str) -> str:
    return f'email_verification_daily_rate:email:{_hash_identifier(email)}'


def _email_verification_key(token_hash: str) -> str:
    return f'email_verification:{token_hash}'


def _email_verification_used_key(token_hash: str) -> str:
    return f'email_verification_used:{token_hash}'


def _email_verification_user_key(user_id: int) -> str:
    return f'email_verification_user:{user_id}'


def _build_verify_url(raw_token: str) -> str:
    public_url = settings.app_public_url.rstrip('/')
    return f'{public_url}/verify-email?token={raw_token}'


async def _enforce_resend_rate_limit(
    email: str,
    redis_pool: Redis,
) -> None:
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
    return await db.scalar(
        select(User)
        .options(selectinload(User.profile))
        .join(UserProfile, UserProfile.user_id == User.id)
        .where(func.lower(UserProfile.email) == email.lower()),
    )


def _profile_email(user: User) -> str:
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
    index_key = _email_verification_user_key(user_id)
    existing_hash = await redis_pool.get(index_key)
    if isinstance(existing_hash, bytes):
        existing_hash = existing_hash.decode('utf-8')
    if isinstance(existing_hash, str) and existing_hash:
        await redis_pool.delete(_email_verification_key(existing_hash))
    await redis_pool.delete(index_key)


async def _create_email_verification_token(
    user: User,
    email: str,
    redis_pool: Redis,
) -> str:
    await _delete_existing_token_for_user(user.id, redis_pool)
    raw_token = secrets.token_urlsafe(48)
    token_hash = _hash_token(raw_token)
    ttl = settings.email_verification_token_ttl_seconds
    await redis_pool.set(
        _email_verification_key(token_hash),
        json.dumps({'user_id': user.id, 'email': email}),
        ex=ttl,
    )
    await redis_pool.set(
        _email_verification_user_key(user.id),
        token_hash,
        ex=ttl,
    )
    return raw_token


async def _delete_token_by_raw_token(
    raw_token: str,
    redis_pool: Redis,
) -> None:
    token_hash = _hash_token(raw_token)
    await redis_pool.delete(_email_verification_key(token_hash))


async def _send_email_verification_email(
    email: str,
    username: str,
    verify_url: str,
) -> None:
    """Send a transactional email verification link through Brevo."""
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
    """Create a one-time verification token and email it to the new user."""
    email = _profile_email(user)
    raw_token = await _create_email_verification_token(
        user,
        email,
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
    """Send another verification email for an unverified account."""
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
    """Verify a one-time email token and advance the account state."""
    token_value = (raw_token or '').strip()
    if not token_value:
        raise HTTPException(
            status_code=400,
            detail={'code': 'invalid_token', 'message': 'Invalid token.'},
        )

    token_hash = _hash_token(token_value)
    redis_key = _email_verification_key(token_hash)
    raw_payload = await redis_pool.getdel(redis_key)

    if raw_payload is None:
        if await redis_pool.get(_email_verification_used_key(token_hash)):
            raise HTTPException(
                status_code=400,
                detail={
                    'code': 'token_used',
                    'message': 'Token already used.',
                },
            )
        raise HTTPException(
            status_code=400,
            detail={
                'code': 'invalid_or_expired_token',
                'message': 'Token is invalid or expired.',
            },
        )

    try:
        if isinstance(raw_payload, bytes):
            raw_payload = raw_payload.decode('utf-8')
        payload = json.loads(raw_payload)
        user_id = int(payload['user_id'])
    except (TypeError, ValueError, KeyError, json.JSONDecodeError):
        raise HTTPException(
            status_code=400,
            detail={'code': 'invalid_token', 'message': 'Invalid token.'},
        )

    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=400,
            detail={'code': 'invalid_token', 'message': 'Invalid token.'},
        )

    now = _now()
    if user.email_verified_at is None:
        user.email_verified_at = now

    if user.status == USER_STATUS_EMAIL_UNVERIFIED:
        user.status = USER_STATUS_PENDING_ADMIN_APPROVAL
    elif user.status in {USER_STATUS_REJECTED, USER_STATUS_SUSPENDED}:
        await db.commit()
        await redis_pool.set(
            _email_verification_used_key(token_hash),
            '1',
            ex=settings.email_verification_token_ttl_seconds,
        )
        raise HTTPException(
            status_code=403,
            detail={
                'code': 'account_not_verifiable',
                'status': user.status,
            },
        )
    elif user.status not in {
        USER_STATUS_ACTIVE,
        USER_STATUS_PENDING_ADMIN_APPROVAL,
    }:
        await db.commit()
        await redis_pool.set(
            _email_verification_used_key(token_hash),
            '1',
            ex=settings.email_verification_token_ttl_seconds,
        )
        raise HTTPException(
            status_code=403,
            detail={'code': 'account_not_active', 'status': user.status},
        )

    await db.commit()
    await redis_pool.delete(_email_verification_user_key(user.id))
    await redis_pool.set(
        _email_verification_used_key(token_hash),
        '1',
        ex=settings.email_verification_token_ttl_seconds,
    )
    return {
        'message': VERIFY_EMAIL_SUCCESS_RESPONSE,
        'code': 'email_verified',
        'status': user.status,
    }
