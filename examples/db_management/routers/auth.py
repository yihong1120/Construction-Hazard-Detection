from __future__ import annotations

import os
from typing import cast
from typing import Literal
from urllib.parse import urlencode

from fastapi import APIRouter
from fastapi import Body
from fastapi import Depends
from fastapi import Header
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from fastapi.responses import RedirectResponse
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.config import Settings
from examples.auth.database import get_db
from examples.auth.models import User
from examples.auth.redis_pool import get_redis_pool
from examples.db_management.deps import get_current_user
from examples.db_management.schemas.auth import AppleAuthRequest
from examples.db_management.schemas.auth import AuthMessageResponse
from examples.db_management.schemas.auth import GoogleAuthRequest
from examples.db_management.schemas.auth import IdentityListResponse
from examples.db_management.schemas.auth import IdentityRead
from examples.db_management.schemas.auth import LogoutRequest
from examples.db_management.schemas.auth import RefreshRequest
from examples.db_management.schemas.auth import ResendVerificationRequest
from examples.db_management.schemas.auth import TokenPair
from examples.db_management.schemas.auth import TokenPairData
from examples.db_management.schemas.auth import UserLogin
from examples.db_management.schemas.auth import VerifyEmailRequest
from examples.db_management.services.auth_services import login_user
from examples.db_management.services.auth_services import logout_user
from examples.db_management.services.auth_services import refresh_tokens
from examples.db_management.services.email_verification_services import (
    resend_verification_email,
)
from examples.db_management.services.email_verification_services import (
    verify_email_token,
)
from examples.db_management.services.oauth_services import link_apple_identity
from examples.db_management.services.oauth_services import link_google_identity
from examples.db_management.services.oauth_services import list_user_identities
from examples.db_management.services.oauth_services import login_with_apple
from examples.db_management.services.oauth_services import login_with_google
from examples.db_management.services.oauth_services import unlink_identity

router = APIRouter(tags=['auth'])
settings = Settings()
LEGACY_WEB_TOKEN_ENDPOINTS_ENABLED = os.getenv(
    'LEGACY_WEB_TOKEN_ENDPOINTS_ENABLED',
    'false',
).lower() in {'1', 'true', 'yes', 'on'}


def _cookie_samesite() -> Literal['lax', 'strict', 'none']:
    return cast(
        Literal['lax', 'strict', 'none'],
        settings.web_refresh_cookie_samesite,
    )


def _is_web_auth_request(request: Request) -> bool:
    """Return whether refresh tokens should be handled by HttpOnly cookie."""
    platform = request.headers.get('x-client-platform', '').strip().lower()
    auth_mode = request.headers.get('x-auth-mode', '').strip().lower()
    if platform in {'web', 'flutter-web', 'browser'}:
        return True
    if auth_mode in {'cookie', 'web-cookie', 'web_cookie'}:
        return True
    return bool(
        request.headers.get('origin')
        or request.headers.get('sec-fetch-site'),
    )


def _reject_legacy_web_token_request(request: Request) -> None:
    if (
        _is_web_auth_request(request)
        and not LEGACY_WEB_TOKEN_ENDPOINTS_ENABLED
    ):
        raise HTTPException(
            status_code=410,
            detail='use_bff_auth_endpoint',
        )


def _set_web_refresh_cookie(response: Response, refresh_token: str) -> None:
    """Set the Web-only refresh token cookie."""
    response.set_cookie(
        key=settings.web_refresh_cookie_name,
        value=refresh_token,
        max_age=settings.web_refresh_cookie_max_age_seconds,
        httponly=True,
        secure=settings.web_refresh_cookie_secure,
        samesite=_cookie_samesite(),
        path=settings.web_refresh_cookie_path,
        domain=settings.web_refresh_cookie_domain or None,
    )


def _clear_web_refresh_cookie(response: Response) -> None:
    """Clear the Web-only refresh token cookie."""
    response.delete_cookie(
        key=settings.web_refresh_cookie_name,
        path=settings.web_refresh_cookie_path,
        domain=settings.web_refresh_cookie_domain or None,
        secure=settings.web_refresh_cookie_secure,
        samesite=_cookie_samesite(),
    )


def _refresh_token_from_cookie(request: Request) -> str | None:
    return request.cookies.get(settings.web_refresh_cookie_name)


def _token_pair_response_data(
    result: TokenPairData,
    *,
    omit_refresh_token: bool,
) -> TokenPair:
    data = dict(result)
    if omit_refresh_token:
        data.pop('refresh_token', None)
    return TokenPair.model_validate(data)


@router.post(
    '/login',
    response_model=TokenPair,
    response_model_exclude_none=True,
)
async def login(
    payload: UserLogin,
    request: Request,
    response: Response,
    x_hcaptcha_bypass_key: str | None = Header(
        None,
        alias='X-HCaptcha-Bypass-Key',
    ),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis_pool),
) -> TokenPair:
    """Authenticate user and return JWT tokens.

    Args:
        payload (UserLogin): User credentials containing username and password.
        db (AsyncSession): Database session.
        redis (Redis): Redis connection pool.

    Returns:
        TokenPair: Generated JWT access and refresh tokens.
    """
    _reject_legacy_web_token_request(request)
    use_web_cookie = _is_web_auth_request(request)
    result: TokenPairData = await login_user(
        payload,
        db,
        redis,
        hcaptcha_bypass_key=x_hcaptcha_bypass_key,
        client_ip=request.client.host if request.client else None,
        hash_refresh_token=use_web_cookie,
    )
    refresh_token = result.get('refresh_token')
    if use_web_cookie and isinstance(refresh_token, str):
        _set_web_refresh_cookie(response, refresh_token)
    return _token_pair_response_data(
        result,
        omit_refresh_token=use_web_cookie,
    )


@router.post('/auth/verify-email', response_model=AuthMessageResponse)
async def verify_email(
    payload: VerifyEmailRequest,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis_pool),
) -> AuthMessageResponse:
    """Verify an email verification token and advance signup status."""
    result = await verify_email_token(payload.token, db, redis)
    return AuthMessageResponse.model_validate(result)


@router.post('/auth/resend-verification', response_model=AuthMessageResponse)
async def resend_email_verification(
    payload: ResendVerificationRequest,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis_pool),
) -> AuthMessageResponse:
    """Send a new verification email for an unverified account."""
    result = await resend_verification_email(str(payload.email), db, redis)
    return AuthMessageResponse.model_validate(result)


@router.post(
    '/auth/google',
    response_model=TokenPair,
    response_model_exclude_none=True,
)
async def google_login(
    payload: GoogleAuthRequest,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis_pool),
) -> TokenPair:
    """Authenticate or register a user with a verified Google ID token."""
    _reject_legacy_web_token_request(request)
    use_web_cookie = _is_web_auth_request(request)
    result = await login_with_google(
        payload.id_token,
        db,
        redis,
        email=payload.email,
        display_name=payload.display_name,
        device_lang=payload.device_lang,
        consent_payload=payload,
        hash_refresh_token=use_web_cookie,
    )
    refresh_token = result.get('refresh_token')
    if use_web_cookie and isinstance(refresh_token, str):
        _set_web_refresh_cookie(response, refresh_token)
    return _token_pair_response_data(
        result,
        omit_refresh_token=use_web_cookie,
    )


@router.post(
    '/auth/apple',
    response_model=TokenPair,
    response_model_exclude_none=True,
)
async def apple_login(
    payload: AppleAuthRequest,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis_pool),
) -> TokenPair:
    """Authenticate or register a user with Sign in with Apple."""
    _reject_legacy_web_token_request(request)
    use_web_cookie = _is_web_auth_request(request)
    result = await login_with_apple(
        payload.identity_token,
        payload.authorization_code,
        db,
        redis,
        email=payload.email,
        given_name=payload.given_name,
        family_name=payload.family_name,
        nonce=payload.nonce,
        device_lang=payload.device_lang,
        consent_payload=payload,
        hash_refresh_token=use_web_cookie,
    )
    refresh_token = result.get('refresh_token')
    if use_web_cookie and isinstance(refresh_token, str):
        _set_web_refresh_cookie(response, refresh_token)
    return _token_pair_response_data(
        result,
        omit_refresh_token=use_web_cookie,
    )


@router.api_route(
    '/auth/apple/callback',
    methods=['GET', 'POST'],
    include_in_schema=False,
)
async def apple_callback(request: Request) -> RedirectResponse:
    """Redirect Apple callback parameters back to the native app."""
    params: list[tuple[str, str]] = [
        (key, value)
        for key, value in request.query_params.multi_items()
    ]
    if request.method == 'POST':
        form = await request.form()
        params.extend(
            (key, str(value))
            for key, value in form.multi_items()
        )

    query = urlencode(params)
    suffix = f'?{query}' if query else ''
    return RedirectResponse(
        (
            f'intent://callback{suffix}'
            '#Intent;package=com.changdar.visionnaire;'
            'scheme=signinwithapple;end'
        ),
        status_code=302,
    )


@router.get('/auth/identities', response_model=IdentityListResponse)
async def get_identities(
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> IdentityListResponse:
    """Return linked provider login methods for the current user."""
    return await list_user_identities(me, db)


@router.post('/auth/identities/google/link', response_model=IdentityRead)
async def link_google(
    payload: GoogleAuthRequest,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> IdentityRead:
    """Link or refresh the current user's Google login identity."""
    return await link_google_identity(me, payload.id_token, db)


@router.post('/auth/identities/apple/link', response_model=IdentityRead)
async def link_apple(
    payload: AppleAuthRequest,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> IdentityRead:
    """Link or refresh the current user's Apple login identity."""
    return await link_apple_identity(
        me,
        payload.identity_token,
        payload.authorization_code,
        db,
        nonce=payload.nonce,
    )


@router.delete('/auth/identities/{identity_id}')
async def unlink_provider_identity(
    identity_id: int,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> dict[str, str]:
    """Unlink a provider login identity owned by the current user."""
    return await unlink_identity(me, identity_id, db)


@router.post('/logout')
async def logout(
    request: Request,
    response: Response,
    payload: LogoutRequest | None = Body(default=None),
    authorization: str | None = Header(None),
    redis: Redis = Depends(get_redis_pool),
) -> dict[str, str]:
    """Invalidate user session by revoking JWT tokens.

    Args:
        payload (LogoutRequest): Contains the refresh token to revoke.
        authorization (Optional[str]): JWT access token from header.
        redis (Redis): Redis connection pool.

    Returns:
        dict[str, str]: Message indicating successful logout.
    """
    refresh_token = (
        payload.refresh_token
        if payload and payload.refresh_token
        else _refresh_token_from_cookie(request)
    )
    await logout_user(refresh_token, authorization, redis)
    _clear_web_refresh_cookie(response)
    return {'message': 'Logged out successfully.'}


@router.post(
    '/refresh',
    response_model=TokenPair,
    response_model_exclude_none=True,
)
async def refresh(
    request: Request,
    response: Response,
    payload: RefreshRequest | None = Body(default=None),
    redis: Redis = Depends(get_redis_pool),
) -> TokenPair:
    """Issue new JWT tokens using a valid refresh token.

    Args:
        payload (RefreshRequest): Contains the refresh token.
        redis (Redis): Redis connection pool.

    Returns:
        TokenPair: Newly issued access and refresh tokens.
    """
    _reject_legacy_web_token_request(request)
    cookie_refresh = _refresh_token_from_cookie(request)
    body_refresh = payload.refresh_token if payload else None
    use_web_cookie = _is_web_auth_request(request) or bool(cookie_refresh)
    refresh_token = cookie_refresh or body_refresh
    result: TokenPairData = await refresh_tokens(
        RefreshRequest(refresh_token=refresh_token),
        redis,
        hash_refresh_token=use_web_cookie,
    )
    new_refresh = result.get('refresh_token')
    if use_web_cookie and isinstance(new_refresh, str):
        _set_web_refresh_cookie(response, new_refresh)
    return _token_pair_response_data(
        result,
        omit_refresh_token=use_web_cookie,
    )
