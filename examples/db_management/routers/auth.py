from __future__ import annotations

from fastapi import APIRouter
from fastapi import Body
from fastapi import Depends
from fastapi import Header
from fastapi import Request
from fastapi import Response
from fastapi.responses import RedirectResponse
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

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
from examples.db_management.services.web_auth_services import (
    apple_callback_redirect,
)
from examples.db_management.services.web_auth_services import (
    clear_web_refresh_cookie,
)
from examples.db_management.services.web_auth_services import (
    is_web_auth_request,
)
from examples.db_management.services.web_auth_services import (
    refresh_token_from_cookie,
)
from examples.db_management.services.web_auth_services import (
    reject_legacy_web_token_request,
)
from examples.db_management.services.web_auth_services import (
    set_web_refresh_cookie,
)
from examples.db_management.services.web_auth_services import (
    token_pair_response,
)

router = APIRouter(tags=['auth'])


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
    """Authenticate a password user and issue a token pair.

    Args:
        payload: Submitted password-login credentials.
        request: HTTP request used to select browser-cookie behaviour.
        response: HTTP response that may receive a refresh-token cookie.
        x_hcaptcha_bypass_key: Optional privileged hCaptcha bypass credential.
        db: Database session used to authenticate the account.
        redis: Redis connection used for token state and rate limiting.

    Returns:
        Issued access-token details; browser clients receive their refresh
            token
        in a secure cookie.

    Raises:
        HTTPException: If the request is a rejected legacy web-token request or
            authentication fails.
    """
    reject_legacy_web_token_request(request)
    use_web_cookie = is_web_auth_request(request)
    result: TokenPairData = await login_user(
        payload,
        db,
        redis,
        hcaptcha_bypass_key=x_hcaptcha_bypass_key,
        client_ip=request.client.host if request.client else None,
        hash_refresh_token=use_web_cookie,
        request=request,
    )
    if use_web_cookie:
        # Web clients keep refresh tokens in an HTTP-only cookie, never JSON.
        set_web_refresh_cookie(response, result['refresh_token'])
    return token_pair_response(
        result,
        omit_refresh_token=use_web_cookie,
    )


@router.post('/auth/verify-email', response_model=AuthMessageResponse)
async def verify_email(
    payload: VerifyEmailRequest,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis_pool),
) -> AuthMessageResponse:
    """Verify an email token and advance the signup lifecycle.

    Args:
        payload: Raw verification token submitted by the client.
        db: Database session used to update the account.
        redis: Redis connection used to consume the one-time token.

    Returns:
        Verification result message and resulting account status.

    Raises:
        HTTPException: If the token is invalid, expired, or already consumed.
    """
    result = await verify_email_token(payload.token, db, redis)
    return AuthMessageResponse.model_validate(result)


@router.post('/auth/resend-verification', response_model=AuthMessageResponse)
async def resend_email_verification(
    payload: ResendVerificationRequest,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis_pool),
) -> AuthMessageResponse:
    """Send a replacement verification email for an unverified account.

    Args:
        payload: Email address for the account requiring verification.
        db: Database session used to locate the account.
        redis: Redis connection used to enforce resend limits.

    Returns:
        Generic message describing the resend result.

    Raises:
        HTTPException: If the resend rate limit is exceeded.
    """
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
    """Authenticate or register a user with a verified Google identity token.

    Args:
        payload: Google token, optional profile claims, and legal consents.
        request: HTTP request used to select browser-cookie behaviour.
        response: HTTP response that may receive a refresh-token cookie.
        db: Database session used to resolve or create the account.
        redis: Redis connection used for token state and rate limiting.

    Returns:
        Issued access-token details; browser clients receive their refresh
            token
        in a secure cookie.

    Raises:
        HTTPException: If provider-token validation or account registration
            fails.
    """
    reject_legacy_web_token_request(request)
    use_web_cookie = is_web_auth_request(request)
    result = await login_with_google(
        payload.id_token,
        db,
        redis,
        email=payload.email,
        display_name=payload.display_name,
        device_lang=payload.device_lang,
        consent_payload=payload,
        hash_refresh_token=use_web_cookie,
        request=request,
    )
    if use_web_cookie:
        set_web_refresh_cookie(response, result['refresh_token'])
    return token_pair_response(
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
    """Authenticate or register a user with Sign in with Apple.

    Args:
        payload: Apple credentials, optional profile claims, and legal
            consents.
        request: HTTP request used to select browser-cookie behaviour.
        response: HTTP response that may receive a refresh-token cookie.
        db: Database session used to resolve or create the account.
        redis: Redis connection used for token state and rate limiting.

    Returns:
        Issued access-token details; browser clients receive their refresh
            token
        in a secure cookie.

    Raises:
        HTTPException: If Apple-token validation or account registration fails.
    """
    reject_legacy_web_token_request(request)
    use_web_cookie = is_web_auth_request(request)
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
        request=request,
    )
    if use_web_cookie:
        set_web_refresh_cookie(response, result['refresh_token'])
    return token_pair_response(
        result,
        omit_refresh_token=use_web_cookie,
    )


@router.api_route(
    '/auth/apple/callback',
    methods=['GET', 'POST'],
    include_in_schema=False,
)
async def apple_callback(request: Request) -> RedirectResponse:
    """Redirect Apple callback parameters to the native application.

    Args:
        request: Apple callback request containing query or form parameters.

    Returns:
        Redirect response targeting the native application's callback URI.
    """
    return await apple_callback_redirect(request)


@router.get('/auth/identities', response_model=IdentityListResponse)
async def get_identities(
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> IdentityListResponse:
    """Return provider login methods linked to the current user.

    Args:
        db: Database session used to load linked identities.
        me: Authenticated account that owns the identities.

    Returns:
        Linked provider identities and password-credential availability.
    """
    return await list_user_identities(me, db)


@router.post('/auth/identities/google/link', response_model=IdentityRead)
async def link_google(
    payload: GoogleAuthRequest,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> IdentityRead:
    """Link or refresh the current user's Google identity.

    Args:
        payload: Google identity token to validate and link.
        db: Database session used to store the identity.
        me: Authenticated account receiving the identity link.

    Returns:
        Persisted Google identity details.

    Raises:
        HTTPException: If the token is invalid or belongs to another user.
    """
    return await link_google_identity(me, payload.id_token, db)


@router.post('/auth/identities/apple/link', response_model=IdentityRead)
async def link_apple(
    payload: AppleAuthRequest,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> IdentityRead:
    """Link or refresh the current user's Apple identity.

    Args:
        payload: Apple credentials used to validate and link the identity.
        db: Database session used to store the identity.
        me: Authenticated account receiving the identity link.

    Returns:
        Persisted Apple identity details.

    Raises:
        HTTPException: If the Apple credentials are invalid or belong to
            another
            user.
    """
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
    """Unlink an external provider identity from the current user.

    Args:
        identity_id: Identifier of the linked identity to remove.
        db: Database session used to remove the identity.
        me: Authenticated account that owns the identity.

    Returns:
        Confirmation message for the unlink operation.

    Raises:
        HTTPException: If the identity does not belong to the current user or
            removing it would leave no authentication method.
    """
    return await unlink_identity(me, identity_id, db)


@router.post('/logout')
async def logout(
    request: Request,
    response: Response,
    payload: LogoutRequest | None = Body(default=None),
    authorization: str | None = Header(None),
    redis: Redis = Depends(get_redis_pool),
) -> dict[str, str]:
    """Invalidate a user session by revoking supplied JWT tokens.

    Args:
        request: HTTP request that may contain a refresh-token cookie.
        response: HTTP response whose refresh-token cookie is cleared.
        payload: Optional request body containing a refresh token.
        authorization: Optional access token from the authorisation header.
        redis: Redis connection used to revoke token state.

    Returns:
        Confirmation message after logout processing.
    """
    refresh_token = (
        payload.refresh_token
        if payload and payload.refresh_token
        else refresh_token_from_cookie(request)
    )
    await logout_user(refresh_token, authorization, redis)
    clear_web_refresh_cookie(response)
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
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis_pool),
) -> TokenPair:
    """Issue a replacement token pair using a valid refresh token.

    Args:
        request: HTTP request that may contain a refresh-token cookie.
        response: HTTP response that may receive the replacement cookie.
        payload: Optional request body containing a refresh token.
        redis: Redis connection used to rotate token state.

    Returns:
        Issued access-token details; browser clients receive their refresh
            token
        in a secure cookie.

    Raises:
        HTTPException: If the request type is disallowed or the refresh token
            cannot be rotated.
    """
    reject_legacy_web_token_request(request)
    cookie_refresh = refresh_token_from_cookie(request)
    body_refresh = payload.refresh_token if payload else None
    use_web_cookie = is_web_auth_request(request) or bool(cookie_refresh)
    refresh_token = cookie_refresh or body_refresh
    # Prefer the HTTP-only cookie so browser refresh tokens never need JSON.
    result: TokenPairData = await refresh_tokens(
        RefreshRequest(refresh_token=refresh_token),
        redis,
        hash_refresh_token=use_web_cookie,
        request=request,
        db=db,
    )
    if use_web_cookie:
        set_web_refresh_cookie(response, result['refresh_token'])
    return token_pair_response(
        result,
        omit_refresh_token=use_web_cookie,
    )
