"""Native Google/Apple assertion exchange endpoints.

These endpoints are intentionally separate from the retired local social-login
handlers.  Their successful sign-in response is a Keycloak authorization URL,
not an application access token.
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Request
from fastapi import Security
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.models import User
from examples.auth.redis_pool import get_redis_pool
from examples.db_management.deps import get_current_user
from examples.db_management.schemas.auth import (
    NativeSocialEmailLinkConfirmRequest,
)
from examples.db_management.schemas.auth import (
    NativeSocialExchangeBeginRequest,
)
from examples.db_management.schemas.auth import (
    NativeSocialExchangeBeginResponse,
)
from examples.db_management.schemas.auth import (
    NativeSocialExchangeCompleteRequest,
)
from examples.db_management.schemas.auth import (
    NativeSocialExchangeCompleteResponse,
)
from examples.db_management.schemas.auth import NativeSocialLinkBeginRequest
from examples.db_management.schemas.auth import NativeSocialLinkBeginResponse
from examples.db_management.schemas.auth import NativeSocialLinkCompleteRequest
from examples.db_management.schemas.auth import NativeSocialLinkResponse
from examples.db_management.services.native_social_exchange_services import (
    begin_native_social_exchange,
)
from examples.db_management.services.native_social_exchange_services import (
    begin_native_social_link,
)
from examples.db_management.services.native_social_exchange_services import (
    complete_native_social_exchange,
)
from examples.db_management.services.native_social_exchange_services import (
    complete_native_social_link,
)
from examples.db_management.services.native_social_exchange_services import (
    confirm_native_social_email_link,
)
from examples.db_management.services.native_social_exchange_services import (
    redeem_keycloak_native_social_exchange,
)

router = APIRouter(prefix='/auth/native-social', tags=['native-social'])


@router.post('/exchanges', response_model=NativeSocialExchangeBeginResponse)
async def begin_exchange(
    payload: NativeSocialExchangeBeginRequest,
    request: Request,
    redis: Redis = Depends(get_redis_pool),
) -> NativeSocialExchangeBeginResponse:
    """Issue the nonce needed by an official Google or Apple SDK."""
    return await begin_native_social_exchange(payload, request, redis)


@router.post(
    '/exchanges/complete',
    response_model=NativeSocialExchangeCompleteResponse,
)
async def complete_exchange(
    payload: NativeSocialExchangeCompleteRequest,
    redis: Redis = Depends(get_redis_pool),
    db: AsyncSession = Depends(get_db),
) -> NativeSocialExchangeCompleteResponse:
    """Validate provider proof and return a standard Keycloak code URL."""
    return await complete_native_social_exchange(payload, redis, db)


@router.post('/links', response_model=NativeSocialLinkBeginResponse)
async def begin_link(
    payload: NativeSocialLinkBeginRequest,
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
    _me: User = Depends(get_current_user),
    redis: Redis = Depends(get_redis_pool),
) -> NativeSocialLinkBeginResponse:
    """Start linking after a fresh Keycloak login (``prompt=login``)."""
    return await begin_native_social_link(payload.provider, credentials, redis)


@router.post('/links/complete', response_model=NativeSocialLinkResponse)
async def complete_link(
    payload: NativeSocialLinkCompleteRequest,
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
    _me: User = Depends(get_current_user),
    redis: Redis = Depends(get_redis_pool),
    db: AsyncSession = Depends(get_db),
) -> NativeSocialLinkResponse:
    """Validate nonce-bound provider assertion and link stable subject."""
    return await complete_native_social_link(
        payload,
        credentials,
        redis,
        db,
        _me,
    )


@router.post(
    '/email-link-confirmations/complete',
    response_model=NativeSocialLinkResponse,
)
async def confirm_email_link(
    payload: NativeSocialEmailLinkConfirmRequest,
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
    _me: User = Depends(get_current_user),
    redis: Redis = Depends(get_redis_pool),
    db: AsyncSession = Depends(get_db),
) -> NativeSocialLinkResponse:
    """Link a verified social identity after a password/MFA confirmation."""
    return await confirm_native_social_email_link(
        payload,
        credentials,
        redis,
        db,
        _me,
    )


@router.post('/keycloak/redeem', include_in_schema=False)
async def redeem_for_keycloak(
    request: Request,
    redis: Redis = Depends(get_redis_pool),
) -> dict[str, str]:
    """Accept only the loopback HMAC call from Keycloak's authenticator."""
    return await redeem_keycloak_native_social_exchange(request, redis)
