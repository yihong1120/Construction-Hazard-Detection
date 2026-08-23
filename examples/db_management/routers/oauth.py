from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Request
from fastapi.responses import RedirectResponse
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.database import get_db
from examples.auth.models import User
from examples.auth.redis_pool import get_redis_pool
from examples.db_management.deps import get_current_user
from examples.db_management.schemas.oauth import MeResponse
from examples.db_management.schemas.oauth import OAuthTokenResponse
from examples.db_management.services.oauth_protocol_services import (
    authorize_native_app,
)
from examples.db_management.services.oauth_protocol_services import (
    current_user_profile,
)
from examples.db_management.services.oauth_protocol_services import (
    exchange_native_token,
)
from examples.db_management.services.oauth_protocol_services import (
    revoke_native_token,
)

router = APIRouter(tags=['oauth'])


@router.get('/oauth/authorize')
async def authorize(
    request: Request,
    response_type: str,
    client_id: str,
    redirect_uri: str,
    code_challenge: str,
    code_challenge_method: str,
    state: str = '',
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis_pool),
) -> RedirectResponse:
    """Create a PKCE authorisation code for a native OAuth client.

    Args:
        request: The authenticated browser request.
        response_type: Requested OAuth response type.
        client_id: Registered native client identifier.
        redirect_uri: Registered callback URI.
        code_challenge: PKCE code challenge.
        code_challenge_method: PKCE transformation method.
        state: Opaque client state returned to the callback.
        redis: Redis connection used to store the short-lived code.

    Returns:
        A redirect to the validated client callback.
    """
    return await authorize_native_app(
        request,
        response_type,
        client_id,
        redirect_uri,
        code_challenge,
        code_challenge_method,
        state,
        redis,
        db,
    )


@router.post('/oauth/token', response_model=OAuthTokenResponse)
async def token(
    request: Request,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis_pool),
) -> OAuthTokenResponse:
    """Exchange a native OAuth grant for a token pair.

    Args:
        request: Request containing form or JSON grant data.
        db: Database session used to resolve the token subject.
        redis: Redis connection storing grant state.

    Returns:
        The issued access and refresh token pair.
    """
    return await exchange_native_token(request, db, redis)


@router.get('/me', response_model=MeResponse)
async def me(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> MeResponse:
    """Return the authenticated user's public profile.

    Args:
        db: Database session used to load profile details.
        user: Authenticated user supplied by the access dependency.

    Returns:
        The user's public OAuth profile.
    """
    return await current_user_profile(user, db)


@router.post('/oauth/revoke', status_code=204)
async def revoke(
    request: Request,
    redis: Redis = Depends(get_redis_pool),
) -> None:
    """Revoke a native OAuth token when supplied by the client.

    Args:
        request: Request containing the optional token to revoke.
        redis: Redis connection containing token state.
    """
    await revoke_native_token(request, redis)
