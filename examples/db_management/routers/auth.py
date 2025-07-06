from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Header
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.database import get_db
from examples.auth.redis_pool import get_redis_pool
from examples.db_management.schemas.auth import LogoutRequest
from examples.db_management.schemas.auth import RefreshRequest
from examples.db_management.schemas.auth import TokenPair
from examples.db_management.schemas.auth import UserLogin
from examples.db_management.services.auth_services import login_user
from examples.db_management.services.auth_services import logout_user
from examples.db_management.services.auth_services import refresh_tokens

router = APIRouter(tags=['auth'])


@router.post('/login', response_model=TokenPair)
async def login(
    payload: UserLogin,
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
    result = await login_user(payload, db, redis)
    return TokenPair(**result)


@router.post('/logout')
async def logout(
    payload: LogoutRequest,
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
    await logout_user(payload.refresh_token, authorization, redis)
    return {'message': 'Logged out successfully.'}


@router.post('/refresh', response_model=TokenPair)
async def refresh(
    payload: RefreshRequest,
    redis: Redis = Depends(get_redis_pool),
) -> TokenPair:
    """Issue new JWT tokens using a valid refresh token.

    Args:
        payload (RefreshRequest): Contains the refresh token.
        redis (Redis): Redis connection pool.

    Returns:
        TokenPair: Newly issued access and refresh tokens.
    """
    result = await refresh_tokens(payload, redis)
    return TokenPair(**result)
