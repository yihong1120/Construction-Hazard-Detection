from __future__ import annotations

import datetime
from typing import Any
from uuid import uuid4

import jwt
from fastapi import HTTPException
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from examples.auth.cache import get_user_data
from examples.auth.cache import set_user_data
from examples.auth.config import Settings
from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import jwt_refresh
from examples.auth.models import Feature
from examples.auth.models import group_features_table
from examples.auth.models import User
from examples.db_management.schemas.auth import RefreshRequest
from examples.db_management.schemas.auth import UserLogin

# Configuration settings for JWT authentication
settings = Settings()
SECRET_KEY = settings.authjwt_secret_key
ALGORITHM = 'HS256'
SUPER_ADMIN = 'ChangDar'
ACCESS_TTL = datetime.timedelta(minutes=60)  # Access token expiry time
REFRESH_TTL = datetime.timedelta(days=30)    # Refresh token expiry time


async def _load_feature_names(
    db: AsyncSession,
    group_id: int | None,
) -> list[str]:
    """Retrieve feature names associated with a specified group.

    Args:
        db (AsyncSession): Database session for queries.
        group_id (Optional[int]): ID of the group.

    Returns:
        list[str]: List of feature names linked to the group.
    """
    if group_id is None:
        return []

    rows = await db.execute(
        select(Feature.feature_name)
        .join(
            group_features_table,
            Feature.id == group_features_table.c.feature_id,
        )
        .where(group_features_table.c.group_id == group_id),
    )

    return [r.feature_name for r in rows]


async def _authenticate(
    db: AsyncSession,
    username: str,
    password: str,
) -> User:
    """Authenticate user credentials and verify active status.

    Args:
        db (AsyncSession): Database session for queries.
        username (str): User's username.
        password (str): User's password.

    Returns:
        User: Authenticated user object.

    Raises:
        HTTPException: When credentials are incorrect or user is inactive.
    """
    user = await db.scalar(select(User).where(User.username == username))

    if not user or not await user.check_password(password):
        raise HTTPException(
            status_code=401, detail='Wrong username or password',
        )

    if not user.is_active:
        raise HTTPException(status_code=403, detail='User inactive')

    return user


async def verify_refresh_token(
    refresh_token: str,
    redis_pool: Redis,
) -> dict[str, Any]:
    """Verify and decode a JWT refresh token.

    Args:
        refresh_token (str): Refresh token to validate.
        redis_pool (Redis): Redis connection pool for caching.

    Returns:
        dict[str, Any]: Decoded token payload.

    Raises:
        HTTPException: If token is invalid, expired, or not recognised.
    """
    try:
        # Decode and verify JWT refresh token
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401, detail='Refresh token has expired',
        )
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail='Invalid refresh token')

    username: str | None = payload.get('subject', {}).get('username')
    if not username:
        raise HTTPException(
            status_code=401, detail='Invalid refresh token payload',
        )

    # Retrieve user's data from Redis cache
    user_data: dict | None = await get_user_data(redis_pool, username)
    if (
        not user_data
        or refresh_token not in user_data.get('refresh_tokens', [])
    ):
        raise HTTPException(
            status_code=401, detail='Refresh token not recognised',
        )

    return payload


async def login_user(
    payload: UserLogin,
    db: AsyncSession,
    redis_pool: Redis,
) -> dict[str, str | int | list[str]]:
    """Authenticate user, issue JWT tokens, and store session in Redis cache.

    Args:
        payload (UserLogin): Login credentials (username and password).
        db (AsyncSession): Database session.
        redis_pool (Redis): Redis connection pool for caching sessions.

    Returns:
        dict[str, str | int | list[str]]:
            Generated tokens and user-related details.
    """
    user = await _authenticate(db, payload.username, payload.password)

    # Retrieve or initialise user cache data
    cache = await get_user_data(redis_pool, user.username) or {
        'db_user': {
            'id': user.id,
            'username': user.username,
            'role': user.role,
            'group_id': user.group_id,
            'is_active': user.is_active,
        },
        'jti_list': [],
        'refresh_tokens': [],
    }

    # Load feature names for user's group
    feature_names = await _load_feature_names(db, user.group_id)
    cache['feature_names'] = feature_names

    # Generate JWT tokens
    new_jti = str(uuid4())
    access_token = jwt_access.create_access_token(
        subject={
            'username': user.username,
            'role': user.role,
            'jti': new_jti,
            'features': feature_names,
        },
        expires_delta=ACCESS_TTL,
    )
    refresh_token = jwt_refresh.create_access_token(
        subject={'username': user.username},
        expires_delta=REFRESH_TTL,
    )

    # Update cache and store in Redis
    cache['jti_list'].append(new_jti)
    cache['refresh_tokens'].append(refresh_token)
    await set_user_data(redis_pool, user.username, cache)

    return {
        'access_token': access_token,
        'refresh_token': refresh_token,
        'username': user.username,
        'role': user.role,
        'user_id': user.id,
        'group_id': user.group_id,
        'feature_names': feature_names,
    }


async def logout_user(
    refresh_token: str,
    authorization: str | None,
    redis_pool: Redis,
) -> None:
    """Invalidate user's tokens on logout.

    Args:
        refresh_token (str): Refresh token to invalidate.
        authorization (Optional[str]): JWT access token from request headers.
        redis_pool (Redis): Redis connection pool.
    """
    if not authorization:
        return

    parts = authorization.split()
    if len(parts) != 2:
        return

    try:
        # Decode access token without expiry validation
        payload = jwt.decode(
            parts[1],
            SECRET_KEY,
            algorithms=[ALGORITHM],
            options={'verify_exp': False},
        )
    except jwt.PyJWTError:
        return

    username = payload.get('username')
    jti = payload.get('jti')

    # Remove the tokens from Redis cache
    cache = await get_user_data(redis_pool, username)
    if not cache:
        return

    cache['jti_list'] = [x for x in cache.get('jti_list', []) if x != jti]
    cache['refresh_tokens'] = [
        x for x in cache.get('refresh_tokens', []) if x != refresh_token
    ]
    await set_user_data(redis_pool, username, cache)


async def refresh_tokens(
    payload: RefreshRequest,
    redis_pool: Redis,
) -> dict[str, str | list[str]]:
    """Issue new JWT tokens using a refresh token.

    Args:
        payload (RefreshRequest): Contains the refresh token.
        redis_pool (Redis): Redis connection pool.

    Returns:
        dict[str, str | list[str]] New JWT access and refresh tokens.

    Raises:
        HTTPException: If refresh token is invalid or missing.
    """
    old_refresh = payload.refresh_token
    if not old_refresh:
        raise HTTPException(status_code=401, detail='Missing refresh token')

    # Verify provided refresh token
    data = await verify_refresh_token(old_refresh, redis_pool)
    username = data['subject']['username']

    cache = await get_user_data(redis_pool, username)
    if not cache or old_refresh not in cache.get('refresh_tokens', []):
        raise HTTPException(status_code=401, detail='Refresh token invalid')

    # Remove old refresh token
    cache['refresh_tokens'].remove(old_refresh)

    # Generate new JWT tokens
    new_jti = str(uuid4())
    access_token = jwt_access.create_access_token(
        subject={
            'username': username,
            'role': cache['db_user']['role'],
            'jti': new_jti,
            'features': cache.get('feature_names', []),
        },
        expires_delta=ACCESS_TTL,
    )
    new_refresh = jwt_refresh.create_access_token(
        subject={'username': username},
        expires_delta=REFRESH_TTL,
    )

    # Update and store new tokens in Redis cache
    cache['jti_list'].append(new_jti)
    cache['refresh_tokens'].append(new_refresh)
    await set_user_data(redis_pool, username, cache)

    return {
        'access_token': access_token,
        'refresh_token': new_refresh,
        'feature_names': cache.get('feature_names', []),
    }
