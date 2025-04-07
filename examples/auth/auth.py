from __future__ import annotations

import jwt
from fastapi import HTTPException
from redis.asyncio import Redis

from examples.auth.cache import get_user_data
from examples.auth.config import Settings

settings: Settings = Settings()

SECRET_KEY: str = settings.authjwt_secret_key
ALGORITHM: str = 'HS256'


async def verify_refresh_token(
    refresh_token: str,
    redis_pool: Redis,
) -> dict[str, str]:
    """
    Verify the provided JWT refresh token.

    Args:
        refresh_token (str): The JWT refresh token provided by the client.
        redis_pool (Redis): The asynchronous Redis connection object.

    Returns:
        dict[str, str]: The decoded JWT payload if valid and authorised.

    Raises:
        HTTPException: Raised if the token is:
            - expired (401: 'Refresh token has expired'),
            - invalid (401: 'Invalid refresh token'),
            - missing required claims (401: 'Invalid refresh token payload'),
            - not associated with the user in Redis
                (401: 'No user data in Redis'),
            - not recognised in the user's refresh token list
                (401: 'Refresh token not recognised').
    """
    try:
        # Decode the JWT token using the secret key and algorithm
        # specified in the settings
        payload: dict = jwt.decode(
            refresh_token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
        )
    except jwt.ExpiredSignatureError:
        # Token has passed its expiry time
        raise HTTPException(
            status_code=401,
            detail='Refresh token has expired',
        )
    except jwt.InvalidTokenError:
        # Token is malformed or tampered with
        raise HTTPException(
            status_code=401,
            detail='Invalid refresh token',
        )

    # Extract username from token payload
    username: str | None = payload.get('subject', {}).get('username')
    if not username:
        # Missing or improperly formatted username field
        raise HTTPException(
            status_code=401,
            detail='Invalid refresh token payload',
        )

    # Retrieve the user's cached data from Redis
    user_data: dict | None = await get_user_data(redis_pool, username)
    if not user_data:
        # User not found in Redis (token might be stale)
        raise HTTPException(
            status_code=401,
            detail='No user data in Redis',
        )

    # Check whether the provided token is still active for this user
    if refresh_token not in user_data.get('refresh_tokens', []):
        raise HTTPException(
            status_code=401,
            detail='Refresh token not recognised',
        )

    # Token is valid and recognised — return its payload
    return payload
