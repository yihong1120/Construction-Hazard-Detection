from __future__ import annotations

import json
from typing import Any

from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from redis.asyncio import Redis

from examples.auth.jwt_config import jwt_access

PROJECT_PREFIX: str = 'construction-hazard-detection'


async def get_user_data(
    redis_pool: Redis,
    username: str,
) -> dict[str, Any] | None:
    """
    Retrieve user data from Redis by username.

    Args:
        redis_pool (Redis): The asynchronous Redis client/connection.
        username (str): The username used to construct the Redis key.

    Returns:
        dict[str, Any] | None: A dictionary containing user data
            if the key exists, or None if no record is found.
    """
    key: str = f"{PROJECT_PREFIX}:user_cache:{username}"
    raw_data: bytes | None = await redis_pool.get(key)
    if raw_data is None:
        return None
    # Convert the raw JSON bytes into a Python dictionary
    return json.loads(raw_data)


async def set_user_data(
    redis_pool: Redis,
    username: str,
    data: dict[str, Any],
) -> None:
    """
    Store user data in Redis by username.

    Args:
        redis_pool (Redis): The asynchronous Redis client/connection.
        username (str): The username used to construct the Redis key.
        data (dict[str, Any]): The user data to be serialised and stored.

    Returns:
        None
    """
    key: str = f"{PROJECT_PREFIX}:user_cache:{username}"
    # Serialise the dictionary into JSON and store it in Redis
    await redis_pool.set(key, json.dumps(data))


async def custom_rate_limiter(
    request: Request,
    credentials=Depends(jwt_access),
) -> int:
    """
    Enforce rate-limiting based on the user's role.

    Args:
        request (Request): The incoming FastAPI request object.
        credentials: JWT credentials from the FastAPI dependency injection.
            Contains a 'subject' dict with fields like
            'username', 'role', and 'jti'.

    Returns:
        int: The number of remaining requests the user can make within the
             current time window.

    Raises:
        HTTPException: If the token is missing essential fields or jti,
                       if the user data is not found in Redis,
                       or if the rate limit has been exceeded.
    """
    payload: dict[str, Any] = credentials.subject
    username: str | None = payload.get('username')
    token_jti: str | None = payload.get('jti')

    # Basic checks to ensure the token payload is valid
    if not isinstance(username, str) or not isinstance(token_jti, str):
        raise HTTPException(
            status_code=401,
            detail='Token is missing or invalid fields',
        )

    # Acquire the Redis client from the app's state
    redis_pool: Redis = request.app.state.redis_client.client
    user_data: dict[str, Any] | None = await get_user_data(
        redis_pool,
        username,
    )
    if not user_data:
        raise HTTPException(
            status_code=401,
            detail='No such user in Redis',
        )

    jti_list: list[str] = user_data.get('jti_list', [])
    if token_jti not in jti_list:
        raise HTTPException(
            status_code=401,
            detail='Token jti is invalid or replaced',
        )

    role: str = payload.get('role', 'user')
    key_prefix: str = f"rate_limit:{role}:{username}:{request.url.path}"

    # Determine rate limit based on the user's role
    if role == 'guest':
        max_requests: int = 24
        window_seconds: int = 86400  # 24 hours
    else:
        max_requests = 3000
        window_seconds = 60  # 1 minute

    # Increment and check the current request count in Redis
    current_requests: int = await redis_pool.incr(key_prefix)
    ttl: int = await redis_pool.ttl(key_prefix)
    if ttl == -1:
        await redis_pool.expire(key_prefix, window_seconds)

    # If the user exceeds the allowed request quota
    if current_requests > max_requests:
        raise HTTPException(
            status_code=429,
            detail='Rate limit exceeded',
        )

    # Return how many requests remain
    return max_requests - current_requests
