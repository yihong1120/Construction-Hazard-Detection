from __future__ import annotations

import json
from typing import Any

from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from fastapi_jwt import JwtAccessBearer
from fastapi_jwt import JwtAuthorizationCredentials
from redis.asyncio import Redis

from .config import Settings

jwt_access = JwtAccessBearer(secret_key=Settings().authjwt_secret_key)


async def get_user_data(
    redis_pool: Redis,
    username: str,
) -> dict[str, Any] | None:
    """
    Retrieve user data from the Redis cache.

    Args:
        redis_pool (Redis): The Redis connection pool
            used to interact with the cache.
        username (str): The username to retrieve the cached data for.

    Returns:
        Optional[dict[str, Any]]: A dictionary containing user data if found,
        otherwise `None`.

    Example:
        >>> data = await get_user_data(redis_pool, "john_doe")
        >>> print(data)
        {"id": 123, "username": "john_doe", "role": "user"}
    """
    # Construct the Redis key for the user.
    key = f"user_cache:{username}"
    # Attempt to retrieve data from Redis.
    raw_data = await redis_pool.get(key)

    if raw_data is None:
        return None  # Return None if no data is found.

    # Parse and return the JSON data as a dictionary.
    return json.loads(raw_data)


async def set_user_data(
    redis_pool: Redis,
    username: str,
    data: dict[str, Any],
) -> None:
    """
    Store user data in the Redis cache.

    Args:
        redis_pool (Redis): The Redis connection pool
            used to interact with the cache.
        username (str): The username for which the data is being stored.
        data (dict[str, Any]): The user data to be cached.
    """
    # Construct the Redis key for the user.
    key = f"user_cache:{username}"
    # Serialise the data dictionary to JSON and store it in Redis.
    await redis_pool.set(key, json.dumps(data))


async def custom_rate_limiter(
    request: Request,
    credentials: JwtAuthorizationCredentials = Depends(jwt_access),
) -> int:
    """
    Custom rate limiter for different user roles.

    Args:
        request (Request): The incoming request.
        credentials (JwtAuthorizationCredentials): The JWT credentials.

    Returns:
        int: The remaining number of requests allowed.
    """
    # Extract the username and token jti from the JWT payload
    payload = credentials.subject
    username = payload.get('username')
    token_jti = payload.get('jti')

    # Ensure the username and token jti are valid strings
    if not isinstance(username, str) or not isinstance(token_jti, str):
        raise HTTPException(
            status_code=401, detail='Token is missing or invalid fields',
        )

    # Get the Redis connection pool from the request state
    redis_pool: Redis = request.app.state.redis_client.client

    user_data = await get_user_data(redis_pool, username)
    if not user_data:
        raise HTTPException(status_code=401, detail='No such user in Redis')

    # Check if the token jti is valid
    jti_list = user_data.get('jti_list', [])
    if token_jti not in jti_list:
        raise HTTPException(
            status_code=401, detail='Token jti is invalid or replaced',
        )

    # Extract the user role and construct the Redis key prefix
    role: str = payload.get('role', 'user')
    key_prefix: str = f"rate_limit:{role}:{username}:{request.url.path}"

    max_requests = 24 if role == 'guest' else 3000
    window_seconds = 86400 if role == 'guest' else 60

    # Increment the request count
    current_requests = await redis_pool.incr(key_prefix)
    ttl = await redis_pool.ttl(key_prefix)

    # Set the expiration only if not already set
    if ttl == -1:
        await redis_pool.expire(key_prefix, window_seconds)

    if current_requests > max_requests:
        raise HTTPException(status_code=429, detail='Rate limit exceeded')

    return max_requests - current_requests
