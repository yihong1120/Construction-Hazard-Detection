# Initialise a simple cache to store user data.
from __future__ import annotations

from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from fastapi_jwt import JwtAccessBearer
from fastapi_jwt import JwtAuthorizationCredentials

from .config import Settings

jwt_access = JwtAccessBearer(secret_key=Settings().authjwt_secret_key)

user_cache: dict = {}


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
    role: str = credentials.subject.get('role', 'user')
    username: str = credentials.subject.get('username', 'unknown')
    key_prefix: str = f"rate_limit:{role}:{username}:{request.url.path}"

    max_requests = 24 if role == 'guest' else 3000
    window_seconds = 86400 if role == 'guest' else 60
    redis_pool = request.app.state.redis_pool

    current_requests = await redis_pool.incr(key_prefix)
    ttl = await redis_pool.ttl(key_prefix)

    # Set the expiration only if not already set
    if ttl == -1:
        await redis_pool.expire(key_prefix, window_seconds)

    if current_requests > max_requests:
        raise HTTPException(status_code=429, detail='Rate limit exceeded')

    remaining_requests = max_requests - current_requests
    return remaining_requests
