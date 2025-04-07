from __future__ import annotations

import redis.asyncio as redis
from fastapi import Request
from fastapi import WebSocket


class RedisClient:
    """
    A class encapsulating an asynchronous Redis client.
    """

    def __init__(self, url: str) -> None:
        """
        Initialise the RedisClient with the provided server URL.

        Args:
            url (str): The URL of the Redis server,
                including authentication details.
        """
        self.url: str = url
        self.client: redis.Redis | None = None

    async def connect(self) -> redis.Redis:
        """
        Establish a connection to the Redis server.

        Returns:
            redis.Redis: An asyncio-compatible Redis client instance.
        """
        if not self.client:
            self.client = await redis.from_url(
                self.url,
                encoding='utf-8',
                # Return raw bytes (faster & more memory efficient)
                decode_responses=False,
            )
        return self.client

    async def close(self) -> None:
        """
        Close the Redis connection if it is currently active.
        """
        if self.client:
            await self.client.aclose()
            self.client = None


async def get_redis_pool(request: Request) -> redis.Redis:
    """
    Retrieve or initialise the Redis client for HTTP routes.

    Args:
        request (Request): The incoming FastAPI request object.

    Returns:
        redis.Redis: An asyncio-compatible Redis client.
    """
    redis_client: RedisClient = request.app.state.redis_client
    if not redis_client.client:
        await redis_client.connect()

    if redis_client.client is None:
        raise RuntimeError('Redis client is not connected.')

    return redis_client.client


async def get_redis_pool_ws(websocket: WebSocket) -> redis.Redis:
    """
    Retrieve or initialise the Redis client for WebSocket routes.

    Args:
        websocket (WebSocket): The active WebSocket connection object.

    Returns:
        redis.Redis: An asyncio-compatible Redis client.
    """
    redis_client: RedisClient = websocket.app.state.redis_client
    if not redis_client.client:
        await redis_client.connect()

    if redis_client.client is None:
        raise RuntimeError('Redis client is not connected.')

    return redis_client.client
