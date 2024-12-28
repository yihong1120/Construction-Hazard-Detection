from __future__ import annotations

import redis.asyncio as redis
from fastapi import Request


class RedisClient:
    """
    A class to represent a Redis client.
    """

    def __init__(self, url: str) -> None:
        """
        Initialises the RedisClient with the provided Redis server URL.

        Args:
            url (str): The URL of the Redis server.
        """
        self.url = url
        self.client: redis.Redis | None = None

    async def connect(self) -> redis.Redis:
        """
        Initialises (or retrieves) the asynchronous Redis client.

        Returns:
            redis.Redis: An asynchronous Redis client instance.

        Raises:
            redis.exceptions.ConnectionError:
                If the connection to the Redis server fails.
        """
        if not self.client:
            # Establish a connection to the Redis server
            #  with encoding and response decoding.
            self.client = await redis.from_url(
                self.url,
                encoding='utf-8',
                decode_responses=True,
            )
        return self.client

    async def close(self) -> None:
        """
        Closes the Redis connection if it is active.

        This method ensures proper cleanup by releasing resources and
        setting the client to `None`.
        """
        if self.client:
            await self.client.close()
            self.client = None


async def get_redis_pool(request: Request) -> redis.Redis:
    """
    Retrieves the Redis client from the application state.

    Args:
        request (Request): The FastAPI request object,
            which contains the application state.

    Returns:
        redis.Redis: An asynchronous Redis client instance.

    Raises:
        RuntimeError: If the RedisClient
            is not properly initialised in the application state.
    """
    redis_client: RedisClient = request.app.state.redis_client
    if not redis_client.client:
        # Establish a connection if the client is not yet connected.
        await redis_client.connect()

    # Raise an error if the client is still not connected.
    if redis_client.client is None:
        raise RuntimeError('Redis client is not connected.')

    return redis_client.client
