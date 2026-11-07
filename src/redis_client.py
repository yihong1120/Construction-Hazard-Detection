from __future__ import annotations

import logging
from typing import cast

import redis.asyncio as redis

from src.settings.connections import RedisSettings


class RedisManager:
    """A class to manage Redis operations."""

    def __init__(
        self,
        redis_host: str | None = None,
        redis_port: int | None = None,
        redis_password: str | None = None,
    ) -> None:
        """Initialises RedisManager with Redis configuration details.

        Args:
            redis_host (str): The Redis server hostname.
            redis_port (int): The Redis server port.
            redis_password (str): The Redis password for authentication.
        """
        self.settings = RedisSettings.from_overrides(
            host=redis_host,
            port=redis_port,
            password=redis_password,
        )
        self.redis_host = self.settings.host
        self.redis_port = self.settings.port
        self.redis_password = self.settings.password.get_secret_value()

        # Create Redis connection
        self.redis = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            password=self.redis_password,
            decode_responses=False,
        )

    async def set(self, key: str, value: bytes) -> None:
        """Set a key-value pair in Redis.

        Args:
            key (str): The key under which to store the value.
            value (bytes): The value to store (in bytes).
        """
        try:
            await self.redis.set(key, value)
        except Exception as e:
            logging.error(f"Error setting Redis key {key}: {str(e)}")

    async def get(self, key: str) -> bytes | None:
        """Retrieve a value from Redis based on the key.

        Args:
            key (str): The key whose value needs to be retrieved.

        Returns:
            bytes | None: The value if found, None otherwise.
        """
        try:
            return cast(bytes | None, await self.redis.get(key))
        except Exception as e:
            logging.error(f"Error retrieving Redis key {key}: {str(e)}")
            return None

    async def delete(self, key: str) -> None:
        """Delete a key from Redis.

        Args:
            key (str): The key to delete from Redis.
        """
        try:
            await self.redis.delete(key)
        except Exception as e:
            logging.error(f"Error deleting Redis key {key}: {str(e)}")

    async def close_connection(self) -> None:
        """Close the Redis connection."""
        try:
            await self.redis.close()
            logging.info('[INFO] Redis connection successfully closed.')
        except Exception as e:
            logging.error(f"[ERROR] Failed to close Redis connection: {e}")
