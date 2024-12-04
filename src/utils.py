from __future__ import annotations

import asyncio
import base64
import logging
import os
from datetime import datetime

import redis.asyncio as redis
from watchdog.events import FileSystemEventHandler


class Utils:
    """
    A class to provide utility functions.
    """

    @staticmethod
    def is_expired(expire_date_str: str | None) -> bool:
        """
        Check if the given expire date string is expired.

        Args:
            expire_date_str (str | None): The expire date string
                in ISO 8601 format.

        Returns:
            bool: True if expired, False otherwise.
        """
        if expire_date_str:
            try:
                expire_date = datetime.fromisoformat(expire_date_str)
                return datetime.now() > expire_date
            except ValueError:
                # If the string cannot be parsed as a valid ISO 8601 date
                return False
        return False

    @staticmethod
    def encode(value: str) -> str:
        """
        Encode a value into a URL-safe Base64 string.

        Args:
            value (str): The value to encode.

        Returns:
            str: The encoded string.
        """
        return base64.urlsafe_b64encode(
            value.encode('utf-8'),
        ).decode('utf-8')


class FileEventHandler(FileSystemEventHandler):
    """
    A class to handle file events.
    """

    def __init__(self, file_path: str, callback, loop):
        """
        Initialises the FileEventHandler instance.

        Args:
            file_path (str): The path of the file to watch.
            callback (Callable): The function to call when file is modified.
            loop (asyncio.AbstractEventLoop): The asyncio event loop.
        """
        self.file_path = os.path.abspath(file_path)
        self.callback = callback
        self.loop = loop

    def on_modified(self, event):
        """
        Called when a file is modified.

        Args:
            event (FileSystemEvent): The event object.
        """
        event_path = os.path.abspath(event.src_path)
        if event_path == self.file_path:
            print(f"[DEBUG] Configuration file modified: {event_path}")
            asyncio.run_coroutine_threadsafe(
                # Ensure the callback is run in the loop
                self.callback(), self.loop,
            )


class RedisManager:
    """
    A class to manage Redis operations.
    """

    def __init__(
        self,
        redis_host: str = '127.0.0.1',
        redis_port: int = 6379,
        redis_password: str = '',
    ) -> None:
        """
        Initialises RedisManager with Redis configuration details.

        Args:
            redis_host (str): The Redis server hostname.
            redis_port (int): The Redis server port.
            redis_password (str): The Redis password for authentication.
        """
        self.redis_host: str = os.getenv('REDIS_HOST') or redis_host
        self.redis_port: int = int(os.getenv('REDIS_PORT') or redis_port)
        self.redis_password: str = os.getenv(
            'REDIS_PASSWORD',
        ) or redis_password

        # Create Redis connection
        self.redis = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            password=self.redis_password,
            decode_responses=False,
        )

    async def set(self, key: str, value: bytes) -> None:
        """
        Set a key-value pair in Redis.

        Args:
            key (str): The key under which to store the value.
            value (bytes): The value to store (in bytes).
        """
        try:
            await self.redis.set(key, value)
        except Exception as e:
            logging.error(f"Error setting Redis key {key}: {str(e)}")

    async def get(self, key: str) -> bytes | None:
        """
        Retrieve a value from Redis based on the key.

        Args:
            key (str): The key whose value needs to be retrieved.

        Returns:
            bytes | None: The value if found, None otherwise.
        """
        try:
            return await self.redis.get(key)
        except Exception as e:
            logging.error(f"Error retrieving Redis key {key}: {str(e)}")
            return None

    async def delete(self, key: str) -> None:
        """
        Delete a key from Redis.

        Args:
            key (str): The key to delete from Redis.
        """
        try:
            await self.redis.delete(key)
        except Exception as e:
            logging.error(f"Error deleting Redis key {key}: {str(e)}")

    async def add_to_stream(
        self,
        stream_name: str,
        data: dict,
        maxlen: int = 10,
    ) -> None:
        """
        Add data to a Redis stream with a maximum length.

        Args:
            stream_name (str): The name of the Redis stream.
            data (dict): The data to add to the stream.
            maxlen (int): The maximum length of the stream.
        """
        try:
            await self.redis.xadd(stream_name, data, maxlen=maxlen)
        except Exception as e:
            logging.error(
                f"Error adding to Redis stream {stream_name}: {str(e)}",
            )

    async def read_from_stream(
        self,
        stream_name: str,
        last_id: str = '0',
    ) -> list:
        """
        Read data from a Redis stream.

        Args:
            stream_name (str): The name of the Redis stream.
            last_id (str): The ID of the last read message.

        Returns:
            list: A list of messages from the stream.
        """
        try:
            return await self.redis.xread({stream_name: last_id})
        except Exception as e:
            logging.error(
                f"Error reading from Redis stream {stream_name}: {str(e)}",
            )
            return []

    async def delete_stream(self, stream_name: str) -> None:
        """
        Delete a Redis stream.

        Args:
            stream_name (str): The name of the Redis stream to delete.
        """
        try:
            await self.redis.delete(stream_name)
            logging.info(f"Deleted Redis stream: {stream_name}")
        except Exception as e:
            logging.error(
                f"Error deleting Redis stream {stream_name}: {str(e)}",
            )

    async def close_connection(self) -> None:
        """
        Close the Redis connection.
        """
        try:
            await self.redis.close()
            print('[INFO] Redis connection successfully closed.')
        except Exception as e:
            print(f"[ERROR] Failed to close Redis connection: {e}")
