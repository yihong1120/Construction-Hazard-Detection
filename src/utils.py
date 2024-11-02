from __future__ import annotations

import asyncio
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
            expire_date = datetime.fromisoformat(expire_date_str)
            return datetime.now() > expire_date
        return False


class FileEventHandler(FileSystemEventHandler):
    """
    A class to handle file events.
    """

    def __init__(self, file_path: str, callback):
        """
        Initialises the FileEventHandler instance.

        Args:
            file_path (str): The path of the file to watch.
            callback (Callable): The function to call when file is modified.
        """
        self.file_path = file_path
        self.callback = callback

    def on_modified(self, event):
        """
        Called when a file is modified.

        Args:
            event (FileSystemEvent): The event object.
        """
        if event.src_path == self.file_path:
            # Run the callback function
            asyncio.run(self.callback())


class RedisManager:
    """
    A class to manage Redis operations.
    """

    def __init__(self):
        """
        Initialise the RedisManager by connecting to Redis.
        """
        self.redis_host: str = os.getenv('redis_host', 'localhost')
        self.redis_port: int = int(os.getenv('redis_port', 6379))
        self.redis_password: str | None = os.getenv('redis_password')

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
