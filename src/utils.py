from __future__ import annotations

import logging
import os
from datetime import datetime

from redis import Redis
from watchdog.events import FileSystemEventHandler


class Utils:
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
            self.callback()


class RedisManager:
    def __init__(self):
        """
        Initialise the RedisManager by connecting to Redis.
        """
        self.redis_host: str = os.getenv('redis_host', 'localhost')
        self.redis_port: int = int(os.getenv('redis_port', '6379'))
        self.redis_password: str | None = os.getenv('redis_password', None)

        # Set decode_responses=False to allow bytes storage
        self.redis: Redis = Redis(
            host=self.redis_host,
            port=self.redis_port,
            password=self.redis_password,
            decode_responses=False,
        )

    def set(self, key: str, value: bytes) -> None:
        """
        Set a key-value pair in Redis.

        Args:
            key (str): The key under which to store the value.
            value (bytes): The value to store (in bytes).
        """
        try:
            self.redis.set(key, value)
        except Exception as e:
            logging.error(f"Error setting Redis key {key}: {str(e)}")

    def get(self, key: str) -> bytes | None:
        """
        Retrieve a value from Redis based on the key.

        Args:
            key (str): The key whose value needs to be retrieved.

        Returns:
            bytes | None: The value if found, None otherwise.
        """
        try:
            return self.redis.get(key)
        except Exception as e:
            logging.error(f"Error retrieving Redis key {key}: {str(e)}")
            return None

    def delete(self, key: str) -> None:
        """
        Delete a key from Redis.

        Args:
            key (str): The key to delete from Redis.
        """
        try:
            self.redis.delete(key)
        except Exception as e:
            logging.error(f"Error deleting Redis key {key}: {str(e)}")
