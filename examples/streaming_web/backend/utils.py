from __future__ import annotations

import base64
import json
import os
import re
from typing import Any

import redis.asyncio as redis
from dotenv import load_dotenv
from fastapi import HTTPException
from fastapi import Request
from fastapi import WebSocket

# Load environment variables
load_dotenv()


class RedisManager:
    """
    Manages asynchronous Redis operations for fetching labels, keys,
    and image data.
    """

    def __init__(
        self,
        redis_host: str = '127.0.0.1',
        redis_port: int = 6379,
        redis_password: str = '',
        max_connections: int = 10,
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

        # Set up Redis connection pool
        self.pool = redis.ConnectionPool(
            host=self.redis_host,
            port=self.redis_port,
            password=self.redis_password,
            decode_responses=False,
            max_connections=max_connections,
        )
        self.client = redis.Redis(connection_pool=self.pool)

    async def get_labels(self) -> list[str]:
        """
        Fetches unique label and stream_name combinations from Redis keys.

        Returns:
            list[str]: A sorted list of unique label-stream_name combinations.
        """
        cursor = 0
        labels: set[str] = set()

        while True:
            # Scan Redis for keys and decode them to UTF-8
            cursor, keys = await self.client.scan(cursor=cursor)
            decoded_keys = [
                key.decode('utf-8', errors='ignore')
                for key in keys
            ]

            # Match and extract unique label and stream_name
            for key in decoded_keys:
                match = re.match(
                    r'stream_frame:([\w\x80-\xFF]+)\|([\w\x80-\xFF]+)', key,
                )
                if not match:
                    continue

                encoded_label, encoded_stream_name = match.groups()
                label = Utils.decode(encoded_label)
                if 'test' in label:
                    continue

                labels.add(label)

            if cursor == 0:  # Exit loop if scan cursor has reached the end
                break

        return sorted(labels)

    async def get_keys_for_label(self, label: str) -> list[str]:
        """
        Retrieves Redis keys that match a given label-stream_name pattern.

        Args:
            label (str): The label-stream_name combination to search for.

        Returns:
            list[str]: A list of Redis keys associated with the given label.
        """
        cursor = 0
        matching_keys: list[str] = []

        # Encode the label to ensure compatibility
        encoded_label = Utils.encode(label)

        while True:
            cursor, keys = await self.client.scan(
                cursor=cursor, match=f"stream_frame:{encoded_label}|*",
            )
            matching_keys.extend(
                key.decode('utf-8')
                for key in keys
                if key.decode('utf-8').startswith('stream_frame:')
            )

            if cursor == 0:  # Exit loop if scan cursor has reached the end
                break

        return sorted(matching_keys)

    async def fetch_latest_frames(
        self, last_ids: dict[str, str],
    ) -> list[dict[str, str]]:
        """
        Fetches only the latest frame for each Redis stream.

        Args:
            last_ids (dict[str, str]): A dictionary mapping stream names to
                their last read message ID.

        Returns:
            list[dict[str, str]]: A list of dictionaries
                with updated frame data for each stream.
        """
        updated_data = []

        for key, last_id in last_ids.items():
            messages = await self.client.xrevrange(key, count=1)
            if not messages:
                continue

            message_id, data = messages[0]
            last_ids[key] = message_id  # Update to the latest message ID
            frame_data = data.get(b'frame')
            if frame_data:
                stream_name = key.split('|')[-1]
                decoded_stream_name = Utils.decode(stream_name)
                image = base64.b64encode(frame_data).decode('utf-8')
                updated_data.append(
                    {'key': decoded_stream_name, 'image': image},
                )

        return updated_data

    async def fetch_latest_frame_for_key(
        self,
        redis_key: str,
        last_id: str,
    ) -> dict[str, str] | None:
        """
        Fetches the latest frame and warnings for a specific Redis key.

        Args:
            redis_key (str): The Redis key to fetch data for.
            last_id (str): The last read message ID.

        Returns:
            dict[str, str] | None: A dictionary with frame and warnings data,
            or None if no new data is found.
        """
        # Fetch the latest message for the specific Redis key
        messages = await self.client.xrevrange(redis_key, min=last_id, count=1)

        if not messages:
            return None

        # Extract the message ID and data
        message_id, data = messages[0]

        frame_data = data.get(b'frame')
        warnings_data = data.get(b'warnings')

        if frame_data:
            # Decode frame and warnings data
            image = base64.b64encode(frame_data).decode('utf-8')
            warnings = warnings_data.decode('utf-8') if warnings_data else ''

            return {
                'id': message_id.decode('utf-8'),
                'image': image,
                'warnings': warnings,
            }

        return None

    async def update_partial_config(self, key: str, value: Any) -> None:
        """
        Update a single key in the Redis configuration cache.
        """
        cached_config = await self.get_config_cache()
        cached_config[key] = value
        await self.set_config_cache(cached_config)

    async def get_partial_config(self, key: str) -> Any:
        """
        Retrieve a single key from the Redis configuration cache.
        """
        cached_config = await self.get_config_cache()
        return cached_config.get(key)

    async def delete_config_cache(self) -> None:
        """
        Clear the Redis configuration cache.
        """
        await self.client.delete('config_cache')

    async def get_config_cache(self) -> dict:
        """
        Retrieve the full configuration from Redis cache.
        """
        cached_config = await self.client.get('config_cache')
        if cached_config:
            return json.loads(cached_config)
        return {}

    async def set_config_cache(self, config: dict, ttl: int = 3600) -> None:
        """
        Save the full configuration to Redis cache.
        """
        await self.client.set('config_cache', json.dumps(config), ex=ttl)

    async def close(self):
        """
        Close the Redis connection pool.
        """
        await self.client.close()


class Utils:
    """
    Contains utility methods for processing and sending frame data.
    """

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

    @staticmethod
    def is_base64(value: str) -> bool:
        """
        Check if the string is a valid Base64 encoded string.

        Args:
            value (str): The string to check.

        Returns:
            bool: True if the string is Base64, False otherwise.
        """
        if not value or not isinstance(value, str):
            return False
        # Base64 strings must be a multiple of 4 in length
        if len(value) % 4 != 0:
            return False
        # Check if the string matches the Base64 regex pattern
        return re.fullmatch(r'^[A-Za-z0-9\-_]+={0,2}$', value) is not None

    @staticmethod
    def decode(value: str) -> str:
        """
        Decode a URL-safe Base64 string.

        Args:
            value (str): The encoded string to decode.

        Returns:
            str: The decoded value.
        """
        if not Utils.is_base64(value):
            return value  # Return the original value if it's not Base64
        return base64.urlsafe_b64decode(value.encode('utf-8')).decode('utf-8')

    @staticmethod
    async def send_frames(
        websocket: WebSocket,
        label: str,
        updated_data: list[dict[str, str]],
    ) -> None:
        """
        Sends the latest frames to the WebSocket client.

        Args:
            websocket (WebSocket): The WebSocket connection object.
            label (str): The label associated with the frames.
            updated_data (list[dict[str, str]]): The latest frames to be sent.
        """
        await websocket.send_json({
            'label': label,
            'images': updated_data,
        })

    @staticmethod
    def load_configuration(config_path: str) -> list[dict]:
        """
        Load a JSON configuration file and return its content as a list.

        Args:
            config_path (str): The path to the JSON configuration file.

        Returns:
            list[dict]: The configuration data as a list of dictionaries.
        """
        try:
            with open(config_path, encoding='utf-8') as file:
                content = file.read()
                return json.loads(content)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return []

    @staticmethod
    def save_configuration(config_path: str, data: list[dict]) -> None:
        """
        Save a list of dictionaries as JSON to a configuration file.

        Args:
            config_path (str): The path to the JSON configuration file.
            data (list[dict]): The data to save to the file

        Raises:
            Exception: If an error occurs while saving the configuration.
        """
        try:
            with open(config_path, mode='w', encoding='utf-8') as file:
                content = json.dumps(data, indent=4, ensure_ascii=False)
                file.write(content)
        except Exception as e:
            print(f"Error saving configuration: {e}")

    @staticmethod
    def verify_localhost(request: Request) -> None:
        """
        Verify that the request is made from localhost.

        Args:
            request (Request): The incoming HTTP request.

        Raises:
            HTTPException: If the request is not made from localhost.
        """
        if request.client.host not in ['127.0.0.1', '::1']:
            raise HTTPException(
                status_code=403,
                detail='Access is restricted to localhost only.',
            )

    @staticmethod
    def update_configuration(
        config_path: str, new_config: list[dict],
    ) -> list[dict]:
        """
        Update the configuration file with new data.

        Args:
            config_path (str): The path to the JSON configuration file.
            new_config (list[dict]): The new configuration data to add.

        Returns:
            list[dict]: The updated configuration data.

        Raises:
            ValueError: If the configuration file
                is not in the expected format.
        """
        current_config = Utils.load_configuration(config_path)

        if not isinstance(current_config, list):
            raise ValueError(
                'Invalid configuration format. Expected a list in JSON file.',
            )

        for new_item in new_config:
            # Check if the item already exists in the configuration
            existing_item = next(
                (
                    item for item in current_config if item['video_url']
                    == new_item['video_url']
                ),
                None,
            )
            if existing_item:
                # If the item exists, update it
                existing_item.update(new_item)
            else:
                # If the item does not exist, add it to the configuration
                current_config.append(new_item)

        # Save the updated configuration to the file
        Utils.save_configuration(config_path, current_config)
        return current_config
