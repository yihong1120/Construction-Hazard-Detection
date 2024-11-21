from __future__ import annotations

import base64
import os
import re

import redis.asyncio as redis
from dotenv import load_dotenv
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

        # Connect to Redis (asynchronous)
        self.client = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            password=self.redis_password,
            decode_responses=False,
        )

        if not self.client.ping():
            raise SystemExit(
                'Exiting application due to Redis connection failure.',
            )

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
                    r'stream_frame:([\w\x80-\xFF]+)_([\w\x80-\xFF]+)', key,
                )
                if not match:
                    continue

                label, stream_name = match.groups()

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

        while True:
            cursor, keys = await self.client.scan(
                cursor=cursor, match=f"stream_frame:{label}_*",
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
                image = base64.b64encode(frame_data).decode('utf-8')
                updated_data.append(
                    {'key': key.split('_')[-1], 'image': image},
                )

        return updated_data


class Utils:
    """
    Contains utility methods for processing and sending frame data.
    """

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
