from __future__ import annotations

import base64
import re
from typing import Any

from fastapi import WebSocket


class Utils:
    """
    A collection of static utility methods for:
    - Encoding/decoding URL-safe base64 strings
    - Sending frame data via WebSocket
    """

    @staticmethod
    def encode(value: str) -> str:
        """
        Encode a given string into a URL-safe Base64 format.

        Args:
            value (str): The string to be encoded.

        Returns:
            str: A URL-safe Base64-encoded representation of the input.
        """
        return base64.urlsafe_b64encode(value.encode('utf-8')).decode('utf-8')

    @staticmethod
    def is_base64(value: str) -> bool:
        """
        Determine whether a string is valid URL-safe Base64.

        Checks:
            1. The input must be a non-empty string.
            2. Its length must be divisible by 4.
            3. It must match the regex for Base64-URL-safe format.

        Args:
            value (str): The string to be validated.

        Returns:
            bool: True if the string is URL-safe Base64, False otherwise.
        """
        if not value or not isinstance(value, str):
            return False
        if len(value) % 4 != 0:
            return False
        return re.fullmatch(r'^[A-Za-z0-9\-_]+={0,2}$', value) is not None

    @staticmethod
    def decode(value: str) -> str:
        """
        Decode a URL-safe Base64 string if it is valid,
        otherwise return the original string.

        Args:
            value (str): The potentially Base64-encoded string.

        Returns:
            str: The decoded string if valid; the original input otherwise.
        """
        if not Utils.is_base64(value):
            return value
        return base64.urlsafe_b64decode(value.encode('utf-8')).decode('utf-8')

    @staticmethod
    async def send_frames(
        websocket: WebSocket,
        label: str,
        updated_data: list[dict[str, Any]],
    ) -> None:
        """
        Asynchronously send the latest frames to the WebSocket client.

        This function assumes the data is in a JSON-serialisable format.

        Args:
            websocket (WebSocket): The active WebSocket connection.
            label (str): A label describing the source of frames.
            updated_data (List[Dict[str, Any]]): A list of frame data.
        """
        await websocket.send_json({
            'label': label,
            'images': updated_data,
        })
