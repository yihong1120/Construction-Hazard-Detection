from __future__ import annotations

import asyncio
import logging
import os
import threading
from datetime import datetime
from datetime import timedelta
from typing import TypedDict

import aiohttp
import cloudinary.uploader
import numpy as np
from dotenv import load_dotenv

from src.notifiers.image_encoding import encode_png
from src.notifiers.image_records import ImageRecordStore


class InputData(TypedDict):
    """Type definition for input data."""
    message: str
    image: bytes | None


class ResultData(TypedDict):
    """Type definition for result data."""
    response_code: int


class LineMessenger:
    """
    A class for managing notifications sent via the LINE Messaging API
    and Cloudinary image upload.
    """

    def __init__(
        self,
        channel_access_token: str | None = None,
        image_record_file: str = 'config/image_records.json',
        check_interval_days: int = 1,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Initialise the LineMessenger instance.

        Args:
            channel_access_token (str | None):
                LINE Messaging API channel access token.
            image_record_file (str): Path to JSON file to store image records.
            check_interval_days (int): Number of days to check for old images.
            session: Lifespan-owned session reused for LINE HTTP requests.
        Raises:
            ValueError:
                If no channel access token is provided
                or found in environment variables.
        """
        load_dotenv()
        self.logger = logging.getLogger(__name__)
        self.channel_access_token: str | None = (
            channel_access_token or os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
        )

        if not self.channel_access_token:
            raise ValueError(
                'LINE_CHANNEL_ACCESS_TOKEN not provided '
                'or in environment variables.',
            )

        # Configure Cloudinary
        cloudinary.config(
            cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
            api_key=os.getenv('CLOUDINARY_API_KEY'),
            api_secret=os.getenv('CLOUDINARY_API_SECRET'),
        )

        # Set image record file and check interval
        self.image_record_file: str = image_record_file
        self.check_interval_days: int = check_interval_days
        self._session = session
        self._records_lock = threading.Lock()
        self._image_records = ImageRecordStore(
            image_record_file,
            self.logger,
        )

        # Load upload records
        self.image_records: dict[str, str] = self.load_image_records()

    # --------------------------------------------------------------------- #
    # Helper functions for image-record JSON
    # --------------------------------------------------------------------- #

    def load_image_records(self) -> dict[str, str]:
        """
        Load image records from JSON file.

        Returns:
            dict[str, str]: The image records dictionary.
        """
        return self._image_records.load()

    def save_image_records(self) -> None:
        """
        Save image records to JSON file.
        """
        self._image_records.save(self.image_records)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    async def push_message(
        self,
        recipient_id: str,
        message: str,
        image_bytes: bytes | None = None,
    ) -> int:
        """
        Send a message via LINE Messaging API, optionally with an image.

        Args:
            recipient_id (str): The recipient ID.
            message (str): The message text.
            image_bytes (bytes | None):
                Raw PNG/JPEG bytes to upload to Cloudinary.

        Returns:
            int: HTTP status code returned by LINE API.
        """
        headers: dict[str, str] = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {self.channel_access_token}",
        }

        messages: list[dict[str, str]] = [{'type': 'text', 'text': message}]
        msg_payload: dict[str, object] = {
            'to': recipient_id,
            'messages': messages,
        }

        public_id: str | None = None

        # ---------- Optional image ----------
        if image_bytes is not None:
            image_url, public_id = await self.upload_image_to_cloudinary(
                image_bytes,
            )
            if not image_url:
                raise ValueError('Failed to upload image to Cloudinary')

            messages.append(
                {
                    'type': 'image',
                    'originalContentUrl': image_url,
                    'previewImageUrl': image_url,
                },
            )

        # Direct scripts retain a short-lived fallback.  MCP passes a session
        # from its lifespan and therefore reuses DNS, TCP, and TLS state.
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                status = await self._post_line_message(
                    session,
                    headers,
                    msg_payload,
                )
        else:
            status = await self._post_line_message(
                self._session,
                headers,
                msg_payload,
            )

        if public_id:
            # Cloudinary and record-store operations are synchronous SDK/file
            # calls.  Keep them outside the MCP event loop.
            await asyncio.to_thread(
                self._record_image_upload_and_cleanup,
                public_id,
            )
        return status

    async def _post_line_message(
        self,
        session: aiohttp.ClientSession,
        headers: dict[str, str],
        payload: dict[str, object],
    ) -> int:
        """Post one LINE payload through the supplied session."""
        async with session.post(
            'https://api.line.me/v2/bot/message/push',
            headers=headers,
            json=payload,
        ) as response:
            response_text = await response.text()
            if response.status == 200:
                self.logger.info('LINE message sent successfully')
            else:
                self.logger.error(
                    'LINE API returned %s: %s',
                    response.status,
                    response_text,
                )
            return response.status

    async def upload_image_to_cloudinary(
        self,
        image_data: bytes,
    ) -> tuple[str, str]:
        """
        Upload image bytes to Cloudinary and return (url, public_id).

        Args:
            image_data (bytes): The image data to upload.

        Returns:
            tuple[str, str]:
                The secure URL and public ID of the uploaded image.
        """
        try:
            loop = asyncio.get_event_loop()
            res = await loop.run_in_executor(
                None,
                lambda: cloudinary.uploader.upload(
                    image_data,
                    resource_type='image',
                ),
            )
            return res['secure_url'], res['public_id']
        except Exception as exc:
            self.logger.error('Failed to upload image to Cloudinary: %s', exc)
            return '', ''

    # --------------------------------------------------------------------- #
    # Image-record housekeeping
    # --------------------------------------------------------------------- #

    def record_image_upload(self, public_id: str) -> None:
        """
        Record upload time for a Cloudinary image.

        Args:
            public_id (str): The public ID of the image.
        """
        self.image_records[public_id] = datetime.now().isoformat()
        self.save_image_records()

    def _record_image_upload_and_cleanup(self, public_id: str) -> None:
        """Persist one upload and due cleanup with one locked file update."""
        with self._records_lock:
            self.image_records[public_id] = datetime.now().isoformat()
            if self._cleanup_is_due():
                self.image_records['last_checked'] = datetime.now().isoformat()
                self._delete_old_images()
            self.save_image_records()

    def _cleanup_is_due(self) -> bool:
        """Return whether the configured Cloudinary cleanup interval elapsed."""
        last_checked = self.image_records.get('last_checked')
        if not last_checked:
            return True
        try:
            last_checked_time = datetime.fromisoformat(last_checked)
        except ValueError:
            return True
        return (
            datetime.now() - last_checked_time
            > timedelta(days=self.check_interval_days)
        )

    def delete_old_images_with_interval(self) -> None:
        """
        Delete images older than 7 days, but perform the check only once
        every `check_interval_days`.
        """
        with self._records_lock:
            if self._cleanup_is_due():
                self.image_records['last_checked'] = datetime.now().isoformat()
                self._delete_old_images()
                self.save_image_records()

    def delete_old_images(self) -> None:
        """
        Delete Cloudinary images older than 7 days.
        """
        with self._records_lock:
            self._delete_old_images()
            self.save_image_records()

    def _delete_old_images(self) -> None:
        """Remove expired records; the caller owns locking and persistence."""
        now: datetime = datetime.now()
        expired: list[str] = [
            pid for pid, ts in self.image_records.items()
            if pid != 'last_checked'
            and now - datetime.fromisoformat(ts) > timedelta(days=7)
        ]

        for pid in expired:
            self.delete_image_from_cloudinary(pid)
            self.image_records.pop(pid, None)

    def delete_image_from_cloudinary(self, public_id: str) -> None:
        """
        Delete an image from Cloudinary via public_id.

        Args:
            public_id (str): The public ID of the image to delete.
        """
        try:
            res = cloudinary.uploader.destroy(public_id)
            if res.get('result') == 'ok':
                self.logger.info('Deleted Cloudinary image %s', public_id)
            else:
                self.logger.error(
                    'Failed to delete Cloudinary image %s: %s',
                    public_id,
                    res,
                )
        except Exception as exc:
            self.logger.error(
                'Error deleting Cloudinary image %s: %s',
                public_id,
                exc,
            )

# ------------------------------------------------------------------------- #
# Example usage
# ------------------------------------------------------------------------- #


async def main() -> None:
    """
    Example usage for sending a message with an image using LineMessenger.
    """
    channel_access_token: str = 'YOUR_CHANNEL_ACCESS_TOKEN'
    recipient_id: str = 'RECIPIENT_USER_ID'
    messenger = LineMessenger(channel_access_token=channel_access_token)

    message: str = 'Hello, LINE Messaging API!'

    # Build a 640×480 black PNG using the shared notification encoder.
    height, width = 480, 640
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame_bytes = encode_png(frame).getvalue()

    resp_code = await messenger.push_message(
        recipient_id=recipient_id,
        message=message,
        image_bytes=frame_bytes,
    )
    print(f"Response code: {resp_code}")


if __name__ == '__main__':
    asyncio.run(main())
