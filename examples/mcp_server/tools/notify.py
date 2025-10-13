from __future__ import annotations

import logging

import cv2
import numpy as np

from examples.mcp_server.config import get_env_var
from src.notifiers.broadcast_notifier import BroadcastNotifier
from src.notifiers.line_notifier_message_api import LineMessenger
from src.notifiers.telegram_notifier import TelegramNotifier


class NotifyTools:
    """Tools for sending notifications via various platforms."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._line_messenger: LineMessenger | None = None
        self._broadcast_notifier: BroadcastNotifier | None = None
        self._telegram_notifier: TelegramNotifier | None = None

    async def line_push(
        self,
        recipient_id: str,
        message: str,
        image_base64: str | None = None,
    ) -> dict:
        """Send a notification via the LINE Messaging API.

        Args:
            recipient_id: LINE user/group/room identifier.
            message: Text message to send.
            image_base64: Optional base64-encoded image.

        Returns:
            dict[str, Any]: Contains ``status_code`` and a ``success`` flag.
        """
        try:
            await self._ensure_line_messenger()
            assert self._line_messenger is not None

            # Convert base64 to bytes if provided
            image_bytes = None
            if image_base64:
                import base64
                if ',' in image_base64:
                    image_base64 = image_base64.split(
                        ',', 1,
                    )[1]  # Remove data URL prefix
                image_bytes = base64.b64decode(image_base64)

            # Send message
            status_code = await self._line_messenger.push_message(
                recipient_id=recipient_id,
                message=message,
                image_bytes=image_bytes,
            )

            return {
                'status_code': status_code,
                'success': status_code == 200,
                'message': (
                    'Message sent successfully'
                    if status_code == 200
                    else f"Failed with status {status_code}"
                ),
            }

        except Exception as e:
            self.logger.error(f"Failed to send LINE message: {e}")
            raise

    async def broadcast_send(
        self,
        message: str,
        broadcast_url: str | None = None,
    ) -> dict:
        """Send a broadcast notification.

        Args:
            message: Message to broadcast.
            broadcast_url: Optional broadcast URL (uses env var when omitted).

        Returns:
            dict[str, Any]: Contains a ``success`` flag and message.
        """
        try:
            await self._ensure_broadcast_notifier(broadcast_url)
            assert self._broadcast_notifier is not None

            # Send broadcast message
            success = self._broadcast_notifier.broadcast_message(message)

            return {
                'success': success,
                'message': (
                    'Broadcast sent successfully'
                    if success
                    else 'Failed to send broadcast'
                ),
            }

        except Exception as e:
            self.logger.error(f"Failed to send broadcast: {e}")
            raise

    async def telegram_send(
        self,
        chat_id: str,
        message: str,
        image_base64: str | None = None,
    ) -> dict:
        """Send a notification via the Telegram Bot API.

        Args:
            chat_id: Telegram chat identifier.
            message: Text message to send.
            image_base64: Optional base64-encoded image.

        Returns:
            dict[str, Any]: Contains a ``success`` flag and message.
        """
        try:
            await self._ensure_telegram_notifier()
            assert self._telegram_notifier is not None

            # Convert base64 to bytes if provided
            image_bytes = None
            if image_base64:
                import base64
                if ',' in image_base64:
                    image_base64 = image_base64.split(
                        ',', 1,
                    )[1]  # Remove data URL prefix
                image_bytes = base64.b64decode(image_base64)

            # Send message using TelegramNotifier API (send_notification)
            np_image = None
            if image_bytes:
                # Convert bytes to numpy array (RGB) for the notifier
                nparr = np.frombuffer(image_bytes, np.uint8)
                bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if bgr is not None:
                    np_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            try:
                await self._telegram_notifier.send_notification(
                    chat_id=chat_id,
                    message=message,
                    image=np_image,
                )
                success = True
            except Exception:
                success = False

            return {
                'success': success,
                'message': (
                    'Telegram message sent successfully'
                    if success
                    else 'Failed to send Telegram message'
                ),
            }

        except Exception as e:
            self.logger.error(f"Failed to send Telegram message: {e}")
            raise

    async def _ensure_line_messenger(self) -> None:
        """Ensure the LINE messenger is initialised."""
        if self._line_messenger is None:
            self._line_messenger = LineMessenger()
            self.logger.info('Initialised LINE messenger')

    async def _ensure_broadcast_notifier(
        self,
        broadcast_url: str | None,
    ) -> None:
        """Ensure the broadcast notifier is initialised."""
        if self._broadcast_notifier is None:
            url = broadcast_url or get_env_var(
                'BROADCAST_URL',
                'http://localhost:8080/broadcast',
            )
            self._broadcast_notifier = BroadcastNotifier(url)
            self.logger.info(
                f"Initialised broadcast notifier: {url}",
            )

    async def _ensure_telegram_notifier(self) -> None:
        """Ensure the Telegram notifier is initialised."""
        if self._telegram_notifier is None:
            self._telegram_notifier = TelegramNotifier()
            self.logger.info('Initialised Telegram notifier')
