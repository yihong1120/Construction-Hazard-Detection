from __future__ import annotations

import base64
import logging

import aiohttp
import cv2
import httpx
import numpy as np

from examples.mcp_server.config import get_env_var
from src.notifiers.broadcast_notifier import BroadcastNotifier
from src.notifiers.line_notifier_message_api import LineMessenger
from src.notifiers.messenger_notifier import MessengerNotifier
from src.notifiers.telegram_notifier import TelegramNotifier
from src.notifiers.wechat_notifier import WeChatNotifier

_broadcast_timeout = httpx.Timeout(10.0, connect=5.0)


class NotifyTools:
    """Tools for sending notifications via various platforms."""

    def __init__(self) -> None:
        """Initialise lazy notification clients."""
        self.logger = logging.getLogger(__name__)
        self._line_messenger: LineMessenger | None = None
        self._line_session: aiohttp.ClientSession | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._messenger_notifier: MessengerNotifier | None = None
        self._telegram_notifier: TelegramNotifier | None = None
        self._wechat_notifier: WeChatNotifier | None = None

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

            image_bytes = _decode_optional_image(image_base64)

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
            url = broadcast_url or get_env_var(
                'BROADCAST_URL',
                'http://localhost:8080/broadcast',
            )
            success = await BroadcastNotifier(
                url,
                client=self._broadcast_http_client(),
            ).broadcast_message(message)

            return {
                'success': success,
                'message': (
                    'Broadcast sent successfully'
                    if success
                    else 'Failed to send broadcast'
                ),
            }

        except httpx.HTTPError as exc:
            self.logger.warning(
                'Broadcast request failed error_type=%s',
                type(exc).__name__,
            )
            return {'success': False, 'message': 'Failed to send broadcast'}

    async def messenger_send(
        self,
        recipient_id: str,
        message: str,
        image_base64: str | None = None,
    ) -> dict:
        """Send a text or image notification through Facebook Messenger."""
        await self._ensure_messenger_notifier()
        assert self._messenger_notifier is not None
        status_code = await self._messenger_notifier.send_notification(
            recipient_id,
            message,
            image=_decode_rgb_image(_decode_optional_image(image_base64)),
        )
        return {
            'status_code': status_code,
            'success': 200 <= status_code < 300,
        }

    async def wechat_send(
        self,
        user_id: str,
        message: str,
        image_base64: str | None = None,
    ) -> dict:
        """Send a text or image notification through WeChat Work."""
        await self._ensure_wechat_notifier()
        assert self._wechat_notifier is not None
        response = await self._wechat_notifier.send_notification(
            user_id,
            message,
            image=_decode_rgb_image(_decode_optional_image(image_base64)),
        )
        return {
            'success': response.get('errcode') == 0,
            'response': response,
        }

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

            image_bytes = _decode_optional_image(image_base64)

            # Send message using TelegramNotifier API (send_notification)
            np_image = _decode_rgb_image(image_bytes)
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
            self._line_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
            )
            self._line_messenger = LineMessenger(session=self._line_session)
            self.logger.info('Initialised LINE messenger')

    def _broadcast_http_client(self) -> httpx.AsyncClient:
        """Return the lifespan-owned asynchronous broadcast client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=_broadcast_timeout,
                limits=httpx.Limits(
                    max_connections=10,
                    max_keepalive_connections=5,
                    keepalive_expiry=30,
                ),
            )
        return self._http_client

    async def _ensure_telegram_notifier(self) -> None:
        """Ensure the Telegram notifier is initialised."""
        if self._telegram_notifier is None:
            self._telegram_notifier = TelegramNotifier()
            self.logger.info('Initialised Telegram notifier')

    async def _ensure_messenger_notifier(self) -> None:
        """Ensure the Messenger feature shares the MCP HTTP transport."""
        if self._messenger_notifier is None:
            self._messenger_notifier = MessengerNotifier(
                client=self._broadcast_http_client(),
            )

    async def _ensure_wechat_notifier(self) -> None:
        """Ensure the WeChat Work feature shares the MCP HTTP transport."""
        if self._wechat_notifier is None:
            self._wechat_notifier = WeChatNotifier(
                client=self._broadcast_http_client(),
            )

    async def close(self) -> None:
        """Close lifespan-owned notification transports."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
        if self._line_session is not None:
            await self._line_session.close()
            self._line_session = None
        self._line_messenger = None
        self._messenger_notifier = None
        self._wechat_notifier = None


def _decode_optional_image(image_base64: str | None) -> bytes | None:
    """Decode a plain Base64 value or a browser data URL when provided."""
    if not image_base64:
        return None
    encoded = image_base64.split(',', 1)[-1]
    return base64.b64decode(encoded)


def _decode_rgb_image(image_bytes: bytes | None) -> np.ndarray | None:
    """Decode optional image bytes to the RGB NumPy format notifiers expect."""
    if not image_bytes:
        return None
    bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
