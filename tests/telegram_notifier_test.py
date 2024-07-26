from __future__ import annotations

import unittest
from io import BytesIO
from unittest.mock import AsyncMock
from unittest.mock import patch

import numpy as np

from src.telegram_notifier import TelegramNotifier


class TestTelegramNotifier(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for the TelegramNotifier class methods.
    """

    telegram_notifier: TelegramNotifier

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the TelegramNotifier instance for tests.
        """
        cls.telegram_notifier = TelegramNotifier(
            bot_token='test_bot_token',
        )

    def test_init(self) -> None:
        """
        Test if the TelegramNotifier instance is initialised correctly.
        """
        self.assertEqual(self.telegram_notifier.bot_token, 'test_bot_token')

    @patch('telegram.Bot.send_message', new_callable=AsyncMock)
    async def test_send_notification_no_image(
        self,
        mock_send_message: AsyncMock,
    ) -> None:
        """
        Test sending a notification without an image.
        """
        mock_send_message.return_value = 'Message sent'
        chat_id: str = 'test_chat_id'
        message: str = 'Hello, Telegram!'
        response: str = await self.telegram_notifier.send_notification(
            chat_id,
            message,
        )
        self.assertEqual(response, 'Message sent')
        mock_send_message.assert_called_once_with(
            chat_id=chat_id, text=message,
        )

    @patch('telegram.Bot.send_photo', new_callable=AsyncMock)
    async def test_send_notification_with_image(
        self,
        mock_send_photo: AsyncMock,
    ) -> None:
        """
        Test sending a notification with an image.
        """
        mock_send_photo.return_value = 'Message sent'
        chat_id: str = 'test_chat_id'
        message: str = 'Hello, Telegram!'
        image: np.ndarray = np.zeros((100, 100, 3), dtype=np.uint8)
        response: str = await self.telegram_notifier.send_notification(
            chat_id, message, image=image,
        )
        self.assertEqual(response, 'Message sent')
        mock_send_photo.assert_called_once()
        args, kwargs = mock_send_photo.call_args
        self.assertEqual(kwargs['chat_id'], chat_id)
        self.assertEqual(kwargs['caption'], message)
        self.assertIsInstance(kwargs['photo'], BytesIO)


if __name__ == '__main__':
    unittest.main()
