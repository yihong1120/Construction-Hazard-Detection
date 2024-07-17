from __future__ import annotations

import unittest
from io import BytesIO
from unittest.mock import patch, AsyncMock

import numpy as np

from src.telegram_notifier import TelegramNotifier


class TestTelegramNotifier(unittest.IsolatedAsyncioTestCase):

    @classmethod
    def setUpClass(cls):
        cls.telegram_notifier = TelegramNotifier(bot_token='test_bot_token')

    def test_init(self):
        """Test if the TelegramNotifier instance is initialised correctly."""
        self.assertEqual(self.telegram_notifier.bot_token, 'test_bot_token')

    @patch('telegram.Bot.send_message', new_callable=AsyncMock)
    async def test_send_notification_no_image(self, mock_send_message):
        """Test sending a notification without an image."""
        mock_send_message.return_value = 'Message sent'
        chat_id = 'test_chat_id'
        message = 'Hello, Telegram!'
        response = await self.telegram_notifier.send_notification(chat_id, message)
        self.assertEqual(response, 'Message sent')
        mock_send_message.assert_called_once_with(
            chat_id=chat_id, text=message,
        )

    @patch('telegram.Bot.send_photo', new_callable=AsyncMock)
    async def test_send_notification_with_image(self, mock_send_photo):
        """Test sending a notification with an image."""
        mock_send_photo.return_value = 'Message sent'
        chat_id = 'test_chat_id'
        message = 'Hello, Telegram!'
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        response = await self.telegram_notifier.send_notification(
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
