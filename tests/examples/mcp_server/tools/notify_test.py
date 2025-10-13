from __future__ import annotations

import base64
import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from examples.mcp_server.tools.notify import NotifyTools


class LinePushTests(unittest.IsolatedAsyncioTestCase):
    """Tests for line_push method."""

    async def test_line_push_success_with_base64(self):
        """Should decode base64 and send successfully."""
        fake_line = AsyncMock()
        fake_line.push_message.return_value = 200
        with patch(
            'examples.mcp_server.tools.notify.LineMessenger',
            return_value=fake_line,
        ):
            tool = NotifyTools()
            img_b64 = base64.b64encode(b'abc').decode()
            res = await tool.line_push('uid', 'hello', img_b64)
        fake_line.push_message.assert_awaited_once()
        self.assertTrue(res['success'])
        self.assertEqual(res['status_code'], 200)
        self.assertIn('successfully', res['message'])

    async def test_line_push_with_data_url_prefix(self):
        """Should handle data URL prefix correctly."""
        fake_line = AsyncMock()
        fake_line.push_message.return_value = 400
        with patch(
            'examples.mcp_server.tools.notify.LineMessenger',
            return_value=fake_line,
        ):
            tool = NotifyTools()
            img = (
                'data:image/png;base64,'
                + base64.b64encode(b'abc').decode()
            )
            res = await tool.line_push('id', 'msg', img)
        self.assertFalse(res['success'])
        self.assertIn('Failed', res['message'])

    async def test_line_push_raises_and_logs(self):
        """Should log and re-raise on error."""
        with (
            patch(
                'examples.mcp_server.tools.notify.LineMessenger',
                side_effect=RuntimeError('boom'),
            ),
            patch(
                'examples.mcp_server.tools.notify.logging.getLogger',
            ) as mock_logger,
        ):
            tool = NotifyTools()
            tool.logger = mock_logger.return_value
            with self.assertRaises(RuntimeError):
                await tool.line_push('u', 'msg')
            tool.logger.error.assert_called_once()


class BroadcastSendTests(unittest.IsolatedAsyncioTestCase):
    """Tests for broadcast_send."""

    async def test_broadcast_send_success(self):
        """Should send broadcast successfully."""
        fake_broadcast = MagicMock()
        fake_broadcast.broadcast_message.return_value = True
        with patch(
            'examples.mcp_server.tools.notify.BroadcastNotifier',
            return_value=fake_broadcast,
        ):
            tool = NotifyTools()
            res = await tool.broadcast_send('message')
        self.assertTrue(res['success'])
        self.assertIn('successfully', res['message'])

    async def test_broadcast_send_failure(self):
        """Should return failure message."""
        fake_broadcast = MagicMock()
        fake_broadcast.broadcast_message.return_value = False
        with patch(
            'examples.mcp_server.tools.notify.BroadcastNotifier',
            return_value=fake_broadcast,
        ):
            tool = NotifyTools()
            res = await tool.broadcast_send('message')
        self.assertFalse(res['success'])
        self.assertIn('Failed', res['message'])

    async def test_broadcast_send_raises_and_logs(self):
        """Should log and raise on exception."""
        with (
            patch(
                'examples.mcp_server.tools.notify.BroadcastNotifier',
                side_effect=RuntimeError('boom'),
            ),
            patch(
                'examples.mcp_server.tools.notify.logging.getLogger',
            ) as mock_logger,
            patch(
                'examples.mcp_server.tools.notify.get_env_var',
                return_value='url',
            ),
        ):
            tool = NotifyTools()
            tool.logger = mock_logger.return_value
            with self.assertRaises(RuntimeError):
                await tool.broadcast_send('fail')
            tool.logger.error.assert_called_once()


class TelegramSendTests(unittest.IsolatedAsyncioTestCase):
    """Tests for telegram_send."""

    async def test_telegram_send_success_with_image(self):
        """Should decode image and send successfully."""
        fake_telegram = AsyncMock()
        fake_telegram.send_notification = AsyncMock()
        with (
            patch(
                'examples.mcp_server.tools.notify.TelegramNotifier',
                return_value=fake_telegram,
            ),
            patch(
                'examples.mcp_server.tools.notify.cv2.imdecode',
                return_value='fake_bgr',
            ),
            patch(
                'examples.mcp_server.tools.notify.cv2.cvtColor',
                return_value='fake_rgb',
            ),
        ):
            img_b64 = base64.b64encode(b'fake').decode()
            tool = NotifyTools()
            res = await tool.telegram_send('chat', 'msg', img_b64)
        fake_telegram.send_notification.assert_awaited_once()
        self.assertTrue(res['success'])
        self.assertIn('successfully', res['message'])

    async def test_telegram_send_imdecode_none(self):
        """Should handle imdecode returning None gracefully."""
        fake_telegram = AsyncMock()
        fake_telegram.send_notification = AsyncMock()
        with (
            patch(
                'examples.mcp_server.tools.notify.TelegramNotifier',
                return_value=fake_telegram,
            ),
            patch(
                'examples.mcp_server.tools.notify.cv2.imdecode',
                return_value=None,
            ),
        ):
            img_b64 = base64.b64encode(b'fake').decode()
            tool = NotifyTools()
            res = await tool.telegram_send('chat', 'msg', img_b64)
        self.assertTrue(res['success'])

    async def test_telegram_send_notification_failure(self):
        """Should return success=False when send_notification raises."""
        fake_telegram = AsyncMock()
        fake_telegram.send_notification = AsyncMock(
            side_effect=RuntimeError('network'),
        )
        with (
            patch(
                'examples.mcp_server.tools.notify.TelegramNotifier',
                return_value=fake_telegram,
            ),
            patch(
                'examples.mcp_server.tools.notify.cv2.imdecode',
                return_value='fake_bgr',
            ),
            patch(
                'examples.mcp_server.tools.notify.cv2.cvtColor',
                return_value='fake_rgb',
            ),
        ):
            tool = NotifyTools()
            img_b64 = base64.b64encode(b'x').decode()
            res = await tool.telegram_send('chat', 'msg', img_b64)
        self.assertFalse(res['success'])
        self.assertIn('Failed', res['message'])

    async def test_telegram_send_logs_and_reraises(self):
        """Should log and re-raise on outer exception."""
        with (
            patch(
                'examples.mcp_server.tools.notify.TelegramNotifier',
                side_effect=RuntimeError('boom'),
            ),
            patch(
                'examples.mcp_server.tools.notify.logging.getLogger',
            ) as mock_logger,
        ):
            tool = NotifyTools()
            tool.logger = mock_logger.return_value
            with self.assertRaises(RuntimeError):
                await tool.telegram_send('id', 'msg')
            tool.logger.error.assert_called_once()

    async def test_telegram_send_text_only_success(self):
        """Should succeed when sending text-only without image_base64."""
        fake_telegram = AsyncMock()
        fake_telegram.send_notification = AsyncMock()
        with patch(
            'examples.mcp_server.tools.notify.TelegramNotifier',
            return_value=fake_telegram,
        ):
            tool = NotifyTools()
            res = await tool.telegram_send('chat', 'plain text')
        fake_telegram.send_notification.assert_awaited_once()
        self.assertTrue(res['success'])

    async def test_telegram_send_with_data_url_prefix(self):
        """Should strip data URL prefix before decoding
        and send successfully.
        """
        fake_telegram = AsyncMock()
        fake_telegram.send_notification = AsyncMock()
        with (
            patch(
                'examples.mcp_server.tools.notify.TelegramNotifier',
                return_value=fake_telegram,
            ),
            patch(
                'examples.mcp_server.tools.notify.cv2.imdecode',
                return_value='bgr',
            ),
            patch(
                'examples.mcp_server.tools.notify.cv2.cvtColor',
                return_value='rgb',
            ),
        ):
            img = (
                'data:image/jpeg;base64,'
                + base64.b64encode(b'fake').decode()
            )
            tool = NotifyTools()
            res = await tool.telegram_send('chat', 'msg', img)
        fake_telegram.send_notification.assert_awaited_once()
        self.assertTrue(res['success'])


class EnsureInitialisationTests(unittest.IsolatedAsyncioTestCase):
    """Tests for internal _ensure_* methods."""

    async def test_ensure_line_messenger_initialises_once(self):
        """Should create messenger only once."""
        with patch(
            'examples.mcp_server.tools.notify.LineMessenger',
        ) as mock_line:
            tool = NotifyTools()
            await tool._ensure_line_messenger()
            await tool._ensure_line_messenger()
            mock_line.assert_called_once()

    async def test_ensure_broadcast_notifier_with_url_and_env(self):
        """Should handle both explicit and env-based URLs."""
        with (
            patch(
                'examples.mcp_server.tools.notify.BroadcastNotifier',
            ) as mock_bcast,
            patch(
                'examples.mcp_server.tools.notify.get_env_var',
                return_value='env_url',
            ),
        ):
            tool = NotifyTools()
            # explicit URL
            await tool._ensure_broadcast_notifier('explicit_url')
            # call again -> should not recreate
            await tool._ensure_broadcast_notifier('explicit_url')
            mock_bcast.assert_called_once()

        # ensure env var version works
        with (
            patch(
                'examples.mcp_server.tools.notify.BroadcastNotifier',
            ) as mock_bcast2,
            patch(
                'examples.mcp_server.tools.notify.get_env_var',
                return_value='env_url',
            ),
        ):
            tool2 = NotifyTools()
            await tool2._ensure_broadcast_notifier(None)
            mock_bcast2.assert_called_once()

    async def test_ensure_telegram_notifier_initialises_once(self):
        """Should create telegram notifier only once."""
        with patch(
            'examples.mcp_server.tools.notify.TelegramNotifier',
        ) as mock_tel:
            tool = NotifyTools()
            await tool._ensure_telegram_notifier()
            await tool._ensure_telegram_notifier()
            mock_tel.assert_called_once()


if __name__ == '__main__':
    unittest.main()

'''
pytest --cov=examples.mcp_server.tools.notify\
    --cov-report=term-missing\
    tests/examples/mcp_server/tools/notify_test.py
'''
