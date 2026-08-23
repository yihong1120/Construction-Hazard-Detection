from __future__ import annotations

import base64
import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx

from examples.mcp_server.tools.notify import _decode_optional_image
from examples.mcp_server.tools.notify import NotifyTools


class LinePushTests(unittest.IsolatedAsyncioTestCase):
    """Tests for line_push method."""

    async def test_line_push_success_with_base64(self) -> None:
        """Should decode base64 and send successfully."""
        fake_line = AsyncMock()
        fake_line.push_message.return_value = 200
        with (
            patch(
                'examples.mcp_server.tools.notify.LineMessenger',
                return_value=fake_line,
            ),
            patch('examples.mcp_server.tools.notify.aiohttp.ClientSession'),
        ):
            tool = NotifyTools()
            img_b64 = base64.b64encode(b'abc').decode()
            res = await tool.line_push('uid', 'hello', img_b64)
        fake_line.push_message.assert_awaited_once()
        self.assertTrue(res['success'])
        self.assertEqual(res['status_code'], 200)
        self.assertIn('successfully', res['message'])

    def test_decode_optional_image_accepts_a_data_url(self) -> None:
        """The shared decoder accepts both raw Base64 and browser data URLs."""
        encoded = base64.b64encode(b'image').decode()

        assert _decode_optional_image(encoded) == b'image'
        assert (
            _decode_optional_image(
                f"data:image/png;base64,{encoded}",
            )
            == b'image'
        )
        assert _decode_optional_image(None) is None

    async def test_line_push_with_data_url_prefix(self) -> None:
        """Should handle data URL prefix correctly."""
        fake_line = AsyncMock()
        fake_line.push_message.return_value = 400
        with (
            patch(
                'examples.mcp_server.tools.notify.LineMessenger',
                return_value=fake_line,
            ),
            patch('examples.mcp_server.tools.notify.aiohttp.ClientSession'),
        ):
            tool = NotifyTools()
            img = 'data:image/png;base64,' + base64.b64encode(b'abc').decode()
            res = await tool.line_push('id', 'msg', img)
        self.assertFalse(res['success'])
        self.assertIn('Failed', res['message'])

    async def test_line_push_raises_and_logs(self) -> None:
        """Should log and re-raise on error."""
        with (
            patch(
                'examples.mcp_server.tools.notify.LineMessenger',
                side_effect=RuntimeError('boom'),
            ),
            patch('examples.mcp_server.tools.notify.aiohttp.ClientSession'),
            patch(
                'examples.mcp_server.tools.notify.logging.getLogger',
            ) as mock_logger,
        ):
            tool = NotifyTools()
            logger = mock_logger.return_value
            tool.logger = logger
            with self.assertRaises(RuntimeError):
                await tool.line_push('u', 'msg')
            logger.error.assert_called_once()


class BroadcastSendTests(unittest.IsolatedAsyncioTestCase):
    """Tests for broadcast_send."""

    async def test_broadcast_send_success(self) -> None:
        """Should send broadcast successfully."""
        client = AsyncMock()
        client.post.return_value = MagicMock(is_success=True, status_code=200)
        tool = NotifyTools()
        tool._http_client = client
        res = await tool.broadcast_send('message')
        self.assertTrue(res['success'])
        self.assertIn('successfully', res['message'])
        client.post.assert_awaited_once()

    async def test_broadcast_send_failure(self) -> None:
        """Should return failure message."""
        client = AsyncMock()
        client.post.return_value = MagicMock(is_success=False, status_code=503)
        tool = NotifyTools()
        tool._http_client = client
        res = await tool.broadcast_send('message')
        self.assertFalse(res['success'])
        self.assertIn('Failed', res['message'])

    async def test_broadcast_send_logs_network_failure(self) -> None:
        """Network failures return the normal failed delivery response."""
        client = AsyncMock()
        client.post.side_effect = httpx.ConnectError('boom')
        with (
            patch(
                'examples.mcp_server.tools.notify.logging.getLogger',
            ) as mock_logger,
        ):
            tool = NotifyTools()
            tool._http_client = client
            logger = mock_logger.return_value
            tool.logger = logger
            result = await tool.broadcast_send('fail')
            self.assertFalse(result['success'])
            logger.warning.assert_called_once()


class TelegramSendTests(unittest.IsolatedAsyncioTestCase):
    """Tests for telegram_send."""

    async def test_telegram_send_success_with_image(self) -> None:
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

    async def test_telegram_send_imdecode_none(self) -> None:
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

    async def test_telegram_send_notification_failure(self) -> None:
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

    async def test_telegram_send_logs_and_reraises(self) -> None:
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
            logger = mock_logger.return_value
            tool.logger = logger
            with self.assertRaises(RuntimeError):
                await tool.telegram_send('id', 'msg')
            logger.error.assert_called_once()

    async def test_telegram_send_text_only_success(self) -> None:
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

    async def test_telegram_send_with_data_url_prefix(self) -> None:
        """Should strip data URL prefix before decoding and send
        successfully."""
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
                'data:image/jpeg;base64,' + base64.b64encode(b'fake').decode()
            )
            tool = NotifyTools()
            res = await tool.telegram_send('chat', 'msg', img)
        fake_telegram.send_notification.assert_awaited_once()
        self.assertTrue(res['success'])


class EnsureInitialisationTests(unittest.IsolatedAsyncioTestCase):
    """Tests for internal _ensure_* methods."""

    async def test_ensure_line_messenger_initialises_once(self) -> None:
        """Should create messenger only once."""
        with (
            patch(
                'examples.mcp_server.tools.notify.LineMessenger',
            ) as mock_line,
            patch(
                'examples.mcp_server.tools.notify.aiohttp.ClientSession',
            ) as mock_session,
        ):
            tool = NotifyTools()
            await tool._ensure_line_messenger()
            await tool._ensure_line_messenger()
            mock_line.assert_called_once()
            mock_session.assert_called_once()

    async def test_broadcast_client_is_reused(self) -> None:
        """One NotifyTools instance owns one reusable broadcast transport."""
        tool = NotifyTools()
        first = tool._broadcast_http_client()
        second = tool._broadcast_http_client()
        self.assertIs(first, second)
        await tool.close()

    async def test_ensure_telegram_notifier_initialises_once(self) -> None:
        """Should create telegram notifier only once."""
        with patch(
            'examples.mcp_server.tools.notify.TelegramNotifier',
        ) as mock_tel:
            tool = NotifyTools()
            await tool._ensure_telegram_notifier()
            await tool._ensure_telegram_notifier()
            mock_tel.assert_called_once()

    async def test_messenger_and_wechat_share_client_and_return_results(
        self,
    ) -> None:
        """Messenger and WeChat initialise once and preserve results.

        Both feature clients use the shared async HTTP transport.
        """
        messenger = AsyncMock()
        messenger.send_notification.return_value = 201
        wechat = AsyncMock()
        wechat.send_notification.return_value = {'errcode': 1}
        client = AsyncMock()
        tool = NotifyTools()
        tool._http_client = client

        with (
            patch(
                'examples.mcp_server.tools.notify.MessengerNotifier',
                return_value=messenger,
            ) as messenger_class,
            patch(
                'examples.mcp_server.tools.notify.WeChatNotifier',
                return_value=wechat,
            ) as wechat_class,
        ):
            messenger_result = await tool.messenger_send('recipient', 'hello')
            wechat_result = await tool.wechat_send('user', 'hello')
            wechat_success = await tool.wechat_send('user', 'again')

        self.assertTrue(messenger_result['success'])
        self.assertFalse(wechat_result['success'])
        self.assertFalse(wechat_success['success'])
        messenger_class.assert_called_once_with(client=client)
        wechat_class.assert_called_once_with(client=client)

    async def test_close_releases_line_and_http_transports(self) -> None:
        """Closing a tool releases transports and clears all lazy clients."""
        tool = NotifyTools()
        tool._http_client = AsyncMock()
        tool._line_session = AsyncMock()
        tool._line_messenger = MagicMock()
        tool._messenger_notifier = MagicMock()
        tool._wechat_notifier = MagicMock()

        await tool.close()

        self.assertIsNone(tool._http_client)
        self.assertIsNone(tool._line_session)
        self.assertIsNone(tool._line_messenger)
        self.assertIsNone(tool._messenger_notifier)
        self.assertIsNone(tool._wechat_notifier)


if __name__ == '__main__':
    unittest.main()
