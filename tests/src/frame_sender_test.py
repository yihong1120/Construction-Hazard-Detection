from __future__ import annotations

import unittest
from collections.abc import MutableMapping
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx
import numpy as np

from src.frame_sender import BackendFrameSender


class TestBackendFrameSender(unittest.IsolatedAsyncioTestCase):
    """Async test suite for ``BackendFrameSender``.

    A fresh sender is created per test in ``setUp`` and closed in
    ``asyncTearDown`` to avoid leaking resources.
    """

    # Common attributes used across tests
    sender: BackendFrameSender
    mock_shared_token: MutableMapping[str, str | bool]
    site: str
    stream_name: str
    frame_bytes: bytes
    warnings_json: str
    cone_polygons_json: str
    pole_polygons_json: str
    detection_items_json: str

    def setUp(self) -> None:
        """Initialise shared sender and common test data before each test."""
        self.mock_shared_token: MutableMapping[str, str | bool] = {
            'access_token': 'init_token',
            'refresh_token': '',
            'is_refreshing': False,
        }

        self.sender: BackendFrameSender = BackendFrameSender(
            api_url='http://testserver.local/api/streaming_web',
            shared_token=self.mock_shared_token,
            max_retries=2,
            timeout=5,
        )

        self.site: str = 'TestSite'
        self.stream_name: str = 'TestStream'
        self.frame_bytes: bytes = b'fake_image_data'

        self.warnings_json: str = '{"some": "warning"}'
        self.cone_polygons_json: str = '{"cone": "polygon"}'
        self.pole_polygons_json: str = '{"pole": "polygon"}'
        self.detection_items_json: str = '{"some": "detection"}'

    async def asyncTearDown(self) -> None:
        """Close network resources after each test to prevent leaks."""
        await self.sender.close()

    # HTTP path: delegate to NetClient
    async def test_send_frame_success_delegates_to_netclient(self) -> None:
        """Delegates to NetClient.http_post and returns its JSON."""
        with patch.object(
            self.sender._net,
            'http_post',
            new_callable=AsyncMock,
        ) as mock_http_post:
            mock_http_post.return_value = {'ok': True}
            # Exercise the HTTP happy path
            result = await self.sender.send_frame(
                site=self.site,
                stream_name=self.stream_name,
                frame_bytes=self.frame_bytes,
                warnings_json=self.warnings_json,
                cone_polygons_json=self.cone_polygons_json,
                pole_polygons_json=self.pole_polygons_json,
                detection_items_json=self.detection_items_json,
                width=640,
                height=480,
            )
            self.assertEqual(result, {'ok': True})
            mock_http_post.assert_awaited_once()

    async def test_send_frame_http_status_error_raised(self) -> None:
        """Propagates httpx.HTTPStatusError from NetClient.http_post.

        Raises:
            httpx.HTTPStatusError: When backend replies with an error.
        """
        with patch.object(
            self.sender._net,
            'http_post',
            new_callable=AsyncMock,
        ) as mock_http_post:
            resp = MagicMock(status_code=403, reason_phrase='Forbidden')
            mock_http_post.side_effect = httpx.HTTPStatusError(
                'Forbidden', request=MagicMock(), response=resp,
            )
            with self.assertRaises(httpx.HTTPStatusError):
                await self.sender.send_frame(
                    self.site,
                    self.stream_name,
                    self.frame_bytes,
                )
            mock_http_post.assert_awaited_once()

    async def test_send_frame_other_exception_propagated(self) -> None:
        """Propagates unexpected exceptions from NetClient.http_post."""
        with patch.object(
            self.sender._net,
            'http_post',
            new_callable=AsyncMock,
        ) as mock_http_post:
            mock_http_post.side_effect = ValueError('boom')
            with self.assertRaises(ValueError):
                await self.sender.send_frame(
                    self.site,
                    self.stream_name,
                    self.frame_bytes,
                )
            mock_http_post.assert_awaited_once()

    async def test_send_frame_all_attempts_exhausted_message(self) -> None:
        """Raises when NetClient retries are exhausted for HTTP."""
        with patch.object(
            self.sender._net,
            'http_post',
            new_callable=AsyncMock,
        ) as mock_http_post:
            mock_http_post.side_effect = RuntimeError(
                'HTTP POST retries exhausted',
            )
            with self.assertRaises(RuntimeError) as ctx:
                await self.sender.send_frame(
                    self.site,
                    self.stream_name,
                    self.frame_bytes,
                )
            self.assertIn('retries', str(ctx.exception))

    # WebSocket path: via NetClient.ws_send_and_receive
    async def test_send_frame_ws_success(self) -> None:
        """Returns server JSON for a successful WS send/receive."""
        with patch.object(
            self.sender._net,
            'ws_send_and_receive',
            new_callable=AsyncMock,
        ) as mock_ws:
            mock_ws.return_value = {'status': 'success'}
            result = await self.sender.send_frame_ws(
                self.site, self.stream_name, self.frame_bytes,
            )
            self.assertEqual(result, {'status': 'success'})

    async def test_send_frame_ws_non_dict_ok(self) -> None:
        """Maps non-dict WS payloads to a generic ok result."""
        with patch.object(
            self.sender._net,
            'ws_send_and_receive',
            new_callable=AsyncMock,
        ) as mock_ws:
            mock_ws.return_value = b'binary'
            result = await self.sender.send_frame_ws(
                self.site, self.stream_name, self.frame_bytes,
            )
            self.assertEqual(result, {'status': 'ok'})

    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_send_frame_ws_retry_then_success(
        self, mock_sleep: AsyncMock,
    ) -> None:
        """Retries once for WS then succeeds and returns ok.

        Args:
            mock_sleep: Patched ``asyncio.sleep`` to avoid real delays.
        """
        with patch.object(
            self.sender._net,
            'ws_send_and_receive',
            new_callable=AsyncMock,
        ) as mock_ws:
            mock_ws.side_effect = [None, {'status': 'ok'}]
            result = await self.sender.send_frame_ws(
                self.site, self.stream_name, self.frame_bytes,
            )
            self.assertEqual(result, {'status': 'ok'})
            self.assertTrue(mock_sleep.await_count >= 1)

    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_send_frame_ws_all_attempts_fail(
        self, mock_sleep: AsyncMock,
    ) -> None:
        """Returns an error after exhausting all WS attempts.

        Args:
            mock_sleep: Patched ``asyncio.sleep`` to avoid real delays.
        """
        with patch.object(
            self.sender._net,
            'ws_send_and_receive',
            new_callable=AsyncMock,
        ) as mock_ws:
            mock_ws.return_value = None
            result = await self.sender.send_frame_ws(
                self.site, self.stream_name, self.frame_bytes,
            )
            self.assertEqual(result['status'], 'error')
            self.assertIn('WebSocket send failed', result['message'])
            self.assertEqual(mock_ws.await_count, 5)

    # send_optimized_frame branches
    @patch('src.utils.Utils.encode_frame')
    async def test_send_optimized_frame_jpeg_websocket_success(
        self, mock_encode_frame: MagicMock,
    ) -> None:
        """Encodes as JPEG then sends via WS and returns success.

        Args:
            mock_encode_frame: Patched encoder for deterministic bytes.
        """
        mock_frame = np.zeros((2, 2, 3), dtype=np.uint8)
        mock_encode_frame.return_value = b'fake_encoded_jpeg'
        with patch.object(
            self.sender,
            'send_frame_ws',
            new_callable=AsyncMock,
        ) as mock_send_frame_ws:
            mock_send_frame_ws.return_value = {'status': 'success'}
            result = await self.sender.send_optimized_frame(
                frame=mock_frame,
                site='s',
                stream_name='k',
                encoding_format='jpeg',
                jpeg_quality=80,
                use_websocket=True,
            )
            self.assertEqual(result, {'status': 'success'})
            mock_send_frame_ws.assert_called_once()
            mock_encode_frame.assert_called_once_with(mock_frame, 'jpeg', 80)

    @patch('src.utils.Utils.encode_frame')
    async def test_send_optimized_frame_png_http_success(
        self, mock_encode_frame: MagicMock,
    ) -> None:
        """Encodes as PNG then sends via HTTP and returns success.

        Args:
            mock_encode_frame: Patched encoder for deterministic bytes.
        """
        mock_frame = np.zeros((2, 2, 3), dtype=np.uint8)
        mock_encode_frame.return_value = b'fake_encoded_png'
        with patch.object(
            self.sender._net,
            'http_post',
            new_callable=AsyncMock,
        ) as mock_http_post:
            mock_http_post.return_value = {'status': 'success'}
            result = await self.sender.send_optimized_frame(
                frame=mock_frame,
                site='s',
                stream_name='k',
                encoding_format='png',
                use_websocket=False,
            )
            self.assertEqual(result, {'status': 'success'})
            mock_http_post.assert_awaited_once()
            mock_encode_frame.assert_called_once_with(mock_frame, 'png')

    @patch('src.utils.Utils.encode_frame')
    async def test_send_optimized_frame_encoding_failure(
        self, mock_encode_frame: MagicMock,
    ) -> None:
        """Returns a failure dict when encoding yields no bytes.

        Args:
            mock_encode_frame: Patched encoder returning None.
        """
        mock_frame = np.zeros((2, 2, 3), dtype=np.uint8)
        mock_encode_frame.return_value = None
        result = await self.sender.send_optimized_frame(
            frame=mock_frame,
            site='s',
            stream_name='k',
        )
        self.assertEqual(
            result, {'success': False, 'error': 'Failed to encode frame'},
        )

    @patch('src.utils.Utils.encode_frame')
    async def test_send_optimized_frame_websocket_fallback_to_http(
        self, mock_encode_frame: MagicMock,
    ) -> None:
        """Falls back to HTTP when WS returns an error dict.

        Args:
            mock_encode_frame: Patched encoder for deterministic bytes.
        """
        mock_frame = np.zeros((2, 2, 3), dtype=np.uint8)
        mock_encode_frame.return_value = b'fake_encoded'
        with (
            patch.object(
                self.sender, 'send_frame_ws', new_callable=AsyncMock,
            ) as mock_ws,
            patch.object(
                self.sender._net, 'http_post', new_callable=AsyncMock,
            ) as mock_http_post,
        ):
            mock_ws.return_value = {'status': 'error', 'message': 'x'}
            mock_http_post.return_value = {'status': 'http_success'}
            result = await self.sender.send_optimized_frame(
                frame=mock_frame,
                site='s',
                stream_name='k',
                use_websocket=True,
            )
            self.assertEqual(result, {'status': 'http_success'})
            mock_ws.assert_awaited_once()
            mock_http_post.assert_awaited_once()

    @patch('src.utils.Utils.encode_frame')
    async def test_send_optimized_frame_general_exception(
        self, mock_encode_frame: MagicMock,
    ) -> None:
        """Catches unexpected exceptions and returns an error dict.

        Args:
            mock_encode_frame: Patched encoder raising an exception.
        """
        mock_frame = np.zeros((2, 2, 3), dtype=np.uint8)
        mock_encode_frame.side_effect = Exception('Encoding error')
        result = await self.sender.send_optimized_frame(
            frame=mock_frame,
            site='s',
            stream_name='k',
        )
        self.assertFalse(result.get('success', True))
        self.assertIn('error', result)

    # close delegates to NetClient
    async def test_close_delegates_to_netclient(self) -> None:
        """Calls NetClient.close to release resources."""
        with patch.object(
            self.sender._net, 'close', new_callable=AsyncMock,
        ) as mock_close:
            await self.sender.close()
            mock_close.assert_awaited_once()

    async def test_close_handles_exception_and_logs(self) -> None:
        """Logs and swallows exceptions raised during close."""
        # Force _net.close to raise to cover the exception branch
        with (
            patch.object(
                self.sender._net, 'close', new_callable=AsyncMock,
            ) as mock_close,
            patch('src.frame_sender.logging.error') as mock_log,
        ):
            mock_close.side_effect = RuntimeError('boom')
            # close should swallow the exception and log an error
            await self.sender.close()
            mock_log.assert_called()
            # Ensure the log message contains our expected text
            args, _ = mock_log.call_args
            self.assertIn('Error closing NetClient', args[0])

    # constructor default token when None
    def test_constructor_with_none_shared_token(self) -> None:
        """Constructs with default shared token when None is provided."""
        sender = BackendFrameSender(shared_token=None)
        expected = {
            'access_token': '',
            'refresh_token': '',
            'is_refreshing': False,
        }
        self.assertEqual(sender.shared_token, expected)


if __name__ == '__main__':
    unittest.main()

"""Quick run:
pytest -q tests/src/frame_sender_test.py
"""
