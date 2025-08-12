from __future__ import annotations

import unittest
from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx

from src.utils import TokenManager
from src.violation_sender import ViolationSender


class TestViolationSender(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for ViolationSender.send_violation.
    """

    site: str
    stream: str
    img: bytes
    ts: datetime
    warn: str
    det: str
    cone: str
    pole: str
    sender: ViolationSender

    def setUp(self) -> None:
        """
        Set up shared test fixtures for each test case.
        """
        self.site = 'Site-A'
        self.stream = 'Cam-01'
        self.img = b'fake_image_data'
        self.ts = datetime(2025, 1, 2, 3, 4, 5)
        self.warn = '[{"code": 1}]'
        self.det = '[]'
        self.cone = '[]'
        self.pole = '[]'
        self.sender = ViolationSender(
            api_url='http://testserver/api/violations',
            max_retries=3,
            timeout=2,
        )

    async def asyncTearDown(self) -> None:
        """Clean up resources after each test."""
        await self.sender.close()

    async def _call(self, inst: ViolationSender) -> str | None:
        """
        Helper to call send_violation with shared parameters.

        Args:
            inst (ViolationSender): The sender instance to use.

        Returns:
            Optional[str]: Violation ID if successful, else None.
        """
        return await inst.send_violation(
            site=self.site,
            stream_name=self.stream,
            image_bytes=self.img,
            detection_time=self.ts,
            warnings_json=self.warn,
            detections_json=self.det,
            cone_polygon_json=self.cone,
            pole_polygon_json=self.pole,
        )

    @patch.object(ViolationSender, '_get_client', new_callable=AsyncMock)
    @patch.object(TokenManager, 'get_valid_token', new_callable=AsyncMock)
    async def test_success(
        self,
        mock_get_valid_token: AsyncMock,
        mock_get_client: AsyncMock,
    ) -> None:
        """
        Test successful violation upload returns correct violation ID.
        """
        mock_get_valid_token.return_value = 'token-abc'

        mock_cli: MagicMock = MagicMock()
        ok_resp: MagicMock = MagicMock()
        ok_resp.raise_for_status = MagicMock()
        ok_resp.json.return_value = {'violation_id': '123'}
        mock_cli.post = AsyncMock(return_value=ok_resp)
        mock_get_client.return_value = mock_cli

        result: str | None = await self._call(self.sender)
        self.assertEqual(result, '123')
        mock_cli.post.assert_awaited_once()
        self.assertEqual(
            mock_cli.post.call_args.args[0],
            'http://testserver/api/violations/upload',
        )

    @patch.object(ViolationSender, '_get_client', new_callable=AsyncMock)
    @patch.object(TokenManager, 'get_valid_token', new_callable=AsyncMock)
    async def test_authenticate_called(
        self,
        mock_get_valid_token: AsyncMock,
        mock_get_client: AsyncMock,
    ) -> None:
        """
        Test that get_valid_token is called and works correctly.
        """
        mock_get_valid_token.return_value = 'valid_token'

        mock_cli: MagicMock = MagicMock()
        ok_resp: MagicMock = MagicMock()
        ok_resp.raise_for_status = MagicMock()
        ok_resp.json.return_value = {'violation_id': '123'}
        mock_cli.post = AsyncMock(return_value=ok_resp)
        mock_get_client.return_value = mock_cli

        result: str | None = await self._call(self.sender)
        self.assertEqual(result, '123')
        mock_get_valid_token.assert_awaited_once()
        mock_cli.post.assert_awaited_once()

    @patch.object(ViolationSender, '_get_client', new_callable=AsyncMock)
    @patch.object(TokenManager, 'get_valid_token', new_callable=AsyncMock)
    @patch.object(TokenManager, 'refresh_token', new_callable=AsyncMock)
    async def test_refresh_token(
        self,
        mock_refresh_token: AsyncMock,
        mock_get_valid_token: AsyncMock,
        mock_get_client: AsyncMock,
    ) -> None:
        """
        Test that refresh_token is called on 401 and succeeds on retry.
        """
        # First call returns expired token, second call returns new token
        mock_get_valid_token.side_effect = ['expired', 'new_token']

        req: httpx.Request = httpx.Request(
            'POST',
            'http://testserver/api/violations/upload',
        )
        resp_401: httpx.Response = httpx.Response(status_code=401, request=req)
        err_401: httpx.HTTPStatusError = httpx.HTTPStatusError(
            '401', request=req, response=resp_401,
        )

        mock_cli: MagicMock = MagicMock()
        ok_resp: MagicMock = MagicMock()
        ok_resp.raise_for_status = MagicMock()
        ok_resp.json.return_value = {'violation_id': '456'}
        mock_cli.post = AsyncMock(side_effect=[err_401, ok_resp])
        mock_get_client.return_value = mock_cli

        result: str | None = await self._call(self.sender)
        self.assertEqual(result, '456')
        mock_refresh_token.assert_awaited_once()
        self.assertEqual(mock_cli.post.await_count, 2)

    @patch.object(ViolationSender, '_get_client', new_callable=AsyncMock)
    @patch.object(TokenManager, 'get_valid_token', new_callable=AsyncMock)
    async def test_retry_exhaustion(
        self,
        mock_get_valid_token: AsyncMock,
        mock_get_client: AsyncMock,
    ) -> None:
        """
        Test that retries are exhausted and RuntimeError is raised
        on repeated timeout.
        """
        mock_get_valid_token.return_value = 'valid_token'

        mock_cli: MagicMock = MagicMock()
        mock_cli.post = AsyncMock(side_effect=httpx.ConnectTimeout('boom'))
        mock_get_client.return_value = mock_cli

        with self.assertRaises(RuntimeError):
            await self._call(self.sender)

        self.assertEqual(mock_cli.post.await_count, self.sender.max_retries)

    @patch.object(ViolationSender, '_get_client', new_callable=AsyncMock)
    @patch.object(TokenManager, 'get_valid_token', new_callable=AsyncMock)
    async def test_unexpected_exception(
        self,
        mock_get_valid_token: AsyncMock,
        mock_get_client: AsyncMock,
    ) -> None:
        """
        Test that unexpected exceptions are re-raised.
        """
        mock_get_valid_token.return_value = 'valid_token'

        mock_cli: MagicMock = MagicMock()
        mock_cli.post = AsyncMock(side_effect=ValueError('boom'))
        mock_get_client.return_value = mock_cli

        with self.assertRaises(ValueError):
            await self._call(self.sender)

    @patch.object(ViolationSender, '_get_client', new_callable=AsyncMock)
    @patch.object(TokenManager, 'get_valid_token', new_callable=AsyncMock)
    async def test_http_error_non_401(
        self,
        mock_get_valid_token: AsyncMock,
        mock_get_client: AsyncMock,
    ) -> None:
        """
        Test that non-401 HTTP errors bubble up as expected.
        """
        mock_get_valid_token.return_value = 'valid_token'

        req: httpx.Request = httpx.Request(
            'POST',
            'http://testserver/api/violations/upload',
        )
        resp_500: httpx.Response = httpx.Response(status_code=500, request=req)
        err_500: httpx.HTTPStatusError = httpx.HTTPStatusError(
            '500', request=req, response=resp_500,
        )

        mock_cli: MagicMock = MagicMock()
        mock_cli.post = AsyncMock(side_effect=err_500)
        mock_get_client.return_value = mock_cli

        with self.assertRaises(httpx.HTTPStatusError):
            await self._call(self.sender)

    @patch.object(ViolationSender, '_get_client', new_callable=AsyncMock)
    @patch.object(TokenManager, 'get_valid_token', new_callable=AsyncMock)
    async def test_no_violation_id(
        self,
        mock_get_valid_token: AsyncMock,
        mock_get_client: AsyncMock,
    ) -> None:
        """
        Test that a successful response with no violation_id returns None.
        """
        mock_get_valid_token.return_value = 'valid_token'

        mock_cli: MagicMock = MagicMock()
        ok_resp: MagicMock = MagicMock()
        ok_resp.raise_for_status = MagicMock()
        ok_resp.json.return_value = {'other': 'x'}
        mock_cli.post = AsyncMock(return_value=ok_resp)
        mock_get_client.return_value = mock_cli

        result: str | None = await self._call(self.sender)
        self.assertIsNone(result)
        mock_cli.post.assert_awaited_once()

    @patch.object(TokenManager, 'get_valid_token', new_callable=AsyncMock)
    async def test_zero_retry_returns_none(
        self,
        mock_get_valid_token: AsyncMock,
    ) -> None:
        """
        Test that max_retries=0 skips the loop and returns None.
        """
        mock_get_valid_token.return_value = 'valid_token'

        no_retry: ViolationSender = ViolationSender(
            api_url='http://testserver/api/violations',
            max_retries=0,
            timeout=1,
        )

        result: str | None = await self._call(no_retry)
        self.assertIsNone(result)

        # Clean up
        await no_retry.close()

    # -------------------------------------------------------------------------
    # Tests for constructor and edge cases
    # -------------------------------------------------------------------------

    def test_constructor_with_none_api_url(self) -> None:
        """Test constructor uses environment variable when api_url is None."""
        with patch('os.getenv', return_value='http://env-url'):
            sender = ViolationSender(api_url=None)
            self.assertEqual(sender.base_url, 'http://env-url')

    def test_constructor_strips_trailing_slash(self) -> None:
        """Test constructor removes trailing slash from api_url."""
        sender = ViolationSender(api_url='http://example.com/')
        self.assertEqual(sender.base_url, 'http://example.com')

    # -------------------------------------------------------------------------
    # Tests for _get_client method
    # -------------------------------------------------------------------------

    async def test_get_client_creates_new_client(self) -> None:
        """Test _get_client creates a new client when none exists."""
        client = await self.sender._get_client()
        self.assertIsNotNone(client)
        self.assertIsInstance(client, httpx.AsyncClient)

    async def test_get_client_reuses_existing_client(self) -> None:
        """Test _get_client reuses existing client if available."""
        client1 = await self.sender._get_client()
        client2 = await self.sender._get_client()
        self.assertIs(client1, client2)

    async def test_get_client_creates_new_when_closed(self) -> None:
        """Test _get_client creates new client when existing one is closed."""
        client1 = await self.sender._get_client()
        await client1.aclose()
        client2 = await self.sender._get_client()
        self.assertIsNot(client1, client2)

    async def test_close_method(self) -> None:
        """Test close method properly closes the client."""
        await self.sender._get_client()  # Create a client
        await self.sender.close()
        self.assertIsNone(self.sender._client)

    async def test_close_method_with_no_client(self) -> None:
        """Test close method handles case when no client exists."""
        # Should not raise any exception
        await self.sender.close()

    # -------------------------------------------------------------------------
    # Tests for error handling and edge cases
    # -------------------------------------------------------------------------

    @patch.object(TokenManager, 'get_valid_token', new_callable=AsyncMock)
    async def test_no_access_token_raises_error(
        self,
        mock_get_valid_token: AsyncMock,
    ) -> None:
        """
        Test that RuntimeError is raised when no access token is available.

        Args:
            mock_get_valid_token (AsyncMock): Mocked token manager's method.
        """
        mock_get_valid_token.return_value = ''  # Empty token

        with self.assertRaises(RuntimeError) as cm:
            await self._call(self.sender)

        self.assertIn('Failed to obtain valid access token', str(cm.exception))

    @patch.object(ViolationSender, '_get_client', new_callable=AsyncMock)
    @patch.object(TokenManager, 'get_valid_token', new_callable=AsyncMock)
    async def test_send_violation_without_optional_params(
        self,
        mock_get_valid_token: AsyncMock,
        mock_get_client: AsyncMock,
    ) -> None:
        """Test send_violation with minimal parameters."""
        mock_get_valid_token.return_value = 'valid_token'

        mock_cli: MagicMock = MagicMock()
        ok_resp: MagicMock = MagicMock()
        ok_resp.raise_for_status = MagicMock()
        ok_resp.json.return_value = {'violation_id': '789'}
        mock_cli.post = AsyncMock(return_value=ok_resp)
        mock_get_client.return_value = mock_cli

        # Call with minimal parameters
        result = await self.sender.send_violation(
            site=self.site,
            stream_name=self.stream,
            image_bytes=self.img,
        )

        self.assertEqual(result, '789')
        mock_cli.post.assert_awaited_once()

        # Verify the data sent
        call_args = mock_cli.post.call_args
        data = call_args.kwargs['data']
        self.assertEqual(data['site'], self.site)
        self.assertEqual(data['stream_name'], self.stream)
        self.assertNotIn('detection_time', data)
        self.assertNotIn('warnings_json', data)

    @patch.object(ViolationSender, '_get_client', new_callable=AsyncMock)
    @patch.object(TokenManager, 'get_valid_token', new_callable=AsyncMock)
    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_exponential_backoff_on_timeout(
        self,
        mock_sleep: AsyncMock,
        mock_get_valid_token: AsyncMock,
        mock_get_client: AsyncMock,
    ) -> None:
        """Test exponential backoff strategy on timeout errors."""
        mock_get_valid_token.return_value = 'valid_token'

        mock_cli: MagicMock = MagicMock()
        # First two attempts timeout, third succeeds
        ok_resp: MagicMock = MagicMock()
        ok_resp.raise_for_status = MagicMock()
        ok_resp.json.return_value = {'violation_id': 'success'}

        mock_cli.post = AsyncMock(
            side_effect=[
                httpx.ConnectTimeout('timeout1'),
                httpx.ConnectTimeout('timeout2'),
                ok_resp,
            ],
        )
        mock_get_client.return_value = mock_cli

        result = await self._call(self.sender)
        self.assertEqual(result, 'success')

        # Verify exponential backoff was used
        self.assertEqual(mock_sleep.await_count, 2)
        sleep_calls = [call.args[0] for call in mock_sleep.await_args_list]
        self.assertEqual(sleep_calls, [1, 2])  # 1s, then 2s backoff

    @patch.object(ViolationSender, '_get_client', new_callable=AsyncMock)
    @patch.object(TokenManager, 'get_valid_token', new_callable=AsyncMock)
    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_exponential_backoff_on_general_exception(
        self,
        mock_sleep: AsyncMock,
        mock_get_valid_token: AsyncMock,
        mock_get_client: AsyncMock,
    ) -> None:
        """Test exponential backoff strategy on general exceptions."""
        mock_get_valid_token.return_value = 'valid_token'

        mock_cli: MagicMock = MagicMock()
        # First attempt fails, second succeeds
        ok_resp: MagicMock = MagicMock()
        ok_resp.raise_for_status = MagicMock()
        ok_resp.json.return_value = {'violation_id': 'success'}

        mock_cli.post = AsyncMock(
            side_effect=[
                ConnectionError('network error'),
                ok_resp,
            ],
        )
        mock_get_client.return_value = mock_cli

        result = await self._call(self.sender)
        self.assertEqual(result, 'success')

        # Verify backoff was used
        mock_sleep.assert_awaited_once_with(1)

    @patch.object(ViolationSender, '_get_client', new_callable=AsyncMock)
    @patch.object(TokenManager, 'get_valid_token', new_callable=AsyncMock)
    @patch.object(TokenManager, 'refresh_token', new_callable=AsyncMock)
    async def test_401_error_retry_exhaustion(
        self,
        mock_refresh_token: AsyncMock,
        mock_get_valid_token: AsyncMock,
        mock_get_client: AsyncMock,
    ) -> None:
        """Test that 401 errors after refresh still get re-raised."""
        mock_get_valid_token.return_value = 'valid_token'

        req: httpx.Request = httpx.Request(
            'POST',
            'http://testserver/api/violations/upload',
        )
        resp_401: httpx.Response = httpx.Response(status_code=401, request=req)
        err_401: httpx.HTTPStatusError = httpx.HTTPStatusError(
            '401', request=req, response=resp_401,
        )

        mock_cli: MagicMock = MagicMock()
        mock_cli.post = AsyncMock(side_effect=err_401)
        mock_get_client.return_value = mock_cli

        with self.assertRaises(httpx.HTTPStatusError):
            await self._call(self.sender)

        # refresh_token is called on each 401 error (max_retries times)
        self.assertEqual(
            mock_refresh_token.await_count,
            self.sender.max_retries,
        )

    @patch.object(ViolationSender, '_get_client', new_callable=AsyncMock)
    @patch.object(TokenManager, 'get_valid_token', new_callable=AsyncMock)
    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_general_exception_retry_exhaustion(
        self,
        mock_sleep: AsyncMock,
        mock_get_valid_token: AsyncMock,
        mock_get_client: AsyncMock,
    ) -> None:
        """Test that general exceptions after all retries get re-raised."""
        mock_get_valid_token.return_value = 'valid_token'

        mock_cli: MagicMock = MagicMock()
        mock_cli.post = AsyncMock(
            side_effect=ConnectionError('persistent error'),
        )
        mock_get_client.return_value = mock_cli

        with self.assertRaises(ConnectionError):
            await self._call(self.sender)

        # Should have attempted all retries
        self.assertEqual(mock_cli.post.await_count, self.sender.max_retries)
        # Should have slept between retries (max_retries - 1 times)
        self.assertEqual(mock_sleep.await_count, self.sender.max_retries - 1)


if __name__ == '__main__':
    unittest.main()


"""
pytest \
    --cov=src.violation_sender \
    --cov-report=term-missing tests/src/violation_sender_test.py
"""
