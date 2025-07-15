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

    @patch('src.violation_sender.httpx.AsyncClient')
    async def test_success(self, mock_ac: MagicMock) -> None:
        """
        Test successful violation upload returns correct violation ID.
        """
        self.sender.shared_token['access_token'] = 'token-abc'

        mock_cli: MagicMock = MagicMock()
        ok_resp: MagicMock = MagicMock()
        ok_resp.raise_for_status = MagicMock()
        ok_resp.json.return_value = {'violation_id': '123'}
        mock_cli.post = AsyncMock(return_value=ok_resp)
        mock_ac.return_value.__aenter__.return_value = mock_cli

        result: str | None = await self._call(self.sender)
        self.assertEqual(result, '123')
        mock_cli.post.assert_awaited_once()
        self.assertEqual(
            mock_cli.post.call_args.args[0],
            'http://testserver/api/violations/upload',
        )

    @patch('src.violation_sender.httpx.AsyncClient')
    @patch.object(TokenManager, 'authenticate', new_callable=AsyncMock)
    async def test_authenticate_called(
        self,
        mock_authenticate: AsyncMock,
        mock_ac: MagicMock,
    ) -> None:
        """
        Test that authenticate is called when no token is present.
        """
        # Patch TokenManager.authenticate and set up mock response
        mock_cli: MagicMock = MagicMock()
        ok_resp: MagicMock = MagicMock()
        ok_resp.raise_for_status = MagicMock()
        ok_resp.json.return_value = {'violation_id': '123'}
        mock_cli.post = AsyncMock(return_value=ok_resp)
        mock_ac.return_value.__aenter__.return_value = mock_cli

        result: str | None = await self._call(self.sender)
        self.assertEqual(result, '123')
        mock_authenticate.assert_awaited_once()
        mock_cli.post.assert_awaited_once()

    @patch('src.violation_sender.httpx.AsyncClient')
    @patch.object(TokenManager, 'refresh_token', new_callable=AsyncMock)
    async def test_refresh_token(
        self,
        mock_refresh_token: AsyncMock,
        mock_ac: MagicMock,
    ) -> None:
        """
        Test that refresh_token is called on 401 and succeeds on retry.
        """
        self.sender.shared_token['access_token'] = 'expired'

        async def _refresh() -> None:
            self.sender.shared_token['access_token'] = 'new'

        mock_refresh_token.side_effect = _refresh

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
        mock_ac.return_value.__aenter__.return_value = mock_cli

        result: str | None = await self._call(self.sender)
        self.assertEqual(result, '456')
        mock_refresh_token.assert_awaited_once()
        self.assertEqual(mock_cli.post.await_count, 2)

    @patch('src.violation_sender.httpx.AsyncClient')
    async def test_retry_exhaustion(self, mock_ac: MagicMock) -> None:
        """
        Test that retries are exhausted and RuntimeError is raised
        on repeated timeout.
        """
        self.sender.shared_token['access_token'] = 't'
        mock_cli: MagicMock = MagicMock()
        mock_cli.post = AsyncMock(side_effect=httpx.ConnectTimeout('boom'))
        mock_ac.return_value.__aenter__.return_value = mock_cli

        with self.assertRaises(RuntimeError):
            await self._call(self.sender)

        self.assertEqual(mock_cli.post.await_count, self.sender.max_retries)

    @patch('src.violation_sender.httpx.AsyncClient')
    async def test_unexpected_exception(self, mock_ac: MagicMock) -> None:
        """
        Test that unexpected exceptions are re-raised.
        """
        self.sender.shared_token['access_token'] = 't'
        mock_cli: MagicMock = MagicMock()
        mock_cli.post = AsyncMock(side_effect=ValueError('boom'))
        mock_ac.return_value.__aenter__.return_value = mock_cli

        with self.assertRaises(ValueError):
            await self._call(self.sender)

    @patch('src.violation_sender.httpx.AsyncClient')
    async def test_http_error_non_401(self, mock_ac: MagicMock) -> None:
        """
        Test that non-401 HTTP errors bubble up as expected.
        """
        self.sender.shared_token['access_token'] = 't'
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
        mock_ac.return_value.__aenter__.return_value = mock_cli

        with self.assertRaises(httpx.HTTPStatusError):
            await self._call(self.sender)

    @patch('src.violation_sender.httpx.AsyncClient')
    async def test_no_violation_id(self, mock_ac: MagicMock) -> None:
        """
        Test that a successful response with no violation_id returns None.
        """
        self.sender.shared_token['access_token'] = 't'
        mock_cli: MagicMock = MagicMock()
        ok_resp: MagicMock = MagicMock()
        ok_resp.raise_for_status = MagicMock()
        ok_resp.json.return_value = {'other': 'x'}
        mock_cli.post = AsyncMock(return_value=ok_resp)
        mock_ac.return_value.__aenter__.return_value = mock_cli

        result: str | None = await self._call(self.sender)
        self.assertIsNone(result)
        mock_cli.post.assert_awaited_once()

    @patch('src.violation_sender.httpx.AsyncClient')
    async def test_zero_retry_returns_none(self, mock_ac: MagicMock) -> None:
        """
        Test that max_retries=0 skips the loop and returns None.
        """
        no_retry: ViolationSender = ViolationSender(
            api_url='http://testserver/api/violations',
            max_retries=0,
            timeout=1,
        )
        no_retry.shared_token['access_token'] = 't'

        result: str | None = await self._call(no_retry)
        self.assertIsNone(result)
        mock_ac.assert_not_called()


if __name__ == '__main__':
    unittest.main()


"""
pytest \
    --cov=src.violation_sender \
    --cov-report=term-missing tests/src/violation_sender_test.py
"""
