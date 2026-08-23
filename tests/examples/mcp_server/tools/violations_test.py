from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx

from examples.mcp_server.tools.violations import ViolationsTools


class TestViolationsTools(unittest.IsolatedAsyncioTestCase):

    """Test suite."""

    async def asyncSetUp(self) -> None:
        """Prepare test fixtures."""
        self.tool = ViolationsTools()
        self.logger = MagicMock()
        self.tool.logger = self.logger

    # --- _ensure_client ---

    @patch(
        'examples.mcp_server.tools.violations.get_env_var',
        side_effect=['https://api', 'https://db'],
    )
    @patch(
        'examples.mcp_server.tools.violations.httpx.AsyncClient',
        autospec=True,
    )
    @patch(
        'examples.mcp_server.tools.violations.TokenManager',
        autospec=True,
    )
    async def test_ensure_client_initializes(
        self,
        mock_token: Any,
        mock_client: Any,
        mock_env: Any,
    ) -> None:
        """Exercise this test."""
        await self.tool._ensure_client()
        mock_client.assert_called_once()
        mock_token.assert_called_once()

    # --- _get_auth_headers ---

    @patch(
        'examples.mcp_server.tools.violations.os.getenv',
        side_effect=lambda k, d=None: 'STATIC_TOKEN'
        if k == 'MCP_STATIC_BEARER'
        else '',
    )
    async def test_get_auth_headers_static_bearer(self, _: Any) -> None:
        """Exercise this test."""
        self.tool._token_manager = MagicMock()
        headers = await self.tool._get_auth_headers()
        self.assertIn('Authorization', headers)
        self.assertTrue(headers['Authorization'].startswith('Bearer'))

    @patch(
        'examples.mcp_server.tools.violations.os.getenv',
        side_effect=lambda k, d=None: 'true'
        if k == 'MCP_ALLOW_NO_AUTH'
        else '',
    )
    async def test_get_auth_headers_no_auth(self, _: Any) -> None:
        """Exercise this test."""
        self.tool._token_manager = MagicMock()
        headers = await self.tool._get_auth_headers()
        self.assertNotIn('Authorization', headers)

    @patch('examples.mcp_server.tools.violations.os.getenv', return_value='')
    async def test_get_auth_headers_valid_token(self, _: Any) -> None:
        """Exercise this test."""
        fake_token = 'XYZ'
        tm = AsyncMock()
        tm.get_valid_token.return_value = fake_token
        self.tool._token_manager = tm
        headers = await self.tool._get_auth_headers()
        self.assertEqual(headers['Authorization'], f'Bearer {fake_token}')

    @patch('examples.mcp_server.tools.violations.os.getenv', return_value='')
    async def test_get_auth_headers_exception(self, _: Any) -> None:
        """Exercise this test."""
        self.tool._token_manager = AsyncMock()
        self.tool._token_manager.get_valid_token.side_effect = RuntimeError(
            'fail',
        )
        with self.assertRaises(RuntimeError):
            await self.tool._get_auth_headers()
        self.logger.error.assert_called_once()

    # --- search ---

    @patch.object(
        ViolationsTools,
        '_get_auth_headers',
        AsyncMock(return_value={'h': 'v'}),
    )
    @patch.object(ViolationsTools, '_ensure_client', AsyncMock())
    async def test_search_success(self) -> None:
        """Exercise this test."""
        mock_response = MagicMock()
        mock_response.json.return_value = {'total': 1, 'items': []}
        mock_response.raise_for_status.return_value = None
        client = AsyncMock()
        client.get.return_value = mock_response
        self.tool._client = client
        self.tool._base_url = 'https://x'
        res = await self.tool.search(site_id=1, keyword='a', limit=999)
        self.assertEqual(res['total'], 1)

    @patch.object(ViolationsTools, '_ensure_client', AsyncMock())
    async def test_search_exception(self) -> None:
        """Exercise this test."""
        with patch.object(
            self.tool,
            '_get_auth_headers',
            new=AsyncMock(side_effect=RuntimeError('bad')),
        ):
            # Ensure client is set to pass the internal assertion
            self.tool._client = AsyncMock()
            self.tool._base_url = 'https://x'
            with self.assertRaises(RuntimeError):
                await self.tool.search()
        self.logger.error.assert_called_once()

    @patch.object(
        ViolationsTools,
        '_get_auth_headers',
        AsyncMock(return_value={'h': 'v'}),
    )
    @patch.object(ViolationsTools, '_ensure_client', AsyncMock())
    async def test_search_with_start_end_time_params(self) -> None:
        # Ensure start_time/end_time branches are covered
        """Exercise this test."""
        mock_response = MagicMock()
        mock_response.json.return_value = {'total': 0, 'items': []}
        # non-awaitable path for _maybe_await
        mock_response.raise_for_status.return_value = None
        client = AsyncMock()
        client.get.return_value = mock_response
        self.tool._client = client
        self.tool._base_url = 'https://x'
        start_time = '2025-01-01T00:00:00Z'
        end_time = '2025-01-02T00:00:00Z'
        await self.tool.search(
            start_time=start_time,
            end_time=end_time,
            limit=10,
        )
        # Verify that params include start_time and end_time
        client.get.assert_awaited()
        called_kwargs = client.get.await_args.kwargs
        self.assertIn('params', called_kwargs)
        self.assertEqual(called_kwargs['params'].get('start_time'), start_time)
        self.assertEqual(called_kwargs['params'].get('end_time'), end_time)

    # --- get ---

    @patch.object(
        ViolationsTools,
        '_get_auth_headers',
        AsyncMock(return_value={'h': 'v'}),
    )
    @patch.object(ViolationsTools, '_ensure_client', AsyncMock())
    async def test_get_success(self) -> None:
        """Exercise this test."""
        resp = MagicMock()
        resp.json.return_value = {'id': 1}
        resp.raise_for_status.return_value = None
        client = AsyncMock()
        client.get.return_value = resp
        self.tool._client = client
        self.tool._base_url = 'https://x'
        result = await self.tool.get(1)
        self.assertEqual(result['id'], 1)

    @patch.object(ViolationsTools, '_ensure_client', AsyncMock())
    async def test_get_exception(self) -> None:
        """Exercise this test."""
        with patch.object(
            self.tool,
            '_get_auth_headers',
            new=AsyncMock(side_effect=RuntimeError('fail')),
        ):
            # Ensure client is set to pass the internal assertion
            self.tool._client = AsyncMock()
            self.tool._base_url = 'https://x'
            with self.assertRaises(RuntimeError):
                await self.tool.get(1)
        self.logger.error.assert_called_once()

    # --- get_image ---

    @patch.object(
        ViolationsTools,
        '_get_auth_headers',
        AsyncMock(return_value={'h': 'v'}),
    )
    @patch.object(ViolationsTools, '_ensure_client', AsyncMock())
    async def test_get_image_as_base64_success(self) -> None:
        """Exercise this test."""
        mock_response = MagicMock()
        mock_response.content = b'data'
        mock_response.headers = {'content-type': 'image/png'}
        mock_response.raise_for_status.return_value = None
        client = AsyncMock()
        client.get.return_value = mock_response
        self.tool._client = client
        self.tool._base_url = 'https://api'
        with patch(
            'examples.mcp_server.tools.violations.base64.b64encode',
            return_value=b'ENCODED',
        ):
            res = await self.tool.get_image('x.jpg', as_base64=True)
        self.assertEqual(res['image_base64'], 'ENCODED')
        self.assertEqual(res['media_type'], 'image/png')

    async def test_get_image_non_base64(self) -> None:
        """Exercise this test."""
        self.tool._base_url = 'https://api'
        res = await self.tool.get_image('x.jpg', as_base64=False)
        self.assertIn('url', res)

    @patch.object(
        ViolationsTools,
        '_get_auth_headers',
        AsyncMock(return_value={'h': 'v'}),
    )
    @patch.object(ViolationsTools, '_ensure_client', AsyncMock())
    async def test_get_image_http_error(self) -> None:
        """Exercise this test."""
        client = AsyncMock()
        response = MagicMock(status_code=404)
        err = httpx.HTTPStatusError('404', request=None, response=response)
        client.get.side_effect = err
        self.tool._client = client
        self.tool._base_url = 'https://api'
        result = await self.tool.get_image('bad', as_base64=True)
        self.assertFalse(result['success'])
        self.assertEqual(result['status_code'], 404)

    @patch.object(ViolationsTools, '_ensure_client', AsyncMock())
    async def test_get_image_general_error(self) -> None:
        """Exercise this test."""
        with patch.object(
            self.tool,
            '_get_auth_headers',
            new=AsyncMock(side_effect=RuntimeError('boom')),
        ):
            # Ensure client is set to pass the internal assertion
            self.tool._client = AsyncMock()
            self.tool._base_url = 'https://api'
            res = await self.tool.get_image('xx', as_base64=True)
        self.assertFalse(res['success'])
        self.assertIn('boom', res['message'])

    # --- my_sites ---

    @patch.object(
        ViolationsTools,
        '_get_auth_headers',
        AsyncMock(return_value={'h': 'v'}),
    )
    @patch.object(ViolationsTools, '_ensure_client', AsyncMock())
    async def test_my_sites_success(self) -> None:
        """Exercise this test."""
        resp = MagicMock()
        resp.json.return_value = [{'id': 1}]
        resp.raise_for_status.return_value = None
        client = AsyncMock()
        client.get.return_value = resp
        self.tool._client = client
        self.tool._base_url = 'https://x'
        res = await self.tool.my_sites()
        self.assertEqual(res[0]['id'], 1)

    @patch.object(ViolationsTools, '_ensure_client', AsyncMock())
    async def test_my_sites_exception(self) -> None:
        """Exercise this test."""
        with patch.object(
            self.tool,
            '_get_auth_headers',
            new=AsyncMock(side_effect=RuntimeError('bad')),
        ):
            # Ensure client is set to pass the internal assertion
            self.tool._client = AsyncMock()
            self.tool._base_url = 'https://x'
            with self.assertRaises(RuntimeError):
                await self.tool.my_sites()
        self.logger.error.assert_called_once()

    # --- close ---

    async def test_close(self) -> None:
        """Exercise this test."""
        client = AsyncMock()
        self.tool._client = client
        await self.tool.close()
        client.aclose.assert_awaited_once()
        self.assertIsNone(self.tool._client)


if __name__ == '__main__':
    unittest.main()

'''
pytest --cov=examples.mcp_server.tools.violations\
    --cov-report=term-missing\
    tests/examples/mcp_server/tools/violations_test.py
'''
