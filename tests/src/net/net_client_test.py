from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import aiohttp
import httpx

from src.net.net_client import NetClient
from src.utils import TokenManager


class TestNetClient(unittest.IsolatedAsyncioTestCase):
    """
    Behavioural tests for NetClient covering core branches.
    """

    def setUp(self) -> None:
        """Set up a client with a fake token manager for reuse.

        Creates a mock ``TokenManager`` and a ``NetClient`` instance with
        short timeouts to keep tests fast.
        """
        # Mock token manager methods used by NetClient
        self.tm: MagicMock = MagicMock()
        self.tm.get_valid_token = AsyncMock(return_value='tkn')
        self.tm.authenticate = AsyncMock()
        self.tm.refresh_token = AsyncMock()

        # Create NetClient with small timeouts/backoff suitable for tests
        self.client: NetClient = NetClient(
            'https://example.com/api',
            token_manager=cast(TokenManager, self.tm),
            timeout=5,
            reconnect_backoff=0.1,
            ws_heartbeat=20,
            ws_send_timeout=1.0,
            ws_recv_timeout=1.0,
            ws_connect_attempts=2,
        )

    def test_build_http_url_leading_slash_added(self) -> None:
        """Relative path without leading slash should be normalised."""
        self.assertEqual(
            self.client.build_http_url('v1/ping'),
            'https://example.com/api/v1/ping',
        )
        self.assertEqual(
            self.client.build_http_url('/v1/ping'),
            'https://example.com/api/v1/ping',
        )

    def test_build_ws_url_scheme_and_path_join(self) -> None:
        """WS scheme is derived from base and path is POSIX-joined."""
        self.assertEqual(
            self.client.build_ws_url('ws'),
            'wss://example.com/api/ws',
        )
        self.assertEqual(
            self.client.build_ws_url('/ws'),
            'wss://example.com/api/ws',
        )

    def test_build_ws_url_for_http_base_uses_ws(self) -> None:
        """For http base, WS scheme should be 'ws'."""
        client = NetClient(
            'http://example.com/base',
            token_manager=cast(TokenManager, self.tm),
        )
        self.assertEqual(client.build_ws_url('/x'), 'ws://example.com/base/x')

    async def test_auth_headers_success_and_fallback(self) -> None:
        """If first retrieval fails, authenticate then retry once."""
        self.tm.get_valid_token.side_effect = [Exception('x'), 'ok']
        headers = await self.client.auth_headers()
        self.tm.authenticate.assert_awaited_once_with(force=True)
        self.assertEqual(headers['Authorization'], 'Bearer ok')
        self.assertIn('User-Agent', headers)

    async def test_http_post_success(self) -> None:
        """Return JSON on successful POST."""
        fake_resp = MagicMock()
        fake_resp.json.return_value = {'ok': True}
        fake_resp.raise_for_status.return_value = None
        fake_client = MagicMock()
        fake_client.post = AsyncMock(return_value=fake_resp)

        cm = MagicMock()
        cm.__aenter__.return_value = fake_client
        cm.__aexit__.return_value = False
        with patch('httpx.AsyncClient', return_value=cm):
            out = await self.client.http_post('/upload', data={'a': 1})
        self.assertEqual(out, {'ok': True})

    async def test_http_post_connect_timeout_then_success(self) -> None:
        """On timeout, wait and retry before succeeding."""
        fake_resp = MagicMock()
        fake_resp.json.return_value = {'ok': 1}
        fake_resp.raise_for_status.return_value = None

        async def post_side_effect(*_a, **_k):
            if not hasattr(post_side_effect, 'called'):
                setattr(post_side_effect, 'called', True)
                raise httpx.ConnectTimeout('t')
            return fake_resp

        fake_client = MagicMock()
        fake_client.post = AsyncMock(side_effect=post_side_effect)
        cm = MagicMock()
        cm.__aenter__.return_value = fake_client
        cm.__aexit__.return_value = False

        with (
            patch('httpx.AsyncClient', return_value=cm),
            patch('asyncio.sleep', new=AsyncMock()) as slp,
        ):
            out = await self.client.http_post('/p', data={'x': 'y'})
        self.assertEqual(out, {'ok': 1})
        # delay uses backoff * attempt (attempt starts at 1 after first fail)
        slp.assert_awaited()

    async def test_http_post_401_refresh_and_retry(self) -> None:
        """401/403 triggers token refresh then retry."""
        response401 = MagicMock(status_code=401)
        http_err = httpx.HTTPStatusError(
            'e', request=MagicMock(), response=response401,
        )

        fake_resp_ok = MagicMock()
        fake_resp_ok.json.return_value = {'ok': 2}
        fake_resp_ok.raise_for_status.return_value = None

        async def post_side_effect(*_a, **_k):
            if not hasattr(post_side_effect, 'done'):
                setattr(post_side_effect, 'done', True)
                raise http_err
            return fake_resp_ok

        fake_client = MagicMock()
        fake_client.post = AsyncMock(side_effect=post_side_effect)
        cm = MagicMock()
        cm.__aenter__.return_value = fake_client
        cm.__aexit__.return_value = False

        with (
            patch('httpx.AsyncClient', return_value=cm),
            patch.object(
                self.client,
                'auth_headers',
                new=AsyncMock(
                    return_value={'Authorization': 'Bearer Z'},
                ),
            ),
        ):
            out = await self.client.http_post('/p', data={'x': 'y'})
        self.tm.refresh_token.assert_awaited()
        self.assertEqual(out, {'ok': 2})

    async def test_http_post_403_raises(self) -> None:
        """403 bubbles up as an error (no refresh retry loop)."""
        response403 = MagicMock(status_code=403)
        http_err = httpx.HTTPStatusError(
            'e', request=MagicMock(), response=response403,
        )
        fake_client = MagicMock()
        fake_client.post = AsyncMock(side_effect=http_err)
        cm = MagicMock()
        cm.__aenter__.return_value = fake_client
        cm.__aexit__.return_value = False

        with patch('httpx.AsyncClient', return_value=cm):
            with self.assertRaises(httpx.HTTPStatusError):
                await self.client.http_post('/p', data={'x': 'y'})

    async def test_http_post_500_raises_immediately(self) -> None:
        """Non-401/403 HTTP errors should be raised without refresh loop."""
        response500 = MagicMock(status_code=500)
        http_err = httpx.HTTPStatusError(
            'boom', request=MagicMock(), response=response500,
        )
        fake_client = MagicMock()
        fake_client.post = AsyncMock(side_effect=http_err)
        cm = MagicMock()
        cm.__aenter__.return_value = fake_client
        cm.__aexit__.return_value = False
        with patch('httpx.AsyncClient', return_value=cm):
            with self.assertRaises(httpx.HTTPStatusError):
                await self.client.http_post('/p', data={'x': 'y'})

    async def test_ensure_ws_reuse(self) -> None:
        """Return existing open WS if same path."""
        fake_ws = MagicMock()
        type(fake_ws).closed = property(lambda _: False)
        self.client._ws = fake_ws
        self.client._ws_path = '/echo'
        out = await self.client.ensure_ws('/echo')
        self.assertIs(out, fake_ws)

    async def test_ensure_ws_create(self) -> None:
        """Open new session and connect via helper when not present."""
        fake_ws = MagicMock()
        with (
            patch.object(self.client, '_open_new_session', new=AsyncMock()),
            patch.object(
                self.client,
                '_connect_with_retries',
                new=AsyncMock(return_value=fake_ws),
            ),
        ):
            out = await self.client.ensure_ws('/echo')
        self.assertIs(out, fake_ws)
        self.assertEqual(self.client._ws_path, '/echo')

    async def test_ws_send_and_receive_reconnect_then_fail(self) -> None:
        """If closed and reconnection returns None, return None."""
        fake_ws = MagicMock()
        type(fake_ws).closed = property(lambda _: True)
        with (
            patch.object(
                self.client,
                'ensure_ws',
                new=AsyncMock(return_value=fake_ws),
            ),
            patch.object(
                self.client,
                '_reconnect_ws',
                new=AsyncMock(return_value=None),
            ),
        ):
            out = await self.client.ws_send_and_receive('/ws', b'payload')
        self.assertIsNone(out)

    async def test_ws_send_and_receive_send_fail(self) -> None:
        """If send returns False, return None."""
        fake_ws = MagicMock()
        type(fake_ws).closed = property(lambda _: False)
        with (
            patch.object(
                self.client,
                'ensure_ws',
                new=AsyncMock(return_value=fake_ws),
            ),
            patch.object(
                self.client,
                '_send_ws_bytes',
                new=AsyncMock(return_value=False),
            ),
        ):
            out = await self.client.ws_send_and_receive('/ws', b'payload')
        self.assertIsNone(out)

    async def test_ws_send_and_receive_success(self) -> None:
        """Happy path delegates to receive helper for JSON result."""
        fake_ws = MagicMock()
        type(fake_ws).closed = property(lambda _: False)
        with (
            patch.object(
                self.client,
                'ensure_ws',
                new=AsyncMock(return_value=fake_ws),
            ),
            patch.object(
                self.client,
                '_send_ws_bytes',
                new=AsyncMock(return_value=True),
            ),
            patch.object(
                self.client,
                '_receive_ws_json',
                new=AsyncMock(return_value={'a': 1}),
            ),
        ):
            out = await self.client.ws_send_and_receive('/ws', b'payload')
        self.assertEqual(out, {'a': 1})

    async def test_ws_send_and_receive_receive_none(self) -> None:
        """If receive returns None, propagate None to caller."""
        fake_ws = MagicMock()
        type(fake_ws).closed = property(lambda _: False)
        with (
            patch.object(
                self.client, 'ensure_ws',
                new=AsyncMock(return_value=fake_ws),
            ),
            patch.object(
                self.client, '_send_ws_bytes',
                new=AsyncMock(return_value=True),
            ),
            patch.object(
                self.client, '_receive_ws_json',
                new=AsyncMock(return_value=None),
            ),
        ):
            out = await self.client.ws_send_and_receive('/ws', b'payload')
        self.assertIsNone(out)

    async def test_close_closes_ws_and_session(self) -> None:
        """Close both ws and session when open, then clear refs."""
        fake_ws = MagicMock()
        type(fake_ws).closed = property(lambda _: False)
        fake_ws.close = AsyncMock()
        fake_sess = MagicMock()
        type(fake_sess).closed = property(lambda _: False)
        fake_sess.close = AsyncMock()
        self.client._ws = fake_ws
        self.client._session = fake_sess
        await self.client.close()
        fake_ws.close.assert_awaited_once()
        fake_sess.close.assert_awaited_once()
        self.assertIsNone(self.client._ws)
        self.assertIsNone(self.client._session)

    async def test_open_new_session_closes_existing_and_creates(self) -> None:
        """Existing session is closed; new one is created with timeouts."""
        old = MagicMock()
        type(old).closed = property(lambda _: False)
        old.close = AsyncMock()
        self.client._session = old

        # Patch ClientSession and ClientTimeout to observe parameters
        fake_session = MagicMock()
        with (
            patch('aiohttp.ClientSession', return_value=fake_session) as cs,
            patch('aiohttp.ClientTimeout', wraps=aiohttp.ClientTimeout),
        ):
            await self.client._open_new_session()

        old.close.assert_awaited_once()
        cs.assert_called_once()
        self.assertIs(self.client._session, fake_session)

    async def test_make_ws_headers_merges(self) -> None:
        """Extra headers should override base where keys overlap."""
        with patch.object(
            self.client,
            'auth_headers',
            new=AsyncMock(return_value={'A': '1', 'B': '2'}),
        ):
            out = await self.client._make_ws_headers({'B': 'X', 'C': '3'})
        self.assertEqual(out, {'A': '1', 'B': 'X', 'C': '3'})

    async def test_make_ws_headers_extra_none(self) -> None:
        """When extra headers is None, return base auth headers only."""
        with patch.object(
            self.client,
            'auth_headers',
            new=AsyncMock(return_value={'Authorization': 'Bearer Z'}),
        ):
            out = await self.client._make_ws_headers(None)
        self.assertEqual(out, {'Authorization': 'Bearer Z'})

    async def test_connect_with_retries_connector_then_success(self) -> None:
        """Connector error then success; sleeps between attempts."""
        sess = MagicMock()
        sess.ws_connect = AsyncMock(
            side_effect=[
                aiohttp.ClientConnectorError(
                    'h', OSError(),
                ), MagicMock(),
            ],
        )
        self.client._session = sess

        with patch('asyncio.sleep', new=AsyncMock()) as slp:
            ws = await self.client._connect_with_retries(
                '/ws',
                headers=None,
            )
        self.assertIsNotNone(ws)
        slp.assert_awaited()

    async def test_connect_with_retries_401_refresh_then_success(self) -> None:
        """401 triggers token refresh and a retry within attempts."""
        err = aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=401,
            message='unauthorised',
            headers={},
        )
        sess = MagicMock()
        sess.ws_connect = AsyncMock(side_effect=[err, MagicMock()])
        self.client._session = sess
        ws = await self.client._connect_with_retries('/ws', headers=None)
        self.assertIsNotNone(ws)
        self.tm.refresh_token.assert_awaited()

    async def test_connect_with_retries_403_raises(self) -> None:
        """403/404 should be raised immediately without retry loop."""
        err = aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=403,
            message='forbidden',
            headers={},
        )
        sess = MagicMock()
        sess.ws_connect = AsyncMock(side_effect=err)
        self.client._session = sess
        with self.assertRaises(aiohttp.ClientResponseError):
            await self.client._connect_with_retries('/ws', headers=None)

    async def test_receive_ws_json_text_and_binary(self) -> None:
        """TEXT and BINARY messages are decoded to JSON dicts."""
        ws = MagicMock()
        msg_text = SimpleNamespace(
            type=aiohttp.WSMsgType.TEXT, data='{"a":1}',
        )
        msg_bin = SimpleNamespace(
            type=aiohttp.WSMsgType.BINARY, data=b'{"b":2}',
        )

        ws.receive = AsyncMock(side_effect=[msg_text, msg_bin])

        out1 = await self.client._receive_ws_json(ws)
        out2 = await self.client._receive_ws_json(ws)
        self.assertEqual(out1, {'a': 1})
        self.assertEqual(out2, {'b': 2})

    async def test_receive_ws_json_close_1008_triggers_refresh_and_close(
        self,
    ) -> None:
        """Policy violation (1008) requests a token refresh then closes."""
        ws = MagicMock()
        msg = SimpleNamespace(type=aiohttp.WSMsgType.CLOSE, data=1008)
        ws.receive = AsyncMock(return_value=msg)
        with patch.object(self.client, 'close', new=AsyncMock()) as c:
            out = await self.client._receive_ws_json(ws)
        self.tm.refresh_token.assert_awaited()
        c.assert_awaited_once()
        self.assertIsNone(out)

    async def test_http_post_retries_exhausted_timeout(self) -> None:
        """If all POST attempts timeout, last attempt raises ConnectTimeout."""
        fake_client = MagicMock()
        fake_client.post = AsyncMock(side_effect=httpx.ConnectTimeout('t'))
        cm = MagicMock()
        cm.__aenter__.return_value = fake_client
        cm.__aexit__.return_value = False
        with (
            patch('httpx.AsyncClient', return_value=cm),
            patch('asyncio.sleep', new=AsyncMock()),
        ):
            with self.assertRaises(httpx.ConnectTimeout):
                await self.client.http_post(
                    '/p', data={'x': 'y'}, max_retries=2,
                )

    async def test_http_post_zero_retries_raises_runtimeerror(self) -> None:
        """Zero retries path hits the final RuntimeError guard."""
        with self.assertRaises(RuntimeError):
            await self.client.http_post('/p', data={'x': 'y'}, max_retries=0)

    async def test_receive_ws_json_close_without_policy(self) -> None:
        """CLOSE with non-1008 data should close without refreshing token."""
        ws = MagicMock()
        msg = SimpleNamespace(type=aiohttp.WSMsgType.CLOSING, data=1000)
        ws.receive = AsyncMock(return_value=msg)
        with (
            patch.object(self.client, 'close', new=AsyncMock()) as c,
        ):
            out = await self.client._receive_ws_json(ws)
        self.tm.refresh_token.assert_not_awaited()
        c.assert_awaited_once()
        self.assertIsNone(out)

    async def test_receive_ws_json_error_type(self) -> None:
        """ERROR type should close and return None."""
        ws = MagicMock()
        msg = SimpleNamespace(type=aiohttp.WSMsgType.ERROR, data=None)
        ws.receive = AsyncMock(return_value=msg)
        with patch.object(self.client, 'close', new=AsyncMock()) as c:
            out = await self.client._receive_ws_json(ws)
        c.assert_awaited_once()
        self.assertIsNone(out)

    async def test_reconnect_ws_failure_returns_none(self) -> None:
        """If ensure_ws raises, _reconnect_ws returns None."""
        with (
            patch.object(self.client, 'close', new=AsyncMock()) as c,
            patch.object(
                self.client, 'ensure_ws',
                new=AsyncMock(side_effect=Exception('x')),
            ),
        ):
            out = await self.client._reconnect_ws('/ws', headers=None)
        c.assert_awaited_once()
        self.assertIsNone(out)

    async def test_send_ws_bytes_success_and_failure(self) -> None:
        """_send_ws_bytes returns True on success; False on exception and
        closes.
        """
        # Success
        ws_ok = MagicMock()
        ws_ok.send_bytes = AsyncMock(return_value=None)
        ok = await self.client._send_ws_bytes(ws_ok, b'x')
        self.assertTrue(ok)
        # Failure path (RuntimeError)
        ws_ng = MagicMock()
        ws_ng.send_bytes = AsyncMock(side_effect=RuntimeError('closing'))
        with patch.object(self.client, 'close', new=AsyncMock()) as c:
            ng = await self.client._send_ws_bytes(ws_ng, b'y')
        self.assertFalse(ng)
        c.assert_awaited_once()

    async def test_receive_ws_json_unhandled_type_returns_none(self) -> None:
        """Unhandled WS type should return None without closing."""
        ws = MagicMock()
        msg = SimpleNamespace(type=aiohttp.WSMsgType.PING, data=None)
        ws.receive = AsyncMock(return_value=msg)
        with patch.object(self.client, 'close', new=AsyncMock()) as c:
            out = await self.client._receive_ws_json(ws)
        self.assertIsNone(out)
        c.assert_not_awaited()

    async def test_reconnect_ws_success(self) -> None:
        """_reconnect_ws closes then re-establishes the connection."""
        ws_new = MagicMock()
        with (
            patch.object(self.client, 'close', new=AsyncMock()) as c,
            patch.object(
                self.client, 'ensure_ws',
                new=AsyncMock(return_value=ws_new),
            ),
        ):
            out = await self.client._reconnect_ws('/ws', headers={'X': '1'})
        self.assertIs(out, ws_new)
        c.assert_awaited_once()

    async def test_connect_with_retries_exhausted_raises(self) -> None:
        """Exhausted attempts should raise ConnectionError."""
        err = asyncio.TimeoutError()
        sess = MagicMock()
        sess.ws_connect = AsyncMock(side_effect=[err, err])
        self.client._session = sess
        with patch('asyncio.sleep', new=AsyncMock()):
            with self.assertRaises(ConnectionError):
                await self.client._connect_with_retries('/ws', headers=None)

    async def test_receive_ws_json_closing_1008(self) -> None:
        """CLOSING with 1008 should refresh token and close."""
        ws = MagicMock()
        msg = SimpleNamespace(type=aiohttp.WSMsgType.CLOSING, data=1008)
        ws.receive = AsyncMock(return_value=msg)
        with patch.object(self.client, 'close', new=AsyncMock()) as c:
            out = await self.client._receive_ws_json(ws)
        self.tm.refresh_token.assert_awaited()
        c.assert_awaited_once()
        self.assertIsNone(out)

    async def test_open_new_session_skips_close_when_already_closed(
        self,
    ) -> None:
        """If prior session is already closed, do not attempt to close it
        again.
        """
        old = MagicMock()
        type(old).closed = property(lambda _: True)
        old.close = AsyncMock()
        self.client._session = old

        fake_session = MagicMock()
        with patch('aiohttp.ClientSession', return_value=fake_session):
            await self.client._open_new_session()

        old.close.assert_not_awaited()
        self.assertIs(self.client._session, fake_session)

    async def test_receive_ws_json_client_connection_error(self) -> None:
        """ClientConnectionError should log, close and return None."""
        ws = MagicMock()
        ws.receive = AsyncMock(side_effect=aiohttp.ClientConnectionError())
        with patch.object(self.client, 'close', new=AsyncMock()) as c:
            out = await self.client._receive_ws_json(ws)
        self.assertIsNone(out)
        c.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()

'''
pytest \
    --cov=src.net.net_client \
    --cov-report=term-missing \
    tests/src/net/net_client_test.py
'''
