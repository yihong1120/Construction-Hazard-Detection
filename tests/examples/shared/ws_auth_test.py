from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

from examples.shared import ws_auth as wa


def _access_payload(
    username: str = 'alice',
    jti: str = 'jti',
    user_id: int = 1,
    role: str = 'user',
    exp: int = 1_700_000_000,
) -> dict[str, object]:
    """Perform access payload.

    Args:
        username: Value used by this callable.
        jti: Value used by this callable.
        user_id: Value used by this callable.
        role: Value used by this callable.
        exp: Value used by this callable.

    Returns:
        The callable result.
    """
    return {
        'subject': {
            'username': username,
            'user_id': user_id,
            'role': role,
            'jti': jti,
            'features': [],
        },
        'jti': jti,
        'exp': exp,
    }


class TestExtractionHelpers(unittest.TestCase):
    """Unit tests for token and model-key extraction helper functions."""

    def __init__(self, methodName: str = 'runTest') -> None:
        """Support __init__."""
        super().__init__(methodName)
        # Factory for creating a minimal WebSocket-like object.

        def _make_ws(
            headers: dict[str, str] | None = None,
            query_params: dict[str, str] | None = None,
        ) -> SimpleNamespace:
            """Create a tiny WebSocket-like object for tests.

            Args:
                headers (dict[str, str] | None):
                    Optional mapping of header names to values.
                query_params (dict[str, str] | None):
                    Optional mapping of query names to values.

            Returns:
                SimpleNamespace: A WebSocket-like object with `headers`,
                `query_params`, an async `close` method, and a `closed` log.
            """
            ws = SimpleNamespace()
            ws.headers = headers or {}
            ws.query_params = query_params or {}
            ws.closed = []

            async def _close(code: int, reason: str) -> None:
                """Simulate closing the WebSocket connection."""
                ws.closed.append((code, reason))

            ws.close = _close
            return ws

        self.make_ws = _make_ws

    def test_extract_token_from_header(self) -> None:
        """Test extracting token from the Authorisation header."""
        ws = self.make_ws(
            headers={
                'authorization': 'Bearer abc.def.ghi',
            },
        )
        self.assertEqual(wa.extract_token_from_ws(ws), 'abc.def.ghi')

    def test_extract_token_from_query(self) -> None:
        """Test extracting token from the `token` query parameter."""
        ws = self.make_ws(
            query_params={
                'token': 't123',
            },
        )
        self.assertEqual(wa.extract_token_from_ws(ws), 't123')

    def test_extract_token_missing(self) -> None:
        """Test behaviour when neither header nor query provides a token."""
        ws = self.make_ws()
        self.assertIsNone(wa.extract_token_from_ws(ws))

    def test_get_model_key_header_then_query(self) -> None:
        """Test prioritising model key extraction from header over query."""
        ws1 = self.make_ws(
            headers={
                'x-model-key': 'mA',
            },
        )
        self.assertEqual(wa.get_model_key_from_ws(ws1), 'mA')

        ws2 = self.make_ws(
            query_params={
                'model': 'mB',
            },
        )
        self.assertEqual(wa.get_model_key_from_ws(ws2), 'mB')

        ws3 = self.make_ws()
        self.assertIsNone(wa.get_model_key_from_ws(ws3))


class TestAuthenticateWebsocket(unittest.IsolatedAsyncioTestCase):
    """End-to-end tests for the `authenticate_websocket` function."""

    def __init__(self, methodName: str = 'runTest') -> None:
        """Support __init__."""
        super().__init__(methodName)
        # Factory for creating a minimal WebSocket-like object.

        def _make_ws(
            headers: dict[str, str] | None = None,
            query_params: dict[str, str] | None = None,
        ) -> SimpleNamespace:
            """Create a tiny WebSocket-like object for tests.

            See `TestExtractionHelpers.make_ws` for details.
            """
            ws = SimpleNamespace()
            ws.headers = headers or {}
            ws.query_params = query_params or {}
            ws.closed = []

            async def _close(code: int, reason: str) -> None:
                """Simulate closing the WebSocket connection."""
                ws.closed.append((code, reason))

            ws.close = _close
            return ws

        self.make_ws = _make_ws
        # Minimal settings substitute used by `jwt.decode` in target code.
        self.settings = SimpleNamespace(
            authjwt_secret_key='secret',
            ALGORITHM='HS256',
        )

    async def test_success_when_jti_active(self) -> None:
        """Test successful authentication when JTI is active in cache."""
        payload = _access_payload(jti='j1')

        ws = self.make_ws(
            headers={
                'authorization': 'Bearer token',
            },
        )
        rds = object()
        settings = self.settings

        with (
            patch('examples.shared.ws_auth.jwt.decode', return_value=payload),
            patch(
                'examples.shared.ws_auth.prune_user_cache',
                new=AsyncMock(),
            ) as mock_prune,
            patch(
                'examples.shared.ws_auth.rate_limiter_service.get_user_data',
                new=AsyncMock(return_value={'jti_list': ['j1']}),
            ) as mock_get,
            patch(
                'examples.shared.ws_auth.rate_limiter_service.set_user_data',
                new=AsyncMock(),
            ) as mock_set,
        ):
            username, jti, out_payload = await wa.authenticate_websocket(
                ws,
                rds,
                settings,
                auto_register_jti=False,
                client_tag='[T]',
            )

        self.assertEqual((username, jti), ('alice', 'j1'))
        self.assertEqual(out_payload, payload)
        mock_prune.assert_awaited_once()
        mock_get.assert_awaited_once()
        mock_set.assert_not_awaited()
        self.assertEqual(ws.closed, [])

    async def test_success_auto_register_missing_jti(self) -> None:
        """Test auto-registration appends and persists missing JTI."""
        payload = _access_payload(
            username='bob',
            jti='j2',
            user_id=7,
            exp=1_700_000_100,
        )
        ws = self.make_ws(
            headers={
                'authorization': 'Bearer token',
            },
        )
        rds = object()
        settings = self.settings

        with (
            patch('examples.shared.ws_auth.jwt.decode', return_value=payload),
            patch('examples.shared.ws_auth.prune_user_cache', new=AsyncMock()),
            patch(
                'examples.shared.ws_auth.rate_limiter_service.get_user_data',
                new=AsyncMock(
                    return_value={
                        'db_user': {
                            'id': 7,
                            'username': 'bob',
                            'role': 'user',
                            'group_id': None,
                            'status': 'active',
                        },
                        'jti_list': [],
                        'jti_meta': {},
                        'refresh_tokens': [],
                        'refresh_token_hashes': [],
                        'refresh_token_families': {},
                        'feature_names': [],
                    },
                ),
            ) as mock_get,
            patch(
                'examples.shared.ws_auth.rate_limiter_service.set_user_data',
                new=AsyncMock(),
            ) as mock_set,
        ):
            username, jti, out_payload = await wa.authenticate_websocket(
                ws,
                rds,
                settings,
                auto_register_jti=True,
                client_tag='[T]',
            )

        self.assertEqual((username, jti), ('bob', 'j2'))
        self.assertEqual(out_payload, payload)
        mock_get.assert_awaited_once()
        mock_set.assert_awaited()
        # Ensure jti was registered in cache argument
        args, kwargs = mock_set.call_args
        self.assertEqual(args[0], rds)
        self.assertEqual(args[1], 'bob')
        cache = args[2]
        self.assertIn('j2', cache['jti_list'])
        self.assertIn('jti_meta', cache)
        self.assertIn('j2', cache['jti_meta'])
        self.assertEqual(ws.closed, [])

    async def test_missing_token_closes_and_exits(self) -> None:
        """Test closing socket and raising SystemExit on missing token."""
        ws = self.make_ws()  # no headers or query
        with self.assertRaises(SystemExit) as cm:
            await wa.authenticate_websocket(ws, object(), self.settings)
        self.assertEqual(str(cm.exception), 'missing_token')
        self.assertTrue(ws.closed)
        code, reason = ws.closed[-1]
        self.assertEqual(code, 1008)
        self.assertIn('Missing authentication token', reason)

    async def test_invalid_token_decode(self) -> None:
        """Test close and SystemExit on invalid token decode."""
        ws = self.make_ws(headers={'authorization': 'Bearer bad'})
        with patch(
            'examples.shared.ws_auth.jwt.decode',
            side_effect=Exception('boom'),
        ):
            with self.assertRaises(SystemExit) as cm:
                await wa.authenticate_websocket(
                    ws,
                    object(),
                    self.settings,
                )
        self.assertEqual(str(cm.exception), 'invalid_token')
        self.assertTrue(ws.closed)
        code, reason = ws.closed[-1]
        self.assertEqual(code, 1008)
        self.assertIn('Invalid token', reason)

    async def test_empty_payload_is_invalid_token(self) -> None:
        """Empty token payloads fail the access-token schema."""
        ws = self.make_ws(headers={'authorization': 'Bearer t'})
        with patch('examples.shared.ws_auth.jwt.decode', return_value={}):
            with self.assertRaises(SystemExit) as cm:
                await wa.authenticate_websocket(ws, object(), self.settings)
        self.assertEqual(str(cm.exception), 'invalid_token')
        code, reason = ws.closed[-1]
        self.assertEqual(code, 1008)
        self.assertIn('Invalid token', reason)

    async def test_rejects_incomplete_access_subject(self) -> None:
        """Incomplete access-token subjects fail at the decode boundary."""
        ws = self.make_ws(headers={'authorization': 'Bearer t'})
        payload = {'subject': {'username': 'eve'}}  # missing jti
        with patch('examples.shared.ws_auth.jwt.decode', return_value=payload):
            with self.assertRaises(SystemExit) as cm:
                await wa.authenticate_websocket(ws, object(), self.settings)
        self.assertEqual(str(cm.exception), 'invalid_token')
        code, reason = ws.closed[-1]
        self.assertEqual(code, 1008)
        self.assertIn('Invalid token', reason)

    async def test_jti_not_active_and_no_auto_register(self) -> None:
        """Test raise and close on inactive JTI without auto-register."""
        from examples.shared import ws_auth as wa

        ws = self.make_ws(headers={'authorization': 'Bearer t'})
        payload = _access_payload(username='zoe', jti='JX')
        with (
            patch('examples.shared.ws_auth.jwt.decode', return_value=payload),
            patch('examples.shared.ws_auth.prune_user_cache', new=AsyncMock()),
            patch(
                'examples.shared.ws_auth.rate_limiter_service.get_user_data',
                new=AsyncMock(return_value={'jti_list': []}),
            ),
        ):
            with self.assertRaises(SystemExit) as cm:
                await wa.authenticate_websocket(
                    ws,
                    object(),
                    self.settings,
                    auto_register_jti=False,
                )
        self.assertEqual(str(cm.exception), 'jti_not_active')
        code, reason = ws.closed[-1]
        self.assertEqual(code, 1008)
        self.assertIn('Token not active', reason)

    async def test_auto_register_with_no_cached_user_creates_default_cache(
        self,
    ) -> None:
        """Test auto register with no cached user creates default cache."""
        payload = _access_payload(
            username='neo',
            jti='jN',
            user_id=42,
            role='admin',
            exp=1_700_000_200,
        )
        ws = self.make_ws(
            headers={
                'authorization': 'Bearer token',
            },
        )
        rds = object()
        settings = self.settings

        with (
            patch('examples.shared.ws_auth.jwt.decode', return_value=payload),
            patch('examples.shared.ws_auth.prune_user_cache', new=AsyncMock()),
            patch(
                'examples.shared.ws_auth.rate_limiter_service.get_user_data',
                new=AsyncMock(return_value=None),
            ),
            patch(
                'examples.shared.ws_auth.rate_limiter_service.set_user_data',
                new=AsyncMock(),
            ) as mock_set,
        ):
            username, jti, out_payload = await wa.authenticate_websocket(
                ws,
                rds,
                settings,
                auto_register_jti=True,
                client_tag='[T]',
            )

        self.assertEqual((username, jti), ('neo', 'jN'))
        self.assertEqual(out_payload, payload)
        args, _ = mock_set.call_args
        # args: rds, username, cache
        self.assertEqual(args[0], rds)
        self.assertEqual(args[1], 'neo')
        cache = args[2]
        # default structure should exist
        self.assertIn('db_user', cache)
        self.assertIn('jti_list', cache)
        self.assertIn('jN', cache['jti_list'])
        # jti_meta recorded from exp
        self.assertIn('jti_meta', cache)
        self.assertEqual(cache['jti_meta']['jN'], 1700000200)

    async def test_auto_register_with_existing_jti_meta_dict(self) -> None:
        """Test preservation of existing jti_meta entries and add current
        JTI."""
        from examples.shared import ws_auth as wa

        payload = _access_payload(
            username='smith',
            jti='jS',
            exp=1_700_000_500,
        )
        ws = self.make_ws(
            headers={
                'authorization': 'Bearer token',
            },
        )
        rds = object()
        settings = self.settings

        cache_with_meta = {
            'db_user': {
                'id': 1,
                'username': 'smith',
                'role': 'user',
                'group_id': None,
                'status': 'active',
            },
            'jti_list': [],
            'jti_meta': {'old': 1},
            'refresh_tokens': [],
            'refresh_token_hashes': [],
            'refresh_token_families': {},
            'feature_names': [],
        }

        with (
            patch('examples.shared.ws_auth.jwt.decode', return_value=payload),
            patch('examples.shared.ws_auth.prune_user_cache', new=AsyncMock()),
            patch(
                'examples.shared.ws_auth.rate_limiter_service.get_user_data',
                new=AsyncMock(return_value=cache_with_meta),
            ),
            patch(
                'examples.shared.ws_auth.rate_limiter_service.set_user_data',
                new=AsyncMock(),
            ) as mock_set,
        ):
            username, jti, _ = await wa.authenticate_websocket(
                ws,
                rds,
                settings,
                auto_register_jti=True,
                client_tag='[T]',
            )

        self.assertEqual((username, jti), ('smith', 'jS'))
        args, _ = mock_set.call_args
        cache = args[2]
        self.assertIsNot(cache, cache_with_meta)
        self.assertIn('jS', cache['jti_list'])
        self.assertIn('old', cache['jti_meta'])
        self.assertIn('jS', cache['jti_meta'])


if __name__ == '__main__':
    unittest.main()
