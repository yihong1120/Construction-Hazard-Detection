from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import TypedDict
from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import HTTPException

from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.db_management.schemas.auth import AccessTokenSubject
from examples.shared import ws_auth as wa


class AccessPayload(TypedDict):
    subject: AccessTokenSubject
    jti: str
    exp: int


def _access_payload(
    username: str = 'alice',
    jti: str = 'jti',
    user_id: int = 1,
    role: str = 'user',
    exp: int = 1_700_000_000,
) -> AccessPayload:
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
    """WebSocket transport delegates security to the shared HTTP verifier."""

    def make_ws(self, scheme: str = 'wss') -> SimpleNamespace:
        """Build a connection using the public deployment origin."""
        return SimpleNamespace(
            headers={'authorization': 'Bearer external-token'},
            query_params={},
            close=AsyncMock(),
            scope={
                'type': 'websocket', 'scheme': scheme,
                'path': '/ws/detect', 'query_string': b'',
                'headers': [(b'host', b'api.example.com')],
                'server': ('api.example.com', 443),
                'client': ('192.0.2.1', 1234),
            },
        )

    async def test_shared_verifier_receives_connection(self) -> None:
        """OIDC payloads expose mapped identities to existing handlers."""
        for is_oidc in (False, True):
            with self.subTest(is_oidc=is_oidc):
                ws = self.make_ws()
                redis = object()
                subject = _access_payload()['subject']
                credentials = JwtAuthorizationCredentials(
                    subject=subject, payload={'sub': 'provider-id', 'exp': 42},
                    is_oidc=is_oidc,
                )
                with patch.object(
                    wa.jwt_access, 'authenticate_token',
                    AsyncMock(return_value=credentials),
                ) as verify:
                    username, jti, payload = await wa.authenticate_websocket(
                        ws, redis,
                    )
                assert verify.await_args is not None
                request, token, passed_redis = verify.await_args.args
                self.assertEqual(
                    str(request.url),
                    'https://api.example.com/ws/detect',
                )
                self.assertEqual(token, 'external-token')
                self.assertIs(passed_redis, redis)
                self.assertEqual((username, jti), ('alice', 'jti'))
                self.assertEqual(payload['subject'], subject)
                self.assertEqual(payload['exp'], 42)
                self.assertEqual(ws.scope['type'], 'websocket')
                ws.close.assert_not_awaited()

    async def test_missing_token_never_calls_verifier(self) -> None:
        """A missing credential closes the connection."""
        ws = self.make_ws()
        ws.headers = {}
        with patch.object(
            wa.jwt_access, 'authenticate_token', AsyncMock(),
        ) as v:
            with self.assertRaises(SystemExit):
                await wa.authenticate_websocket(ws, object())
            v.assert_not_awaited()
        self.assertEqual(ws.close.await_args.kwargs['code'], 1008)

    async def test_verification_failures_close_connection(self) -> None:
        """Invalid/revoked tokens and deployment failures remain denied."""
        for status in (401, 409, 503):
            with self.subTest(status=status):
                ws = self.make_ws()
                with patch.object(
                    wa.jwt_access, 'authenticate_token',
                    AsyncMock(side_effect=HTTPException(status, 'rejected')),
                ):
                    with self.assertRaises(SystemExit):
                        await wa.authenticate_websocket(
                            ws, object(),
                        )
                self.assertEqual(
                    ws.close.await_args.kwargs['code'],
                    1011 if status == 503 else 1008,
                )

    async def test_query_token_uses_same_verifier(self) -> None:
        """Existing query-token clients retain the same verification policy."""
        ws = self.make_ws('ws')
        ws.headers = {}
        ws.query_params = {'token': 'query-token'}
        credentials = JwtAuthorizationCredentials(
            subject=_access_payload()['subject'],
        )
        with patch.object(
            wa.jwt_access, 'authenticate_token',
            AsyncMock(return_value=credentials),
        ) as verify:
            await wa.authenticate_websocket(ws, object())
        assert verify.await_args is not None
        self.assertEqual(verify.await_args.args[1], 'query-token')
        self.assertEqual(verify.await_args.args[0].url.scheme, 'http')
