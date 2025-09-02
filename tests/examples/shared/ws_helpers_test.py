from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

from examples.shared import ws_helpers as wh


class TestConfigFlags(unittest.TestCase):
    """
    Unit tests for configuration flags used by WebSocket helpers.
    """

    def test_get_auto_register_jti_true_false(self) -> None:
        """
        Test parsing WS_AUTO_REGISTER_JTI case-insensitively.
        """
        with patch.dict('os.environ', {'WS_AUTO_REGISTER_JTI': 'true'}):
            self.assertTrue(wh.get_auto_register_jti())
        with patch.dict('os.environ', {'WS_AUTO_REGISTER_JTI': 'TRUE'}):
            self.assertTrue(wh.get_auto_register_jti())
        with patch.dict('os.environ', {'WS_AUTO_REGISTER_JTI': 'false'}):
            self.assertFalse(wh.get_auto_register_jti())
        with patch.dict('os.environ', {}, clear=True):
            self.assertFalse(wh.get_auto_register_jti())


class TestTimerAndPayload(unittest.TestCase):
    """
    Unit tests for timer utilities and timeout payload generation.
    """

    def test_start_session_timer_uses_monotonic(self) -> None:
        """
        Test that start_session_timer uses time.monotonic.
        """
        with patch(
            'examples.shared.ws_helpers.time.monotonic',
            return_value=123.45,
        ):
            self.assertEqual(wh.start_session_timer(), 123.45)

    def test_session_timeout_payload_shape(self) -> None:
        """
        Test the structure of session_timeout_payload.
        """
        payload = wh.session_timeout_payload()
        self.assertEqual(
            payload,
            {
                'status': 'closing',
                'reason': 'session_timeout',
                'message': 'WebSocket session reached 5-minute limit.',
            },
        )


class TestTimeoutBehaviour(unittest.IsolatedAsyncioTestCase):
    """
    Behavioural tests for time-based WebSocket session shutdown.
    """

    def __init__(self, methodName: str = 'runTest') -> None:
        super().__init__(methodName)
        # Minimal WebSocket double with an async close method and call log.
        self.ws: SimpleNamespace = SimpleNamespace()
        # Holds a history of (code, reason) tuples for assertions.
        self.ws.closed_calls = []

        async def _close(code: int, reason: str) -> None:
            """Simulate closing the WebSocket connection."""
            self.ws.closed_calls.append((code, reason))

        # Attach the async close to our substitute.
        self.ws.close = _close

    async def test_timeout_closes_and_sends_json(self) -> None:
        """
        Test session timeout with JSON notice and closure.
        """
        ws = self.ws
        # Make session exceed limit: set small limit and large monotonic.
        with (
            patch('examples.shared.ws_helpers.WS_MAX_SESSION_SECONDS', 5.0),
            patch(
                'examples.shared.ws_helpers.time.monotonic',
                return_value=12.0,
            ),
            patch(
                'examples.shared.ws_helpers._safe_websocket_send_json',
                new=AsyncMock(return_value=True),
            ) as mock_send_json,
        ):
            closed = await wh.check_and_maybe_close_on_timeout(
                websocket=ws,
                session_start=0.0,
                client_label='clientA',
                use_text=False,
            )
        self.assertTrue(closed)
        self.assertEqual(
            ws.closed_calls, [
                (1000, 'Session timeout (5 minutes)'),
            ],
        )
        mock_send_json.assert_awaited_once()
        args, kwargs = mock_send_json.call_args
        self.assertEqual(args[0], ws)
        self.assertIsInstance(args[1], dict)
        self.assertEqual(args[2], 'clientA')

    async def test_timeout_closes_and_sends_text(self) -> None:
        """
        Test session timeout with text notice and closure.
        """
        ws = self.ws
        with (
            patch(
                'examples.shared.ws_helpers.WS_MAX_SESSION_SECONDS',
                1.0,
            ),
            patch(
                'examples.shared.ws_helpers.time.monotonic',
                return_value=10.0,
            ),
            patch(
                'examples.shared.ws_helpers._safe_websocket_send_text',
                new=AsyncMock(return_value=True),
            ) as mock_send_text,
        ):
            closed = await wh.check_and_maybe_close_on_timeout(
                websocket=ws,
                session_start=0.0,
                client_label='clientB',
                use_text=True,
            )
        self.assertTrue(closed)
        self.assertEqual(
            ws.closed_calls, [
                (1000, 'Session timeout (5 minutes)'),
            ],
        )
        mock_send_text.assert_awaited_once()
        args, kwargs = mock_send_text.call_args
        self.assertEqual(args[0], ws)
        # First argument after websocket is a JSON string.
        self.assertIsInstance(args[1], str)
        self.assertEqual(args[2], 'clientB')

    async def test_not_timeout_returns_false(self) -> None:
        """
        Test behaviour when session has not timed out.
        """
        ws = self.ws
        with (
            patch('examples.shared.ws_helpers.WS_MAX_SESSION_SECONDS', 60.0),
            patch(
                'examples.shared.ws_helpers.time.monotonic',
                return_value=10.0,
            ),
            patch(
                'examples.shared.ws_helpers._safe_websocket_send_json',
                new=AsyncMock(),
            ) as mock_send_json,
            patch(
                'examples.shared.ws_helpers._safe_websocket_send_text',
                new=AsyncMock(),
            ) as mock_send_text,
        ):
            closed = await wh.check_and_maybe_close_on_timeout(
                websocket=ws,
                session_start=5.0,
                client_label='clientC',
            )
        self.assertFalse(closed)
        self.assertEqual(ws.closed_calls, [])
        mock_send_json.assert_not_called()
        mock_send_text.assert_not_called()


class TestLogEveryN(unittest.TestCase):
    """
    Unit tests for the periodic logging helper.
    """

    def test_logs_only_on_n_multiples(self) -> None:
        """
        Test logging only on exact multiples of n.
        """
        with patch('builtins.print') as mock_print:
            for i in range(1, 11):
                wh.log_every_n('prefix', i, unit='items', n=3)
        # Should print at 3, 6, 9.
        self.assertEqual(mock_print.call_count, 3)
        calls = [c.args[0] for c in mock_print.call_args_list]
        self.assertIn('prefix: Processed 3 items', calls[0])


class TestAuthenticateWsOrNone(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for the authenticate_ws_or_none function.
    """

    def __init__(self, methodName: str = 'runTest') -> None:
        super().__init__(methodName)
        # Minimal WS substitute with a close method and call log.
        self.ws: SimpleNamespace = SimpleNamespace()
        # Holds a history of (code, reason) tuples for assertions.
        self.ws.closed_calls = []

        async def _close(code: int, reason: str) -> None:
            """Simulate closing the WebSocket connection."""
            self.ws.closed_calls.append((code, reason))

        self.ws.close = _close

    async def test_authenticate_success(self) -> None:
        """
        Test successful authentication.
        """
        ws = self.ws
        rds: object = object()
        settings = SimpleNamespace()

        async def fake_auth(
            websocket: object,
            rds_arg: object,
            settings_arg: object,
            *,
            auto_register_jti: bool,
            client_tag: str,
        ) -> tuple[str, str, dict[str, str]]:
            """Simulate successful authentication."""
            # Validate the wrapper forwards parameters verbatim.
            assert websocket is ws
            assert rds_arg is rds
            assert settings_arg is settings
            assert auto_register_jti is True
            assert client_tag == 'TAG'
            return 'alice', 'jti-1', {'role': 'user'}

        with patch(
            'examples.shared.ws_helpers.authenticate_websocket',
            new=fake_auth,
        ):
            res = await wh.authenticate_ws_or_none(
                websocket=ws,
                rds=rds,
                settings=settings,
                auto_register_jti=True,
                client_tag='TAG',
            )
        self.assertEqual(res, ('alice', {'role': 'user'}))

    async def test_authenticate_failure_returns_none_tuple(self) -> None:
        """
        Test behaviour on authentication failure.
        """
        ws = self.ws
        rds: object = object()
        # Provide a minimal settings substitute that satisfies SettingsLike
        settings = SimpleNamespace(
            authjwt_secret_key='s',
            ALGORITHM='HS256',
        )

        async def fake_auth_raise(*args: object, **kwargs: object) -> object:
            """Simulate authentication failure."""
            raise SystemExit('invalid_token')

        with patch(
            'examples.shared.ws_helpers.authenticate_websocket',
            new=fake_auth_raise,
        ):
            res = await wh.authenticate_ws_or_none(
                websocket=ws,
                rds=rds,
                settings=settings,
                auto_register_jti=False,
                client_tag='ANY',
            )
        self.assertEqual(res, (None, None))


if __name__ == '__main__':
    unittest.main()


'''
pytest \
    --cov=examples.shared.ws_helpers \
    --cov-report=term-missing \
    tests/examples/shared/ws_helpers_test.py
'''
