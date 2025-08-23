from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import examples.shared.ws_utils as ws_utils


class TestWSUtils(unittest.IsolatedAsyncioTestCase):
    """
    Tests for the safe WebSocket utilities.
    """

    def setUp(self) -> None:
        """Prepare a mocked WebSocket in a connected state.

        Notes:
            - ``client_state.value`` is set to ``1`` to represent CONNECTED.
            - ``client`` is set truthy to emulate a valid client tuple.
        """

        self.websocket: MagicMock = MagicMock()
        self.websocket.client_state = MagicMock()
        self.websocket.client = True
        # CONNECTED = 1
        self.websocket.client_state.value = 1
        self.client_info: str = 'test-client'

    def test_is_websocket_connected_true(self) -> None:
        """It should return True when the mock indicates a live connection."""

        self.assertTrue(ws_utils._is_websocket_connected(self.websocket))

    def test_is_websocket_connected_no_client_state(self) -> None:
        """It should return False when ``client_state`` is missing."""

        del self.websocket.client_state
        self.assertFalse(ws_utils._is_websocket_connected(self.websocket))

    def test_is_websocket_connected_wrong_state(self) -> None:
        """It should return False when the state is not CONNECTED (1)."""

        self.websocket.client_state.value = 0
        self.assertFalse(ws_utils._is_websocket_connected(self.websocket))

    def test_is_websocket_connected_no_client(self) -> None:
        """It should return False when the client tuple is falsy or missing."""

        self.websocket.client = None
        self.assertFalse(ws_utils._is_websocket_connected(self.websocket))

    def test_is_websocket_connected_exception(self) -> None:
        """Return False if attribute access raises unexpectedly."""

        ws: object = object()
        self.assertFalse(ws_utils._is_websocket_connected(ws))

    def test_is_websocket_connected_exception_branch(self) -> None:
        """It should swallow non-AttributeError exceptions and return False."""

        class Exploding:
            """Helper that raises on any attribute access."""

            def __getattribute__(self, name: str) -> object:
                raise RuntimeError('boom')

        # hasattr raises a non-AttributeError; ws_utils catches it and
        # returns False via the exception handler.
        self.assertFalse(ws_utils._is_websocket_connected(Exploding()))

    async def test_safe_websocket_send_json_success(self) -> None:
        """Sending JSON succeeds when connected and no exception is raised."""

        self.websocket.send_json = AsyncMock()
        result: bool = await ws_utils._safe_websocket_send_json(
            self.websocket, {'a': 1}, self.client_info,
        )
        self.websocket.send_json.assert_awaited_once_with({'a': 1})
        self.assertTrue(result)

    async def test_safe_websocket_send_json_not_connected(self) -> None:
        """Sending JSON is skipped and returns False if not connected."""

        self.websocket.client_state.value = 0
        self.websocket.send_json = AsyncMock()
        result: bool = await ws_utils._safe_websocket_send_json(
            self.websocket, {'a': 1}, self.client_info,
        )
        self.assertFalse(result)
        self.websocket.send_json.assert_not_called()

    async def test_safe_websocket_send_json_exception(self) -> None:
        """Sending JSON returns False when an exception occurs during send."""

        self.websocket.send_json = AsyncMock(side_effect=Exception('fail'))
        result: bool = await ws_utils._safe_websocket_send_json(
            self.websocket, {'a': 1}, self.client_info,
        )
        self.assertFalse(result)

    async def test_safe_websocket_send_text_success(self) -> None:
        """Sending text succeeds when connected and no exception is raised."""

        self.websocket.send_text = AsyncMock()
        result: bool = await ws_utils._safe_websocket_send_text(
            self.websocket, 'hi', self.client_info,
        )
        self.websocket.send_text.assert_awaited_once_with('hi')
        self.assertTrue(result)

    async def test_safe_websocket_send_text_not_connected(self) -> None:
        """Sending text is skipped and returns False if not connected."""

        self.websocket.client_state.value = 0
        self.websocket.send_text = AsyncMock()
        result: bool = await ws_utils._safe_websocket_send_text(
            self.websocket, 'hi', self.client_info,
        )
        self.assertFalse(result)
        self.websocket.send_text.assert_not_called()

    async def test_safe_websocket_send_text_exception(self) -> None:
        """Sending text returns False when an exception occurs during send."""

        self.websocket.send_text = AsyncMock(side_effect=Exception('fail'))
        result: bool = await ws_utils._safe_websocket_send_text(
            self.websocket, 'hi', self.client_info,
        )
        self.assertFalse(result)

    async def test_safe_websocket_send_bytes_success(self) -> None:
        """Sending bytes succeeds when connected and no exception is raised."""

        self.websocket.send_bytes = AsyncMock()
        result: bool = await ws_utils._safe_websocket_send_bytes(
            self.websocket, b'abc', self.client_info,
        )
        self.websocket.send_bytes.assert_awaited_once_with(b'abc')
        self.assertTrue(result)

    async def test_safe_websocket_send_bytes_not_connected(self) -> None:
        """Sending bytes is skipped and returns False if not connected."""

        self.websocket.client_state.value = 0
        self.websocket.send_bytes = AsyncMock()
        result: bool = await ws_utils._safe_websocket_send_bytes(
            self.websocket, b'abc', self.client_info,
        )
        self.assertFalse(result)
        self.websocket.send_bytes.assert_not_called()

    async def test_safe_websocket_send_bytes_exception(self) -> None:
        """Sending bytes returns False when an exception occurs during send."""

        self.websocket.send_bytes = AsyncMock(side_effect=Exception('fail'))
        result: bool = await ws_utils._safe_websocket_send_bytes(
            self.websocket, b'abc', self.client_info,
        )
        self.assertFalse(result)

    async def test_safe_websocket_receive_text_success(self) -> None:
        """Receiving text returns the payload on success when connected."""

        self.websocket.receive_text = AsyncMock(return_value='hello')
        result: str | None = await ws_utils._safe_websocket_receive_text(
            self.websocket, self.client_info,
        )
        self.websocket.receive_text.assert_awaited_once()
        self.assertEqual(result, 'hello')

    async def test_safe_websocket_receive_text_not_connected(self) -> None:
        """Receiving text returns None when the socket is not connected."""

        self.websocket.client_state.value = 0
        self.websocket.receive_text = AsyncMock()
        result: str | None = await ws_utils._safe_websocket_receive_text(
            self.websocket, self.client_info,
        )
        self.assertIsNone(result)
        self.websocket.receive_text.assert_not_called()

    async def test_safe_websocket_receive_text_exception(self) -> None:
        """Receiving text returns None when an exception occurs during read."""

        self.websocket.receive_text = AsyncMock(side_effect=Exception('fail'))
        result: str | None = await ws_utils._safe_websocket_receive_text(
            self.websocket, self.client_info,
        )
        self.assertIsNone(result)

    async def test_safe_websocket_receive_bytes_success(self) -> None:
        """Receiving bytes returns the payload on success when connected."""

        self.websocket.receive_bytes = AsyncMock(return_value=b'xyz')
        result: bytes | None = await ws_utils._safe_websocket_receive_bytes(
            self.websocket, self.client_info,
        )
        self.websocket.receive_bytes.assert_awaited_once()
        self.assertEqual(result, b'xyz')

    async def test_safe_websocket_receive_bytes_not_connected(self) -> None:
        """Receiving bytes returns None when the socket is not connected."""

        self.websocket.client_state.value = 0
        self.websocket.receive_bytes = AsyncMock()
        result: bytes | None = await ws_utils._safe_websocket_receive_bytes(
            self.websocket, self.client_info,
        )
        self.assertIsNone(result)
        self.websocket.receive_bytes.assert_not_called()

    async def test_safe_websocket_receive_bytes_exception(self) -> None:
        """Receiving bytes returns None when a read exception occurs."""

        self.websocket.receive_bytes = AsyncMock(side_effect=Exception('fail'))
        result: bytes | None = await ws_utils._safe_websocket_receive_bytes(
            self.websocket, self.client_info,
        )
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()

'''
pytest \
    --cov=examples.shared.ws_utils \
    --cov-report=term-missing \
    tests/examples/shared/ws_utils_test.py
'''
