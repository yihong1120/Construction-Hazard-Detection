from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace
from typing import Callable
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from starlette.websockets import WebSocketDisconnect

from examples.streaming_web.backend.ws_handlers import DELIMITER
from examples.streaming_web.backend.ws_handlers import handle_frames_ws
from examples.streaming_web.backend.ws_handlers import handle_label_stream_ws
from examples.streaming_web.backend.ws_handlers import handle_stream_ws
from examples.streaming_web.backend.ws_handlers import label_stream_loop
from examples.streaming_web.backend.ws_handlers import parse_and_process_action
from examples.streaming_web.backend.ws_handlers import prepare_label_keys
from examples.streaming_web.backend.ws_handlers import process_stream_action
from examples.streaming_web.backend.ws_handlers import (
    receive_text_with_timeout,
)
from examples.streaming_web.backend.ws_handlers import send_updated_frames
from examples.streaming_web.backend.ws_handlers import store_frame_from_bytes
"""Unit tests for streaming WebSocket handlers.

These tests cover the high-level control flow of the streaming backend
handlers in ``examples.streaming_web.backend.ws_handlers``. The tests
mock Redis and network I/O to focus on:

- Preparing label keys and initial handshake.
- Pushing updated frames and pull-on-demand.
- Streaming loops with timeout / disconnect handling.
- Action parsing and response behaviours.

The intent is to verify contracts and branching behaviour rather than
exercise heavy business logic. British English is used in docstrings
and comments for consistency.
"""


class WsHandlersTest(unittest.IsolatedAsyncioTestCase):
    """End-to-end tests for the streaming WebSocket handler helpers.

    A lightweight WebSocket double is created for each test case. The
    methods simulate ``accept`` and ``close`` and record basic state,
    allowing assertions without a live server.
    """

    # Factory for creating a fake WebSocket object
    make_ws: Callable[..., SimpleNamespace]
    # The default WebSocket instance used by tests
    ws: SimpleNamespace

    def setUp(self) -> None:
        """Build a fresh fake WebSocket and store helper references.

        The fake object records accept/close interactions and exposes a
        ``client.host`` attribute similar to Starlette's WebSocket.
        """

        def _make_ws(host: str = 'virtual') -> SimpleNamespace:
            """Construct a minimal WebSocket-like object for tests.

            Args:
                host: Client host address to record on the object.

            Returns:
                A ``SimpleNamespace`` with ``accept``/``close`` awaitables,
                a ``client.host`` entry, and state for assertions.
            """

            ws = SimpleNamespace()
            ws.client = SimpleNamespace(host=host)
            ws.accepted = False
            # Store close events as (code, reason)
            ws.closed = cast(list[tuple[int | None, str | None]], [])

            async def _accept() -> None:
                ws.accepted = True

            async def _close(
                code: int | None = None,
                reason: str | None = None,
            ) -> None:
                ws.closed.append((code, reason))

            # Attach awaitable methods to mimic WebSocket API
            ws.accept = AsyncMock(side_effect=_accept)
            ws.close = AsyncMock(side_effect=_close)
            return ws

        # Expose factory and a default instance on the test object
        self.make_ws = _make_ws
        self.ws = self.make_ws('virtual')

    async def test_prepare_label_keys_no_keys(self) -> None:
        """Closes the socket when no label keys are available."""
        ws = self.ws
        rds = MagicMock()
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'get_keys_for_label',
                new=AsyncMock(return_value=[]),
        ):
            with patch(
                    'examples.streaming_web.backend.ws_handlers.'
                    '_safe_websocket_send_json',
                    new=AsyncMock(return_value=True),
            ) as send_json:
                res = await prepare_label_keys(
                    ws, rds, 'labelA', '1.2.3.4', 'alice',
                )
                self.assertIsNone(res)
                send_json.assert_awaited()
                # should be closed with reason
                self.assertTrue(ws.closed)

    async def test_prepare_label_keys_with_keys(self) -> None:
        """Returns initial last-id map when label keys exist."""
        ws = self.ws
        rds = MagicMock()
        with patch(
            'examples.streaming_web.backend.'
            'ws_handlers.get_keys_for_label',
            new=AsyncMock(return_value=['k1', 'k2']),
        ):
            res = await prepare_label_keys(ws, rds, 'labelB', '1.2.3.4', 'bob')
            self.assertEqual(res, {'k1': '0', 'k2': '0'})

    async def test_send_updated_frames_basic(self) -> None:
        """Sends one frame successfully and updates the last id."""
        ws = self.ws
        updated = [
            {
                'id': '10',
                'key': 'k1',
                'frame_bytes': b'xxx',
                'warnings': '',
                'cone_polygons': '',
                'pole_polygons': '',
                'detection_items': '',
                'width': 100,
                'height': 80,
            },
        ]
        last_ids: dict[str, str] = {'k1': '0'}
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                '_is_websocket_connected',
                side_effect=[True, True],
        ) as is_conn:
            with patch(
                    'examples.streaming_web.backend.ws_handlers.'
                    '_safe_websocket_send_bytes',
                    new=AsyncMock(return_value=True),
            ) as send_bytes:
                cnt = await send_updated_frames(
                    ws,
                    cast(list[dict[str, str]], updated),
                    last_ids,
                    'ip',
                    'user',
                )
                self.assertEqual(cnt, 1)
                self.assertEqual(last_ids['k1'], '10')
                send_bytes.assert_awaited()
                self.assertTrue(is_conn.called)

    async def test_send_updated_frames_disconnect_during_prepare(self) -> None:
        """Aborts sending when disconnected before transmit."""
        ws = self.ws
        updated = [
            {
                'id': '1', 'key': 'k1', 'frame_bytes': b'abc',
                'warnings': '', 'cone_polygons': '',
                'pole_polygons': '', 'detection_items': '',
                'width': 1, 'height': 1,
            },
        ]
        last_ids: dict[str, str] = {'k1': '0'}
        # First check connected, second check (before send) disconnected
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                '_is_websocket_connected',
                side_effect=[True, False],
        ):
            cnt = await send_updated_frames(
                ws,
                cast(list[dict[str, str]], updated),
                last_ids,
                'ip',
                'user',
            )
            self.assertEqual(cnt, 0)

    async def test_label_stream_loop_timeout_immediate(self) -> None:
        """Returns 0 when timeout check closes immediately."""
        ws = self.ws
        rds = MagicMock()
        last_ids: dict[str, str] = {}
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'check_and_maybe_close_on_timeout',
                new=AsyncMock(return_value=True),
        ):
            cnt = await label_stream_loop(ws, rds, last_ids, 'ip', 'user')
            self.assertEqual(cnt, 0)

    async def test_label_stream_loop_one_iteration(self) -> None:
        """Performs one iteration and returns the send count."""
        ws = self.ws
        rds = MagicMock()
        last_ids: dict[str, str] = {'k1': '0'}
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'check_and_maybe_close_on_timeout',
                side_effect=[False, True],
        ):
            with patch(
                    'examples.streaming_web.backend.ws_handlers.'
                    'fetch_latest_frames',
                    new=AsyncMock(
                        return_value=[
                            {'key': 'k1', 'id': '2', 'frame_bytes': b'x'},
                        ],
                    ),
            ):
                with patch(
                        'examples.streaming_web.backend.ws_handlers.'
                        'send_updated_frames',
                        new=AsyncMock(return_value=2),
                ):
                    with patch(
                            'examples.streaming_web.backend.ws_handlers.'
                            '_is_websocket_connected',
                            return_value=True,
                    ):
                        with patch(
                                'examples.streaming_web.backend.ws_handlers.'
                                'asyncio.sleep',
                                new=AsyncMock(),
                        ):
                            cnt = await label_stream_loop(
                                ws, rds, last_ids, 'ip', 'user',
                            )
                            self.assertEqual(cnt, 2)

    async def test_process_stream_action_ping(self) -> None:
        """Replies to ping and preserves the last id."""
        ws = self.ws
        rds = MagicMock()
        with patch(
            'examples.streaming_web.backend.ws_handlers.'
            '_safe_websocket_send_text',
            new=AsyncMock(return_value=True),
        ) as send_text:
            cont, last = await process_stream_action(
                ws, rds, 'redis:k', {'action': 'ping'}, '0', 'ip', 'user',
            )
            self.assertTrue(cont)
            self.assertEqual(last, '0')
            send_text.assert_awaited()

    async def test_process_stream_action_pull_success_v2(self) -> None:
        """Fetches, sends latest frame, and updates last id."""
        ws = self.ws
        rds = MagicMock()
        frame = {
            'id': '5',
            'frame_bytes': b'data',
            'warnings': '',
            'cone_polygons': '',
            'pole_polygons': '',
            'detection_items': '',
            'width': 10,
            'height': 20,
        }
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'fetch_latest_frame_for_key',
                new=AsyncMock(return_value=frame),
        ):
            with patch(
                    'examples.streaming_web.backend.ws_handlers.'
                    '_is_websocket_connected',
                    side_effect=[True, True],
            ):
                with patch(
                        'examples.streaming_web.backend.ws_handlers.'
                        '_safe_websocket_send_bytes',
                        new=AsyncMock(return_value=True),
                ) as send_bytes:
                    cont, last = await process_stream_action(
                        ws, rds, 'redis:k', {
                            'action': 'pull',
                        }, '0', 'ip', 'user',
                    )
                    self.assertTrue(cont)
                    self.assertEqual(last, '5')
                    send_bytes.assert_awaited()

    async def test_process_stream_action_pull_timeout_v2(self) -> None:
        """Handles fetch timeout and keeps the last id unchanged."""
        ws = self.ws
        rds = MagicMock()
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'fetch_latest_frame_for_key',
                side_effect=asyncio.TimeoutError,
        ):
            with patch(
                    'examples.streaming_web.backend.ws_handlers.'
                    '_safe_websocket_send_text',
                    new=AsyncMock(return_value=True),
            ) as send_text:
                cont, last = await process_stream_action(
                    ws, rds, 'redis:k', {'action': 'pull'}, '7', 'ip', 'user',
                )
                self.assertTrue(cont)
                self.assertEqual(last, '7')
                send_text.assert_awaited()

    async def test_send_updated_frames_skip_empty_bytes(self) -> None:
        """Skips update entries that have empty frame bytes."""
        ws = self.ws
        updated = [
            {'id': '2', 'key': 'k1', 'frame_bytes': b''},
        ]
        last_ids: dict[str, str] = {'k1': '0'}
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                '_is_websocket_connected',
                return_value=True,
        ):
            cnt = await send_updated_frames(
                ws,
                cast(list[dict[str, str]], updated),
                last_ids,
                'ip',
                'user',
            )
            self.assertEqual(cnt, 0)
            self.assertEqual(last_ids['k1'], '0')

    async def test_send_updated_frames_send_failure(self) -> None:
        """Handles send failure and does not increment count."""
        ws = self.ws
        updated = [
            {'id': '3', 'key': 'k1', 'frame_bytes': b'x'},
        ]
        last_ids: dict[str, str] = {'k1': '0'}
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                '_is_websocket_connected',
                side_effect=[True, True],
        ):
            with patch(
                    'examples.streaming_web.backend.ws_handlers.'
                    '_safe_websocket_send_bytes',
                    new=AsyncMock(return_value=False),
            ):
                cnt = await send_updated_frames(
                    ws,
                    cast(list[dict[str, str]], updated),
                    last_ids,
                    'ip',
                    'user',
                )
                self.assertEqual(cnt, 0)

    async def test_send_updated_frames_early_disconnect(self) -> None:
        """Stops early when the socket is not connected."""
        ws = self.ws
        updated = [
            {'id': '4', 'key': 'k1', 'frame_bytes': b'x'},
        ]
        last_ids = {'k1': '0'}
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                '_is_websocket_connected',
                return_value=False,
        ):
            cnt = await send_updated_frames(
                ws,
                cast(list[dict[str, str]], updated),
                last_ids,
                'ip',
                'user',
            )
            self.assertEqual(cnt, 0)

    async def test_label_stream_loop_disconnected(self) -> None:
        """Exits loop when the socket is disconnected."""
        ws = self.ws
        rds = MagicMock()
        last_ids: dict[str, str] = {}
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'check_and_maybe_close_on_timeout',
                new=AsyncMock(return_value=False),
        ):
            with patch(
                    'examples.streaming_web.backend.ws_handlers.'
                    '_is_websocket_connected',
                    return_value=False,
            ):
                cnt = await label_stream_loop(ws, rds, last_ids, 'ip', 'user')
                self.assertEqual(cnt, 0)

    async def test_label_stream_loop_fetch_timeout(self) -> None:
        """Sleeps on fetch timeout and returns 0 updates."""
        ws = self.ws
        rds = MagicMock()
        last_ids: dict[str, str] = {}
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'check_and_maybe_close_on_timeout',
                side_effect=[False, True],
        ):
            with patch(
                    'examples.streaming_web.backend.ws_handlers.'
                    '_is_websocket_connected',
                    return_value=True,
            ):
                with patch(
                        'examples.streaming_web.backend.ws_handlers.'
                        'fetch_latest_frames',
                        new=AsyncMock(side_effect=asyncio.TimeoutError),
                ):
                    with patch(
                            'examples.streaming_web.backend.ws_handlers.'
                            'asyncio.sleep',
                            new=AsyncMock(),
                    ):
                        cnt = await label_stream_loop(
                            ws, rds, last_ids, 'ip', 'user',
                        )
                        self.assertEqual(cnt, 0)

    async def test_label_stream_loop_exception(self) -> None:
        """Catches exceptions from fetch and returns 0 updates."""
        ws = self.ws
        rds = MagicMock()
        last_ids: dict[str, str] = {}
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'check_and_maybe_close_on_timeout',
                new=AsyncMock(return_value=False),
        ):
            with patch(
                    'examples.streaming_web.backend.ws_handlers.'
                    '_is_websocket_connected',
                    return_value=True,
            ):
                with patch(
                        'examples.streaming_web.backend.ws_handlers.'
                        'fetch_latest_frames',
                        new=AsyncMock(side_effect=RuntimeError('boom')),
                ):
                    cnt = await label_stream_loop(
                        ws, rds, last_ids, 'ip', 'user',
                    )
                    self.assertEqual(cnt, 0)

    async def test_label_stream_loop_happy_path(self) -> None:
        """Sends updated frames and returns the number sent."""
        ws = self.ws
        rds = MagicMock()
        last_ids = {'k1': '0'}
        updated = [{
            'id': '8', 'key': 'k1', 'frame_bytes': b'Y',
            'warnings': '',
            'cone_polygons': '', 'pole_polygons': '',
            'detection_items': '',
            'width': 1, 'height': 1,
        }]
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'check_and_maybe_close_on_timeout',
                side_effect=[False, True],
        ):
            with patch(
                    'examples.streaming_web.backend.ws_handlers.'
                    '_is_websocket_connected',
                    return_value=True,
            ):
                with patch(
                        'examples.streaming_web.backend.ws_handlers.'
                        'fetch_latest_frames',
                        new=AsyncMock(return_value=updated),
                ):
                    with patch(
                            'examples.streaming_web.backend.ws_handlers.'
                            'send_updated_frames',
                            new=AsyncMock(return_value=1),
                    ):
                        cnt = await label_stream_loop(
                            ws, rds, last_ids, 'ip', 'user',
                        )
                        self.assertEqual(cnt, 1)

    async def test_label_stream_loop_empty_update(self) -> None:
        """Sleeps when no updates are available and returns 0."""
        ws = self.ws
        rds = MagicMock()
        last_ids = {'k1': '0'}
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'check_and_maybe_close_on_timeout',
                side_effect=[False, True],
        ):
            with patch(
                    'examples.streaming_web.backend.ws_handlers.'
                    '_is_websocket_connected',
                    return_value=True,
            ):
                with patch(
                        'examples.streaming_web.backend.ws_handlers.'
                        'fetch_latest_frames',
                        new=AsyncMock(return_value=[]),
                ):
                    with patch(
                            'examples.streaming_web.backend.ws_handlers.'
                            'asyncio.sleep',
                            new=AsyncMock(),
                    ):
                        cnt = await label_stream_loop(
                            ws, rds, last_ids, 'ip', 'user',
                        )
                        self.assertEqual(cnt, 0)

    async def test_process_stream_action_unknown(self) -> None:
        """Responds to an unknown action and keeps last id."""
        ws = self.ws
        rds = MagicMock()
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                '_safe_websocket_send_text',
                new=AsyncMock(return_value=True),
        ) as send_text:
            cont, last = await process_stream_action(
                ws, rds, 'rk', {'action': 'unknown'}, '9', 'ip', 'user',
            )
            self.assertTrue(cont)
            self.assertEqual(last, '9')
            send_text.assert_awaited()

    async def test_process_stream_action_pull_not_connected_early(
            self,
    ) -> None:
        """Returns False when disconnected before sending."""
        ws = self.ws
        rds = MagicMock()
        frame = {
            'id': '11', 'frame_bytes': b'x', 'warnings': '',
            'cone_polygons': '',
            'pole_polygons': '', 'detection_items': '',
            'width': 1, 'height': 1,
        }
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'fetch_latest_frame_for_key',
                new=AsyncMock(return_value=frame),
        ):
            with patch(
                    'examples.streaming_web.backend.ws_handlers.'
                    '_is_websocket_connected',
                    side_effect=[False],
            ):
                cont, last = await process_stream_action(
                    ws, rds, 'rk', {'action': 'pull'}, '0', 'ip', 'user',
                )
                self.assertFalse(cont)
                self.assertEqual(last, '11')

    async def test_process_stream_action_pull_send_failure(self) -> None:
        """Returns False when sending the frame fails."""
        ws = self.ws
        rds = MagicMock()
        frame = {
            'id': '12', 'frame_bytes': b'x', 'warnings': '',
            'cone_polygons': '',
            'pole_polygons': '', 'detection_items': '',
            'width': 1, 'height': 1,
        }
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'fetch_latest_frame_for_key',
                new=AsyncMock(return_value=frame),
        ):
            with patch(
                    'examples.streaming_web.backend.ws_handlers.'
                    '_is_websocket_connected',
                    side_effect=[True, True],
            ):
                with patch(
                        'examples.streaming_web.backend.ws_handlers.'
                        '_safe_websocket_send_bytes',
                        new=AsyncMock(return_value=False),
                ):
                    cont, last = await process_stream_action(
                        ws, rds, 'rk', {'action': 'pull'}, '0', 'ip', 'user',
                    )
                    self.assertFalse(cont)
                    self.assertEqual(last, '12')

    async def test_process_stream_action_no_frame(self) -> None:
        """Keeps the last id when no frame is available."""
        ws = self.ws
        rds = MagicMock()
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'fetch_latest_frame_for_key',
                new=AsyncMock(return_value=None),
        ):
            cont, last = await process_stream_action(
                ws, rds, 'rk', {'action': 'pull'}, '99', 'ip', 'user',
            )
            self.assertTrue(cont)
            self.assertEqual(last, '99')

    async def test_process_stream_action_pull_success(self) -> None:
        """Sends the frame and updates the last id on success."""
        ws = self.ws
        rds = MagicMock()
        frame = {
            'id': '21', 'frame_bytes': b'x', 'warnings': '',
            'cone_polygons': '',
            'pole_polygons': '', 'detection_items': '',
            'width': 3, 'height': 4,
        }
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'fetch_latest_frame_for_key',
                new=AsyncMock(return_value=frame),
        ):
            with patch(
                    'examples.streaming_web.backend.ws_handlers.'
                    '_is_websocket_connected',
                    side_effect=[True, True],
            ):
                with patch(
                        'examples.streaming_web.backend.ws_handlers.'
                        '_safe_websocket_send_bytes',
                        new=AsyncMock(return_value=True),
                ):
                    cont, last = await process_stream_action(
                        ws, rds, 'rk', {'action': 'pull'}, '0', 'ip', 'user',
                    )
                    self.assertTrue(cont)
                    self.assertEqual(last, '21')

    async def test_process_stream_action_disconnect_during_prepare(
            self,
    ) -> None:
        """Aborts when disconnected between header and send."""
        ws = self.ws
        rds = MagicMock()
        frame = {
            'id': '31', 'frame_bytes': b'x', 'warnings': '',
            'cone_polygons': '',
            'pole_polygons': '', 'detection_items': '',
            'width': 3, 'height': 4,
        }
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'fetch_latest_frame_for_key',
                new=AsyncMock(return_value=frame),
        ):
            # First connectivity check True (before header),
            # second False (before send)
            with patch(
                    'examples.streaming_web.backend.ws_handlers.'
                    '_is_websocket_connected',
                    side_effect=[True, False],
            ):
                cont, last = await process_stream_action(
                    ws, rds, 'rk', {'action': 'pull'}, '0', 'ip', 'user',
                )
                self.assertFalse(cont)
                self.assertEqual(last, '31')

    async def test_process_stream_action_pull_timeout(self) -> None:
        """Reports timeout and preserves the last id value."""
        ws = self.ws
        rds = MagicMock()
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'fetch_latest_frame_for_key',
                new=AsyncMock(side_effect=asyncio.TimeoutError),
        ):
            with patch(
                    'examples.streaming_web.backend.ws_handlers.'
                    '_safe_websocket_send_text',
                    new=AsyncMock(return_value=True),
            ):
                cont, last = await process_stream_action(
                    ws, rds, 'rk', {'action': 'pull'}, '7', 'ip', 'user',
                )
                self.assertTrue(cont)
                self.assertEqual(last, '7')

    async def test_receive_text_with_timeout_msg_none(self) -> None:
        """Returns None when no message is received within time."""
        ws = self.ws
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                '_safe_websocket_receive_text',
                new=AsyncMock(return_value=None),
        ):
            msg = await receive_text_with_timeout(
                ws, 'ip', 'user', timeout=0.1,
            )
            self.assertIsNone(msg)
            self.assertEqual(ws.closed, [])

    async def test_receive_text_with_timeout_timeout_v2(self) -> None:
        """Returns None and records a close on timeout path."""
        ws = self.ws
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                '_safe_websocket_receive_text',
                new=AsyncMock(side_effect=asyncio.TimeoutError),
        ):
            msg = await receive_text_with_timeout(
                ws, 'ip', 'user', timeout=0.01,
            )
            self.assertIsNone(msg)
            self.assertTrue(ws.closed)

    async def test_parse_and_process_action_generic_exception(
            self,
    ) -> None:
        """Logs and replies when processing raises an exception."""
        ws = self.ws
        rds = MagicMock()
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'process_stream_action',
                new=AsyncMock(side_effect=RuntimeError('x')),
        ):
            with patch(
                    'examples.streaming_web.backend.ws_handlers.'
                    '_safe_websocket_send_text',
                    new=AsyncMock(return_value=True),
            ) as send_text:
                cont, last = await parse_and_process_action(
                    '"{}"', ws, rds, 'rk', '0', 'ip', 'user',
                )
                self.assertTrue(cont)
                self.assertEqual(last, '0')
                send_text.assert_awaited()

    async def test_parse_and_process_action_invalid_json_v2(self) -> None:
        """Handles invalid JSON and preserves the last id."""
        ws = self.ws
        rds = MagicMock()
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                '_safe_websocket_send_text',
                new=AsyncMock(return_value=True),
        ) as send_text:
            cont, last = await parse_and_process_action(
                '{bad', ws, rds, 'rk', '0', 'ip', 'user',
            )
            self.assertTrue(cont)
            self.assertEqual(last, '0')
            send_text.assert_awaited()

    async def test_handle_label_stream_ws_no_keys(self) -> None:
        """Accepts and ends early when no label keys are returned."""
        ws = self.ws
        rds = MagicMock()
        settings = MagicMock()
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'authenticate_ws_or_none',
                new=AsyncMock(return_value=('bob', None)),
        ):
            with patch(
                    'examples.streaming_web.backend.ws_handlers.'
                    'prepare_label_keys',
                    new=AsyncMock(return_value=None),
            ):
                await handle_label_stream_ws(ws, 'lab', rds, settings)
                self.assertTrue(ws.accepted)

    async def test_handle_label_stream_ws_disconnect_and_general_exc(
            self,
    ) -> None:
        """Handles disconnect and a general exception on close."""
        ws = self.ws
        rds = MagicMock()
        settings = MagicMock()
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'authenticate_ws_or_none',
                new=AsyncMock(return_value=('user', None)),
        ):
            with patch(
                    'examples.streaming_web.backend.ws_handlers.'
                    'prepare_label_keys',
                    new=AsyncMock(return_value={'k': '0'}),
            ):
                # WebSocketDisconnect path
                with patch(
                        'examples.streaming_web.backend.ws_handlers.'
                        'label_stream_loop',
                        new=AsyncMock(side_effect=WebSocketDisconnect()),
                ):
                    await handle_label_stream_ws(ws, 'lab', rds, settings)
                # General exception path with close raising
                ws2 = self.make_ws()
                with patch(
                        'examples.streaming_web.backend.ws_handlers.'
                        'label_stream_loop',
                        new=AsyncMock(side_effect=RuntimeError('err')),
                ):
                    with patch.object(
                            ws2,
                            'close',
                            new=AsyncMock(
                                side_effect=RuntimeError('close-fail'),
                            ),
                    ):
                        await handle_label_stream_ws(ws2, 'lab', rds, settings)

    async def test_handle_stream_ws_receive_none_and_log_calls(self) -> None:
        """Breaks on None and logs on next loop iteration."""
        ws = self.ws
        rds = MagicMock()
        settings = MagicMock()
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'authenticate_ws_or_none',
                new=AsyncMock(return_value=('user', None)),
        ):
            # Case 1: receive returns None -> break
            with patch(
                    'examples.streaming_web.backend.ws_handlers.'
                    'check_and_maybe_close_on_timeout',
                    new=AsyncMock(return_value=False),
            ):
                with patch(
                        'examples.streaming_web.backend.ws_handlers.'
                        'receive_text_with_timeout',
                        new=AsyncMock(return_value=None),
                ):
                    await handle_stream_ws(ws, 'l', 'k', rds, settings)
            # Case 2: one iteration cont True -> log called,
            # then break by timeout check
            with patch(
                    'examples.streaming_web.backend.ws_handlers.'
                    'check_and_maybe_close_on_timeout',
                    side_effect=[False, True],
            ):
                with patch(
                        'examples.streaming_web.backend.ws_handlers.'
                        'receive_text_with_timeout',
                        new=AsyncMock(return_value='{"action":"ping"}'),
                ):
                    with patch(
                            'examples.streaming_web.backend.ws_handlers.'
                            'parse_and_process_action',
                            new=AsyncMock(return_value=(True, '1')),
                    ):
                        with patch(
                                'examples.streaming_web.backend.ws_handlers.'
                                'log_every_n',
                                new=MagicMock(),
                        ) as log_n:
                            await handle_stream_ws(ws, 'l', 'k', rds, settings)
                            self.assertTrue(log_n.called)

    async def test_handle_stream_ws_parse_returns_false(self) -> None:
        """Exits the loop when the parser indicates to stop."""
        ws = self.ws
        rds = MagicMock()
        settings = MagicMock()
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'authenticate_ws_or_none',
                new=AsyncMock(return_value=('user', None)),
        ):
            with patch(
                    'examples.streaming_web.backend.ws_handlers.'
                    'check_and_maybe_close_on_timeout',
                    new=AsyncMock(return_value=False),
            ):
                with patch(
                        'examples.streaming_web.backend.ws_handlers.'
                        'receive_text_with_timeout',
                        new=AsyncMock(return_value='{"action":"pull"}'),
                ):
                    with patch(
                            'examples.streaming_web.backend.ws_handlers.'
                            'parse_and_process_action',
                            new=AsyncMock(return_value=(False, '2')),
                    ):
                        await handle_stream_ws(ws, 'l', 'k', rds, settings)

    async def test_handle_stream_ws_auth_fail(self) -> None:
        """Accepts then returns early when authentication fails."""
        ws = self.ws
        rds = MagicMock()
        settings = MagicMock()
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'authenticate_ws_or_none',
                new=AsyncMock(return_value=(None, None)),
        ):
            await handle_stream_ws(ws, 'lab', 'k', rds, settings)

    async def test_handle_frames_ws_timeout(self) -> None:
        """Closes the socket when a receive timeout occurs."""
        ws = self.ws
        rds = MagicMock()
        settings = MagicMock()
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'authenticate_ws_or_none',
                new=AsyncMock(return_value=('user', None)),
        ):
            with patch(
                    'examples.streaming_web.backend.ws_handlers.'
                    'check_and_maybe_close_on_timeout',
                    new=AsyncMock(return_value=False),
            ):
                with patch(
                        'examples.streaming_web.backend.ws_handlers.'
                        '_safe_websocket_receive_bytes',
                        new=AsyncMock(side_effect=asyncio.TimeoutError),
                ):
                    with patch.object(ws, 'close', new=AsyncMock()) as close:
                        await handle_frames_ws(ws, rds, settings)
                        close.assert_awaited()

    async def test_handle_frames_ws_value_error(self) -> None:
        """Sends an error JSON when payload is not delimited."""
        ws = self.ws
        rds = MagicMock()
        settings = MagicMock()
        bad_payload = b'not-delimited'
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'authenticate_ws_or_none',
                new=AsyncMock(return_value=('user', None)),
        ):
            with patch(
                    'examples.streaming_web.backend.ws_handlers.'
                    '_safe_websocket_receive_bytes',
                    new=AsyncMock(side_effect=[bad_payload, None]),
            ):
                with patch(
                        'examples.streaming_web.backend.ws_handlers.'
                        '_safe_websocket_send_json',
                        new=AsyncMock(return_value=True),
                ) as send_json:
                    await handle_frames_ws(ws, rds, settings)
                    send_json.assert_awaited()

    async def test_handle_frames_ws_generic_exception(self) -> None:
        """Sends an error JSON when storing to Redis fails."""
        ws = self.ws
        rds = MagicMock()
        settings = MagicMock()
        header = json.dumps(
            {'label': 'l', 'key': 'k', 'width': 1, 'height': 2},
        ).encode()
        payload = header + DELIMITER + b'B'
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'authenticate_ws_or_none',
                new=AsyncMock(return_value=('user', None)),
        ):
            with patch(
                    'examples.streaming_web.backend.ws_handlers.'
                    '_safe_websocket_receive_bytes',
                    new=AsyncMock(side_effect=[payload, None]),
            ):
                with patch(
                        'examples.streaming_web.backend.ws_handlers.'
                        'store_to_redis',
                        new=AsyncMock(side_effect=RuntimeError('store-fail')),
                ):
                    with patch(
                            'examples.streaming_web.backend.ws_handlers.'
                            '_safe_websocket_send_json',
                            new=AsyncMock(return_value=True),
                    ) as send_json:
                        await handle_frames_ws(ws, rds, settings)
                        send_json.assert_awaited()

    async def test_handle_frames_ws_early_timeout_check(self) -> None:
        """Returns early when the pre-timeout check triggers."""
        ws = self.ws
        rds = MagicMock()
        settings = MagicMock()
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'authenticate_ws_or_none',
                new=AsyncMock(return_value=('user', None)),
        ):
            with patch(
                    'examples.streaming_web.backend.ws_handlers.'
                    'check_and_maybe_close_on_timeout',
                    new=AsyncMock(return_value=True),
            ):
                await handle_frames_ws(ws, rds, settings)

    async def test_handle_frames_ws_auth_fail(self) -> None:
        """Returns early when authentication is not provided."""
        ws = self.ws
        rds = MagicMock()
        settings = MagicMock()
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'authenticate_ws_or_none',
                new=AsyncMock(return_value=(None, None)),
        ):
            await handle_frames_ws(ws, rds, settings)

    async def test_handle_frames_ws_receive_none(self) -> None:
        """Returns when no bytes are received from the socket."""
        ws = self.ws
        rds = MagicMock()
        settings = MagicMock()
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'authenticate_ws_or_none',
                new=AsyncMock(return_value=('user', None)),
        ):
            with patch(
                    'examples.streaming_web.backend.ws_handlers.'
                    'check_and_maybe_close_on_timeout',
                    new=AsyncMock(return_value=False),
            ):
                with patch(
                        'examples.streaming_web.backend.ws_handlers.'
                        '_safe_websocket_receive_bytes',
                        new=AsyncMock(return_value=None),
                ):
                    await handle_frames_ws(ws, rds, settings)

    async def test_store_frame_from_bytes(self) -> None:
        """Stores the payload into Redis and acknowledges it."""
        ws = self.ws
        rds = MagicMock()
        header = {
            'label': 'lab', 'key': 'k', 'warnings_json': 'w',
            'cone_polygons_json': 'c', 'pole_polygons_json': 'p',
            'detection_items_json': 'd', 'width': 10, 'height': 20,
        }
        payload = json.dumps(header).encode('utf-8') + \
            DELIMITER + b'FRAMEBYTES'
        with patch(
                'examples.streaming_web.backend.ws_handlers.store_to_redis',
                new=AsyncMock(),
        ) as store:
            with patch(
                    'examples.streaming_web.backend.ws_handlers.'
                    '_safe_websocket_send_json',
                    new=AsyncMock(return_value=True),
            ) as send_json:
                await store_frame_from_bytes(ws, rds, payload, 'ip', 'user')
                store.assert_awaited_once()
                send_json.assert_awaited()

    async def test_receive_text_with_timeout_ok(self) -> None:
        """Returns the received text when within the timeout window."""
        ws = self.ws
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                '_safe_websocket_receive_text',
                new=AsyncMock(return_value='hello'),
        ):
            msg = await receive_text_with_timeout(
                ws, 'ip', 'user', timeout=0.1,
            )
            self.assertEqual(msg, 'hello')

    async def test_receive_text_with_timeout_timeout(self) -> None:
        """Closes and returns None when a timeout occurs."""
        ws = self.ws
        # Make the inner receive raise TimeoutError when awaited
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                '_safe_websocket_receive_text',
                new=AsyncMock(side_effect=asyncio.TimeoutError),
        ):
            with patch.object(ws, 'close', new=AsyncMock()) as close:
                msg = await receive_text_with_timeout(
                    ws, 'ip', 'user', timeout=0.01,
                )
                self.assertIsNone(msg)
                close.assert_awaited()

    async def test_parse_and_process_action_success(self) -> None:
        """Parses valid JSON and processes the action successfully."""
        ws = self.ws
        rds = MagicMock()
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'process_stream_action',
                new=AsyncMock(return_value=(True, '9')),
        ):
            cont, last = await parse_and_process_action(
                '{"action":"ping"}', ws, rds, 'rk', '0', 'ip', 'user',
            )
            self.assertTrue(cont)
            self.assertEqual(last, '9')

    async def test_parse_and_process_action_invalid_json(self) -> None:
        """Handles invalid JSON input and keeps last id unchanged."""
        ws = self.ws
        rds = MagicMock()
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                '_safe_websocket_send_text',
                new=AsyncMock(return_value=True),
        ) as send_text:
            cont, last = await parse_and_process_action(
                '{invalid', ws, rds, 'rk', '42', 'ip', 'user',
            )
            self.assertTrue(cont)
            self.assertEqual(last, '42')
            send_text.assert_awaited()

    async def test_handle_label_stream_ws_auth_fail(self) -> None:
        """Accepts but exits early when authentication fails."""
        ws = self.ws
        rds = MagicMock()
        settings = MagicMock()
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'authenticate_ws_or_none',
                new=AsyncMock(return_value=(None, None)),
        ):
            await handle_label_stream_ws(ws, 'lab', rds, settings)
            # accepted but returns early after auth fail
            self.assertTrue(ws.accepted)

    async def test_handle_stream_ws_immediate_close_check(self) -> None:
        """Returns after immediate timeout check, post-accept."""
        ws = self.ws
        rds = MagicMock()
        settings = MagicMock()
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'authenticate_ws_or_none',
                new=AsyncMock(return_value=('user', None)),
        ):
            with patch(
                    'examples.streaming_web.backend.ws_handlers.'
                    'check_and_maybe_close_on_timeout',
                    new=AsyncMock(return_value=True),
            ):
                await handle_stream_ws(ws, 'lab', 'k', rds, settings)
                self.assertTrue(ws.accepted)

    async def test_handle_frames_ws_receive_and_store(self) -> None:
        """Receives a payload, stores it, and acknowledges success."""
        ws = self.ws
        rds = MagicMock()
        settings = MagicMock()
        header = json.dumps(
            {'label': 'l', 'key': 'k', 'width': 1, 'height': 2},
        ).encode()
        payload = header + DELIMITER + b'B'
        with patch(
                'examples.streaming_web.backend.ws_handlers.'
                'authenticate_ws_or_none',
                new=AsyncMock(return_value=('user', None)),
        ):
            with patch(
                    'examples.streaming_web.backend.ws_handlers.'
                    '_safe_websocket_receive_bytes',
                    side_effect=[payload, None],
            ):
                with patch(
                        'examples.streaming_web.backend.ws_handlers.'
                        'store_to_redis',
                        new=AsyncMock(),
                ) as store:
                    with patch(
                            'examples.streaming_web.backend.ws_handlers.'
                            '_safe_websocket_send_json',
                            new=AsyncMock(return_value=True),
                    ) as send_json:
                        await handle_frames_ws(ws, rds, settings)
                        self.assertTrue(ws.accepted)
                        store.assert_awaited()
                        send_json.assert_awaited()


if __name__ == '__main__':
    unittest.main()

'''
pytest \
    --cov=examples.streaming_web.backend.ws_handlers \
    --cov-report=term-missing \
    tests/examples/streaming_web/backend/ws_handlers_test.py
'''
