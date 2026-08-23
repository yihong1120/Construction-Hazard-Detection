from __future__ import annotations

import asyncio
import json
import unittest
from collections.abc import Coroutine
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import WebSocketDisconnect

from examples.auth.config import Settings
from examples.streaming_web import streaming_metadata_handlers as handlers
from examples.streaming_web.schemas import FrameOutData


def _frame() -> FrameOutData:
    """Build one valid metadata record for transport helper tests.

    Returns:
        Compact metadata payload emitted by the Redis reader.
    """
    return {
        'id': '17-0',
        'has_warning': True,
        'key': 'Camera A',
        'stream_id': '12',
        'redis_key': 'metadata:site-a:camera-a',
    }


class TestMetadataSseHelpers(unittest.IsolatedAsyncioTestCase):
    """Verify compact SSE helper branches without a Redis server."""

    def test_encode_and_frame_helpers_handle_events_and_heartbeats(
        self,
    ) -> None:
        """SSE events include IDs and idle frames keep the connection alive."""
        encoded = handlers._encode_sse_event(
            {'id': '17-0', 'has_warning': True},
        )
        no_id = handlers._encode_sse_event({'has_warning': False})
        last_id, heartbeat, event = handlers._metadata_frame_event(
            _frame(),
            'metadata-key',
            '',
            20.0,
            1.0,
        )
        idle_id, idle_heartbeat, idle_event = handlers._metadata_frame_event(
            None,
            'metadata-key',
            last_id,
            40.0,
            heartbeat,
        )

        self.assertIn(b'id: 17-0', encoded)
        self.assertNotIn(b'id:', no_id)
        self.assertEqual(last_id, '17-0')
        self.assertEqual(heartbeat, 20.0)
        self.assertIn(b'has_warning', cast(bytes, event))
        self.assertEqual(idle_id, '17-0')
        self.assertEqual(idle_heartbeat, 40.0)
        self.assertEqual(idle_event, b': keepalive\n\n')

    async def test_overlay_demand_and_ready_helpers_cover_due_states(
        self,
    ) -> None:
        """Overlay demand is renewed only when due and ready once present."""
        rds = MagicMock()
        rds.set = AsyncMock()
        rds.exists = AsyncMock(return_value=1)

        unchanged = await handlers._refresh_overlay_demand_if_due(
            rds,
            None,
            30,
            10.0,
            4.0,
            5.0,
        )
        refreshed = await handlers._refresh_overlay_demand_if_due(
            rds,
            'overlay:demand',
            30,
            10.0,
            4.0,
            14.0,
        )
        ready = await handlers._overlay_ready_event(
            rds,
            'overlay:ready',
            {'stream_id': '12'},
        )
        sent, repeated = await handlers._next_overlay_ready_event(
            rds,
            'overlay:ready',
            {'stream_id': '12'},
            True,
        )

        self.assertEqual(unchanged, 4.0)
        self.assertEqual(refreshed, 14.0)
        rds.set.assert_awaited_once_with('overlay:demand', b'1', ex=30)
        self.assertIn(b'overlay_ready', cast(bytes, ready))
        self.assertTrue(sent)
        self.assertIsNone(repeated)

    def test_error_and_heartbeat_helpers_rate_limit_events(self) -> None:
        """Redis failures emit one event then fall back to heartbeats."""
        event_state = handlers._metadata_read_error_event(
            'metadata-key',
            RuntimeError('offline'),
            20.0,
            float('-inf'),
            float('-inf'),
            1.0,
        )
        idle_state = handlers._metadata_read_error_event(
            'metadata-key',
            RuntimeError('offline'),
            21.0,
            event_state[0],
            event_state[1],
            event_state[2],
        )
        _, inactive_heartbeat = handlers._heartbeat_event(5.0, 1.0)

        self.assertIn(b'redis_error', cast(bytes, event_state[3]))
        self.assertIsNone(idle_state[3])
        self.assertIsNone(inactive_heartbeat)


class TestMetadataWebSocketHelpers(unittest.IsolatedAsyncioTestCase):
    """Verify WebSocket message helper behaviour with safe-send doubles."""

    async def test_receive_task_handles_disconnect_invalid_json_and_ping(
        self,
    ) -> None:
        """Completed client tasks preserve disconnect and ping semantics."""
        websocket = MagicMock()

        async def receive(value: str | None) -> str | None:
            """Return a completed receive value for one helper invocation.

            Args:
                value: Client text, or ``None`` after a disconnect.

            Returns:
                The requested completed task value.
            """
            return value

        disconnected = asyncio.create_task(receive(None))
        await disconnected
        invalid = asyncio.create_task(receive('{not json'))
        await invalid
        ping = asyncio.create_task(receive(json.dumps({'action': 'ping'})))
        await ping

        with patch.object(
            handlers,
            '_safe_websocket_send_text',
            new=AsyncMock(return_value=True),
        ) as send_text:
            self.assertFalse(
                await handlers._handle_metadata_receive_task(
                    websocket,
                    disconnected,
                    '[WebSocket-Metadata] client',
                ),
            )
            self.assertTrue(
                await handlers._handle_metadata_receive_task(
                    websocket,
                    invalid,
                    '[WebSocket-Metadata] client',
                ),
            )
            self.assertTrue(
                await handlers._handle_metadata_receive_task(
                    websocket,
                    ping,
                    '[WebSocket-Metadata] client',
                ),
            )

        send_text.assert_awaited_once()

    async def test_send_frame_and_connection_check_return_transport_result(
        self,
    ) -> None:
        """Frame sending and timeout checks return their Boolean outcomes."""
        websocket = MagicMock()

        with (
            patch.object(
                handlers,
                '_safe_websocket_send_json',
                new=AsyncMock(side_effect=[False, True]),
            ),
            patch.object(
                handlers,
                'check_and_maybe_close_on_timeout',
                new=AsyncMock(side_effect=[True, False]),
            ),
            patch.object(
                handlers,
                '_is_websocket_connected',
                return_value=True,
            ),
        ):
            failed = await handlers._send_metadata_websocket_frame(
                websocket,
                _frame(),
                'metadata-key',
                '17-0',
                '[WebSocket-Metadata] client',
            )
            sent = await handlers._send_metadata_websocket_frame(
                websocket,
                _frame(),
                'metadata-key',
                '17-0',
                '[WebSocket-Metadata] client',
            )
            timed_out = await handlers._metadata_websocket_is_active(
                websocket,
                1.0,
                'client',
            )
            active = await handlers._metadata_websocket_is_active(
                websocket,
                1.0,
                'client',
            )

        self.assertFalse(failed)
        self.assertTrue(sent)
        self.assertFalse(timed_out)
        self.assertTrue(active)


class TestMetadataTransportFlows(unittest.IsolatedAsyncioTestCase):
    """Verify SSE and WebSocket transports across their lifecycle branches."""

    async def test_sse_generator_handles_timeout_error_and_frame(self) -> None:
        """SSE transport sends keepalives, errors, and frames before closing.

        It always closes the fan-out subscription afterwards.
        """
        subscription = MagicMock()
        subscription.close = AsyncMock()
        request = MagicMock()
        request.is_disconnected = AsyncMock(
            side_effect=[False, False, False, True],
        )
        event_loop = MagicMock()
        event_loop.time = MagicMock(side_effect=[0.0, 20.0, 21.0, 22.0])

        with (
            patch.object(
                handlers.metadata_fanout,
                'subscribe',
                new=AsyncMock(return_value=subscription),
            ),
            patch.object(
                handlers.asyncio, 'get_running_loop',
                return_value=event_loop,
            ),
            patch.object(
                handlers.asyncio,
                'wait_for',
                new=AsyncMock(
                    side_effect=[
                        asyncio.TimeoutError(),
                        RuntimeError('Redis offline'),
                        _frame(),
                    ],
                ),
            ),
            patch.object(handlers.asyncio, 'sleep', new=AsyncMock()),
        ):
            iterator = handlers.metadata_stream_generator(
                request,
                MagicMock(),
                'metadata-key',
            )
            connected = await anext(iterator)
            heartbeat = await anext(iterator)
            error = await anext(iterator)
            frame = await anext(iterator)
            with self.assertRaises(StopAsyncIteration):
                await anext(iterator)

        self.assertEqual(connected, b'retry: 15000\n: connected\n\n')
        self.assertEqual(heartbeat, b': keepalive\n\n')
        self.assertIn(b'redis_error', error)
        self.assertIn(b'has_warning', frame)
        subscription.close.assert_awaited_once()

    async def test_sse_overlay_events_and_missing_ready_state_are_handled(
        self,
    ) -> None:
        """SSE immediately sends one overlay-ready event when available.

        Missing demand and ready state remain no-ops for normal video streams.
        """
        rds = MagicMock()
        rds.exists = AsyncMock(return_value=0)
        self.assertIsNone(
            await handlers._overlay_ready_event(rds, 'overlay:ready', {}),
        )
        await handlers._refresh_overlay_demand(rds, None, None)

        subscription = MagicMock()
        subscription.close = AsyncMock()
        request = MagicMock()
        request.is_disconnected = AsyncMock(side_effect=[False, True])
        event_loop = MagicMock()
        event_loop.time = MagicMock(side_effect=[0.0, 1.0])
        with (
            patch.object(
                handlers.metadata_fanout,
                'subscribe',
                new=AsyncMock(return_value=subscription),
            ),
            patch.object(
                handlers.asyncio, 'get_running_loop',
                return_value=event_loop,
            ),
            patch.object(
                handlers,
                '_next_overlay_ready_event',
                new=AsyncMock(
                    return_value=(
                        True, b'event: overlay_ready\n\n',
                    ),
                ),
            ),
        ):
            iterator = handlers.metadata_stream_generator(
                request,
                rds,
                'metadata-key',
                overlay_ready_key='overlay:ready',
                overlay_ready_payload={'stream_id': '12'},
            )
            await anext(iterator)
            overlay_event = await anext(iterator)
            with self.assertRaises(StopAsyncIteration):
                await anext(iterator)

        self.assertEqual(overlay_event, b'event: overlay_ready\n\n')
        subscription.close.assert_awaited_once()

    async def test_sse_turns_returned_reader_exceptions_into_error_events(
        self,
    ) -> None:
        """A returned reader exception follows the standard SSE error path.

        The client receives the same limited error event as a thrown error.
        """
        subscription = MagicMock()
        subscription.close = AsyncMock()
        request = MagicMock()
        request.is_disconnected = AsyncMock(side_effect=[False, True])
        event_loop = MagicMock()
        event_loop.time = MagicMock(side_effect=[0.0, 1.0])

        async def return_reader_error(
            awaitable: Coroutine[object, object, object],
            timeout: float,
        ) -> FrameOutData | Exception:
            """Close the unused subscription awaitable and return an error.

            Args:
                awaitable: Subscription coroutine passed to ``wait_for``.
                timeout: Requested metadata read timeout.

            Returns:
                Reader error passed through the fan-out subscription.
            """
            self.assertEqual(timeout, handlers._metadata_client_tick_seconds)
            awaitable.close()
            return RuntimeError('redis offline')

        with (
            patch.object(
                handlers.metadata_fanout,
                'subscribe',
                new=AsyncMock(return_value=subscription),
            ),
            patch.object(
                handlers.asyncio, 'get_running_loop',
                return_value=event_loop,
            ),
            patch.object(
                handlers.asyncio, 'wait_for',
                new=return_reader_error,
            ),
            patch.object(handlers.asyncio, 'sleep', new=AsyncMock()),
        ):
            iterator = handlers.metadata_stream_generator(
                request,
                MagicMock(),
                'metadata-key',
            )
            await anext(iterator)
            error_event = await anext(iterator)
            with self.assertRaises(StopAsyncIteration):
                await anext(iterator)

        self.assertIn(b'redis_error', error_event)

    async def test_websocket_push_ignores_reader_errors_and_cancels_receive(
        self,
    ) -> None:
        """WebSocket delivery ignores reader errors and cleans up tasks.

        The pending receive task is cancelled during shutdown.
        """
        subscription = MagicMock()
        subscription.close = AsyncMock()
        frames: list[FrameOutData | Exception] = [
            RuntimeError('offline'),
            _frame(),
        ]

        async def get_frame() -> FrameOutData | Exception:
            """Return the next value published by the fan-out subscription.

            Returns:
                A metadata frame or reader exception for the push loop.
            """
            return frames.pop(0)

        subscription.get = get_frame
        receive_gate = asyncio.Event()

        async def wait_for_message(*_args: object) -> str | None:
            """Remain pending until the transport closes its receive task.

            Returns:
                No value because the task is cancelled during clean-up.
            """
            await receive_gate.wait()
            return None

        with (
            patch.object(
                handlers.metadata_fanout,
                'subscribe',
                new=AsyncMock(return_value=subscription),
            ),
            patch.object(
                handlers,
                '_metadata_websocket_is_active',
                new=AsyncMock(side_effect=[True, True, False]),
            ),
            patch.object(
                handlers, '_safe_websocket_receive_text',
                wait_for_message,
            ),
            patch.object(
                handlers,
                '_send_metadata_websocket_frame',
                new=AsyncMock(return_value=True),
            ) as send_frame,
        ):
            count = await handlers.metadata_push_loop(
                MagicMock(),
                MagicMock(),
                'metadata-key',
                '127.0.0.1',
                'alice',
            )

        self.assertEqual(count, 1)
        send_frame.assert_awaited_once()
        subscription.close.assert_awaited_once()

    async def test_websocket_push_handles_completed_messages_and_timeouts(
        self,
    ) -> None:
        """WebSocket metadata accepts client pings and ignores idle reads.

        A failed safe send ends the loop without retaining the subscription.
        """
        subscription = MagicMock()
        subscription.close = AsyncMock()
        subscription.get = AsyncMock()
        outcomes: list[FrameOutData | Exception] = [
            asyncio.TimeoutError(),
            _frame(),
        ]

        async def wait_for_result(
            awaitable: Coroutine[object, object, object],
            timeout: float,
        ) -> FrameOutData:
            """Close the mocked awaitable and return the queued result.

            Args:
                awaitable: Subscription coroutine supplied to ``wait_for``.
                timeout: Unused timeout selected by the production loop.

            Returns:
                Next frame from the deterministic queue.

            Raises:
                Exception: Queued timeout used to exercise the idle path.
            """
            self.assertEqual(timeout, handlers._metadata_client_tick_seconds)
            awaitable.close()
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        async def active_after_task_runs(*_args: object) -> bool:
            """Yield once so each receive task completes before inspection.

            Returns:
                Whether the loop should process another transport iteration.
            """
            await asyncio.sleep(0)
            return len(outcomes) >= 0
        with (
            patch.object(
                handlers.metadata_fanout,
                'subscribe',
                new=AsyncMock(return_value=subscription),
            ),
            patch.object(
                handlers,
                '_metadata_websocket_is_active',
                new=active_after_task_runs,
            ),
            patch.object(
                handlers,
                '_handle_metadata_receive_task',
                new=AsyncMock(return_value=True),
            ),
            patch.object(
                handlers.asyncio,
                'wait_for',
                new=wait_for_result,
            ),
            patch.object(
                handlers,
                '_send_metadata_websocket_frame',
                new=AsyncMock(return_value=False),
            ),
        ):
            count = await handlers.metadata_push_loop(
                MagicMock(),
                MagicMock(),
                'metadata-key',
                '127.0.0.1',
                'alice',
            )

        self.assertEqual(count, 0)
        subscription.close.assert_awaited_once()

    async def test_websocket_push_stops_when_client_disconnects(
        self,
    ) -> None:
        """A completed receive task with no message ends the push loop."""
        subscription = MagicMock()
        subscription.close = AsyncMock()

        async def active_after_receive(*_args: object) -> bool:
            """Yield once so the initial receive task completes.

            Returns:
                Whether the connection is initially active.
            """
            await asyncio.sleep(0)
            return True

        with (
            patch.object(
                handlers.metadata_fanout,
                'subscribe',
                new=AsyncMock(return_value=subscription),
            ),
            patch.object(
                handlers,
                '_metadata_websocket_is_active',
                new=active_after_receive,
            ),
            patch.object(
                handlers,
                '_safe_websocket_receive_text',
                new=AsyncMock(return_value=None),
            ),
        ):
            count = await handlers.metadata_push_loop(
                MagicMock(),
                MagicMock(),
                'metadata-key',
                '127.0.0.1',
                'alice',
            )

        self.assertEqual(count, 0)
        subscription.close.assert_awaited_once()

    async def test_metadata_websocket_authorisation_and_stream_id_routes(
        self,
    ) -> None:
        """WebSocket handlers reject invalid access and resolve canonical keys.

        The encoded stream route delegates to the shared handler.
        """
        settings = cast(Settings, MagicMock())

        anonymous_socket = MagicMock()
        anonymous_socket.client = SimpleNamespace(host='127.0.0.1')
        anonymous_socket.accept = AsyncMock()
        with patch.object(
            handlers,
            'authenticate_ws_or_none',
            new=AsyncMock(return_value=(None, None)),
        ):
            await handlers.handle_metadata_ws(
                anonymous_socket,
                'Site A',
                'Camera A',
                MagicMock(),
                settings,
            )
        anonymous_socket.accept.assert_awaited_once()

        missing_user_socket = MagicMock()
        missing_user_socket.client = SimpleNamespace(host='127.0.0.1')
        missing_user_socket.accept = AsyncMock()
        missing_user_socket.close = AsyncMock()
        missing_db = MagicMock()
        missing_db.close = AsyncMock()
        with (
            patch.object(
                handlers,
                'authenticate_ws_or_none',
                new=AsyncMock(return_value=('alice', None)),
            ),
            patch.object(
                handlers,
                'load_user_access_context',
                new=AsyncMock(side_effect=RuntimeError('missing user')),
            ),
        ):
            await handlers.handle_metadata_ws(
                missing_user_socket,
                'Site A',
                'Camera A',
                MagicMock(),
                settings,
                cast(handlers.AsyncSession, missing_db),
            )
        missing_user_socket.close.assert_awaited_once_with(
            code=4001,
            reason='User not found',
        )
        missing_db.close.assert_awaited_once()

        denied_socket = MagicMock()
        denied_socket.client = SimpleNamespace(host='127.0.0.1')
        denied_socket.accept = AsyncMock()
        denied_socket.close = AsyncMock()
        denied_db = MagicMock()
        denied_db.close = AsyncMock()
        with (
            patch.object(
                handlers,
                'authenticate_ws_or_none',
                new=AsyncMock(return_value=('alice', None)),
            ),
            patch.object(
                handlers,
                'load_user_access_context',
                new=AsyncMock(return_value=(None, [], 'viewer')),
            ),
        ):
            await handlers.handle_metadata_ws(
                denied_socket,
                'Site A',
                'Camera A',
                MagicMock(),
                settings,
                cast(handlers.AsyncSession, denied_db),
            )
        denied_socket.close.assert_awaited_once_with(
            code=4003,
            reason='Access denied',
        )

        active_socket = MagicMock()
        active_socket.client = SimpleNamespace(host='127.0.0.1')
        active_socket.accept = AsyncMock()
        with (
            patch.object(
                handlers,
                'authenticate_ws_or_none',
                new=AsyncMock(return_value=('alice', None)),
            ),
            patch.object(
                handlers,
                'get_metadata_site_generation',
                new=AsyncMock(return_value=9),
            ),
            patch.object(
                handlers, 'build_metadata_key',
                return_value='metadata-key',
            ),
            patch.object(
                handlers,
                'metadata_push_loop',
                new=AsyncMock(side_effect=WebSocketDisconnect()),
            ),
        ):
            await handlers.handle_metadata_ws(
                active_socket,
                'Site A',
                'Camera A',
                MagicMock(),
                settings,
            )

        override_socket = MagicMock()
        override_socket.client = SimpleNamespace(host='127.0.0.1')
        override_socket.accept = AsyncMock()
        with (
            patch.object(
                handlers,
                'authenticate_ws_or_none',
                new=AsyncMock(return_value=('alice', None)),
            ),
            patch.object(
                handlers,
                'metadata_push_loop',
                new=AsyncMock(return_value=1),
            ) as push_loop,
        ):
            await handlers.handle_metadata_ws(
                override_socket,
                'Site A',
                'Camera A',
                MagicMock(),
                settings,
                redis_key_override='explicit-key',
            )
        push_loop_args = push_loop.await_args
        assert push_loop_args is not None
        self.assertEqual(push_loop_args.args[2], 'explicit-key')

        with (
            patch.object(
                handlers,
                'get_metadata_site_generation',
                new=AsyncMock(return_value=9),
            ),
            patch.object(
                handlers,
                'build_metadata_key_from_stream_id',
                return_value='metadata:12',
            ),
            patch.object(
                handlers,
                'handle_metadata_ws',
                new=AsyncMock(),
            ) as handle_ws,
        ):
            await handlers.handle_metadata_stream_id_ws(
                MagicMock(),
                'Site A',
                '12',
                MagicMock(),
                settings,
            )

        handle_ws.assert_awaited_once()
