from __future__ import annotations

import asyncio
import json
import unittest
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import WebSocketDisconnect

from examples.streaming_web.schemas import FrameOutData
from examples.streaming_web.ws_handlers import _build_metadata_payload
from examples.streaming_web.ws_handlers import _encode_sse_event
from examples.streaming_web.ws_handlers import (
    handle_metadata_stream_id_ws,
)
from examples.streaming_web.ws_handlers import (
    handle_metadata_ws,
)
from examples.streaming_web.ws_handlers import metadata_push_loop
from examples.streaming_web.ws_handlers import (
    metadata_stream_generator,
)


class WsHandlersTest(unittest.IsolatedAsyncioTestCase):
    """Tests for metadata-only live stream handlers."""

    def make_ws(self, host: str = 'virtual') -> SimpleNamespace:
        """Support make_ws."""
        ws = SimpleNamespace()
        ws.client = SimpleNamespace(host=host)
        ws.accept = AsyncMock()
        ws.close = AsyncMock()
        return ws

    async def _first_event(self, iterator: AsyncIterator[bytes]) -> bytes:
        """Support _first_event."""
        return await anext(iterator)

    def test_build_metadata_payload_uses_compact_warning_state(self) -> None:
        """Exercise this test."""
        payload = _build_metadata_payload(
            cast(FrameOutData, {'has_warning': 'true'}),
        )

        self.assertEqual(payload, {'has_warning': True})

    def test_build_metadata_payload_defaults_to_no_warning(self) -> None:
        """Exercise this test."""
        payload = _build_metadata_payload(cast(FrameOutData, {}))

        self.assertEqual(payload, {'has_warning': False})
        payload = _build_metadata_payload(
            cast(FrameOutData, {'has_warning': True}),
        )
        self.assertEqual(payload, {'has_warning': True})

    def test_encode_sse_event_includes_event_and_payload(self) -> None:
        """Exercise this test."""
        encoded = _encode_sse_event({'id': '1-0', 'has_warning': True})

        self.assertIn(b'id: 1-0', encoded)
        self.assertIn(b'event: metadata', encoded)
        self.assertIn(b'"has_warning":true', encoded)

    def test_encode_sse_event_without_id(self) -> None:
        """Exercise this test."""
        encoded = _encode_sse_event({'has_warning': False})

        self.assertNotIn(b'id:', encoded)
        self.assertIn(b'"has_warning":false', encoded)

    def test_encode_sse_event_supports_named_events(self) -> None:
        """Exercise this test."""
        encoded = _encode_sse_event(
            {'state': 'ready'},
            event_type='overlay_ready',
        )

        self.assertIn(b'event: overlay_ready', encoded)
        self.assertIn(b'"state":"ready"', encoded)

    async def test_metadata_stream_yields_initial_retry(self) -> None:
        """Exercise this test."""
        request = MagicMock()
        request.is_disconnected = AsyncMock(return_value=True)

        event = await self._first_event(
            metadata_stream_generator(request, MagicMock(), 'metadata-key'),
        )

        self.assertEqual(event, b'retry: 15000\n: connected\n\n')

    async def test_metadata_stream_yields_metadata_event(self) -> None:
        """Exercise this test."""
        request = MagicMock()
        request.is_disconnected = AsyncMock(side_effect=[False, True])
        rds = MagicMock()

        with patch(
            'examples.streaming_web.ws_handlers.'
            'fetch_latest_metadata_for_key',
            new=AsyncMock(return_value={'id': '2-0', 'has_warning': 'true'}),
        ):
            iterator = metadata_stream_generator(request, rds, 'metadata-key')
            await anext(iterator)
            event = await anext(iterator)

        self.assertIn(b'id: 2-0', event)
        self.assertIn(b'"has_warning":true', event)

    async def test_metadata_stream_yields_overlay_ready_event(self) -> None:
        """Exercise this test."""
        request = MagicMock()
        request.is_disconnected = AsyncMock(side_effect=[False, True])
        rds = MagicMock()
        rds.exists = AsyncMock(return_value=1)

        iterator = metadata_stream_generator(
            request,
            rds,
            'metadata-key',
            overlay_ready_key='media_overlay_ready:overlay-path',
            overlay_ready_payload={
                'state': 'ready',
                'playback_url': '/hazard/media/overlay/index.m3u8',
            },
        )
        await anext(iterator)
        event = await anext(iterator)

        self.assertIn(b'event: overlay_ready', event)
        self.assertIn(b'"state":"ready"', event)
        self.assertIn(b'/hazard/media/overlay/index.m3u8', event)

    async def test_metadata_stream_refreshes_overlay_demand(self) -> None:
        """Exercise this test."""
        request = MagicMock()
        request.is_disconnected = AsyncMock(side_effect=[False, True])
        rds = MagicMock()
        rds.set = AsyncMock()

        with (
            patch(
                'examples.streaming_web.ws_handlers.'
                'fetch_latest_metadata_for_key',
                new=AsyncMock(return_value=None),
            ),
            patch(
                'examples.streaming_web.ws_handlers.asyncio.sleep',
                new=AsyncMock(),
            ),
        ):
            iterator = metadata_stream_generator(
                request,
                rds,
                'metadata-key',
                overlay_demand_key='media_overlay_demand:base:emgtVFc',
                overlay_demand_ttl_seconds=90,
                overlay_demand_refresh_seconds=1.0,
            )
            await anext(iterator)
            with self.assertRaises(StopAsyncIteration):
                await anext(iterator)

        rds.set.assert_awaited_once_with(
            'media_overlay_demand:base:emgtVFc',
            b'1',
            ex=90,
        )

    async def test_metadata_stream_handles_timeout_then_disconnect(
        self,
    ) -> None:
        """Exercise this test."""
        request = MagicMock()
        request.is_disconnected = AsyncMock(side_effect=[False, True])

        with (
            patch(
                'examples.streaming_web.ws_handlers.'
                'fetch_latest_metadata_for_key',
                new=AsyncMock(side_effect=asyncio.TimeoutError),
            ),
            patch(
                'examples.streaming_web.ws_handlers.asyncio.sleep',
                new=AsyncMock(),
            ) as sleep,
        ):
            iterator = metadata_stream_generator(
                request,
                MagicMock(),
                'metadata-key',
            )
            await anext(iterator)
            with self.assertRaises(StopAsyncIteration):
                await anext(iterator)

        sleep.assert_awaited_once()

    async def test_metadata_stream_handles_read_error(self) -> None:
        """Exercise this test."""
        request = MagicMock()
        request.is_disconnected = AsyncMock(side_effect=[False, True])

        with (
            patch(
                'examples.streaming_web.ws_handlers.'
                'fetch_latest_metadata_for_key',
                new=AsyncMock(side_effect=RuntimeError('redis down')),
            ),
            patch(
                'examples.streaming_web.ws_handlers.asyncio.sleep',
                new=AsyncMock(),
            ) as sleep,
        ):
            iterator = metadata_stream_generator(
                request,
                MagicMock(),
                'metadata-key',
            )
            await anext(iterator)
            event = await anext(iterator)
            with self.assertRaises(StopAsyncIteration):
                await anext(iterator)

        self.assertIn(b'event: redis_error', event)
        self.assertIn(b'"source":"redis"', event)
        self.assertIn(b'"code":"redis_unavailable"', event)
        sleep.assert_awaited_once_with(1.0)

    async def test_metadata_stream_yields_keepalive(self) -> None:
        """Exercise this test."""
        request = MagicMock()
        request.is_disconnected = AsyncMock(side_effect=[False, True])
        loop = MagicMock()
        loop.time.side_effect = [0.0, 16.0]

        with (
            patch(
                'examples.streaming_web.ws_handlers.'
                'fetch_latest_metadata_for_key',
                new=AsyncMock(return_value=None),
            ),
            patch(
                'examples.streaming_web.ws_handlers.asyncio.'
                'get_running_loop',
                return_value=loop,
            ),
        ):
            iterator = metadata_stream_generator(
                request,
                MagicMock(),
                'metadata-key',
            )
            await anext(iterator)
            event = await anext(iterator)

        self.assertEqual(event, b': keepalive\n\n')

    async def test_metadata_stream_sleeps_without_keepalive(self) -> None:
        """Exercise this test."""
        request = MagicMock()
        request.is_disconnected = AsyncMock(side_effect=[False, True])
        loop = MagicMock()
        loop.time.side_effect = [0.0, 1.0]

        with (
            patch(
                'examples.streaming_web.ws_handlers.'
                'fetch_latest_metadata_for_key',
                new=AsyncMock(return_value=None),
            ),
            patch(
                'examples.streaming_web.ws_handlers.asyncio.'
                'get_running_loop',
                return_value=loop,
            ),
            patch(
                'examples.streaming_web.ws_handlers.asyncio.sleep',
                new=AsyncMock(),
            ) as sleep,
        ):
            iterator = metadata_stream_generator(
                request,
                MagicMock(),
                'metadata-key',
            )
            await anext(iterator)
            with self.assertRaises(StopAsyncIteration):
                await anext(iterator)

        sleep.assert_awaited_once()

    async def test_metadata_push_loop_responds_to_ping(self) -> None:
        """Exercise this test."""
        ws = self.make_ws()
        rds = MagicMock()

        with patch(
            'examples.streaming_web.ws_handlers.'
            '_safe_websocket_receive_text',
            new=AsyncMock(side_effect=['{"action":"ping"}', None]),
        ), patch(
            'examples.streaming_web.ws_handlers.'
            '_safe_websocket_send_text',
            new=AsyncMock(return_value=True),
        ) as send_text, patch(
            'examples.streaming_web.ws_handlers.'
            '_is_websocket_connected',
            return_value=True,
        ), patch(
            'examples.streaming_web.ws_handlers.'
            'check_and_maybe_close_on_timeout',
            new=AsyncMock(return_value=False),
        ), patch(
            'examples.streaming_web.ws_handlers.'
            'fetch_latest_metadata_for_key',
            new=AsyncMock(return_value=None),
        ):
            count = await metadata_push_loop(
                cast(Any, ws), rds, 'metadata-key', '127.0.0.1', 'alice',
            )

        self.assertEqual(count, 0)
        await_args = send_text.await_args
        self.assertIsNotNone(await_args)
        assert await_args is not None
        sent_payload = json.loads(await_args.args[1])
        self.assertEqual(sent_payload, {'action': 'pong'})

    async def test_metadata_push_loop_stops_on_session_timeout(self) -> None:
        """Exercise this test."""
        ws = self.make_ws()

        async def wait_forever(*_args: object) -> None:
            """Support wait_forever."""
            await asyncio.Event().wait()

        with (
            patch(
                'examples.streaming_web.ws_handlers.'
                '_safe_websocket_receive_text',
                side_effect=wait_forever,
            ),
            patch(
                'examples.streaming_web.ws_handlers.'
                'check_and_maybe_close_on_timeout',
                new=AsyncMock(return_value=True),
            ),
        ):
            count = await metadata_push_loop(
                cast(Any, ws),
                MagicMock(),
                'metadata-key',
                '127.0.0.1',
                'alice',
            )

        self.assertEqual(count, 0)

    async def test_metadata_push_loop_stops_when_socket_disconnected(
        self,
    ) -> None:
        """Exercise this test."""
        with (
            patch(
                'examples.streaming_web.ws_handlers.'
                '_safe_websocket_receive_text',
                new=AsyncMock(return_value=None),
            ),
            patch(
                'examples.streaming_web.ws_handlers.'
                'check_and_maybe_close_on_timeout',
                new=AsyncMock(return_value=False),
            ),
            patch(
                'examples.streaming_web.ws_handlers.'
                '_is_websocket_connected',
                return_value=False,
            ),
        ):
            count = await metadata_push_loop(
                cast(Any, self.make_ws()),
                MagicMock(),
                'metadata-key',
                '127.0.0.1',
                'alice',
            )

        self.assertEqual(count, 0)

    async def test_metadata_push_loop_handles_bad_json_timeout_and_send_fail(
        self,
    ) -> None:
        """Exercise this test."""
        ws = self.make_ws()
        fetch = AsyncMock(
            side_effect=[
                asyncio.TimeoutError,
                {'id': '1-0', 'has_warning': 'yes'},
            ],
        )

        with (
            patch(
                'examples.streaming_web.ws_handlers.'
                '_safe_websocket_receive_text',
                new=AsyncMock(side_effect=['not-json', None]),
            ),
            patch(
                'examples.streaming_web.ws_handlers.'
                '_safe_websocket_send_json',
                new=AsyncMock(return_value=False),
            ) as send_json,
            patch(
                'examples.streaming_web.ws_handlers.'
                '_is_websocket_connected',
                return_value=True,
            ),
            patch(
                'examples.streaming_web.ws_handlers.'
                'check_and_maybe_close_on_timeout',
                new=AsyncMock(return_value=False),
            ),
            patch(
                'examples.streaming_web.ws_handlers.'
                'fetch_latest_metadata_for_key',
                new=fetch,
            ),
            patch(
                'examples.streaming_web.ws_handlers.asyncio.sleep',
                new=AsyncMock(),
            ),
        ):
            count = await metadata_push_loop(
                cast(Any, ws),
                MagicMock(),
                'metadata-key',
                '127.0.0.1',
                'alice',
            )

        self.assertEqual(count, 0)
        send_json.assert_awaited_once()

    async def test_metadata_push_loop_ignores_bad_json_ping_message(
        self,
    ) -> None:
        """Exercise this test."""
        ws = self.make_ws()

        with (
            patch(
                'examples.streaming_web.ws_handlers.'
                '_safe_websocket_receive_text',
                new=AsyncMock(side_effect=['not-json', None]),
            ),
            patch(
                'examples.streaming_web.ws_handlers.'
                '_is_websocket_connected',
                return_value=True,
            ),
            patch(
                'examples.streaming_web.ws_handlers.'
                'check_and_maybe_close_on_timeout',
                new=AsyncMock(side_effect=[False, True]),
            ),
            patch(
                'examples.streaming_web.ws_handlers.'
                'fetch_latest_metadata_for_key',
                new=AsyncMock(return_value=None),
            ),
        ):
            count = await metadata_push_loop(
                cast(Any, ws),
                MagicMock(),
                'metadata-key',
                '127.0.0.1',
                'alice',
            )

        self.assertEqual(count, 0)

    async def test_metadata_push_loop_parses_completed_bad_json_task(
        self,
    ) -> None:
        """Exercise this test."""
        ws = self.make_ws()

        class DoneReceiveTask:
            """Tests for DoneReceiveTask."""

            def done(self) -> bool:
                """Support done."""
                return True

            def result(self) -> str:
                """Support result."""
                return 'not-json'

        def close_coroutine_task(coro: object) -> DoneReceiveTask:
            """Support close_coroutine_task."""
            close = getattr(coro, 'close', None)
            if close is not None:
                close()
            return DoneReceiveTask()

        with (
            patch(
                'examples.streaming_web.ws_handlers.asyncio.'
                'create_task',
                side_effect=close_coroutine_task,
            ),
            patch(
                'examples.streaming_web.ws_handlers.'
                '_is_websocket_connected',
                return_value=True,
            ),
            patch(
                'examples.streaming_web.ws_handlers.'
                'check_and_maybe_close_on_timeout',
                new=AsyncMock(side_effect=[False, True]),
            ),
            patch(
                'examples.streaming_web.ws_handlers.'
                'fetch_latest_metadata_for_key',
                new=AsyncMock(return_value=None),
            ),
            patch(
                'examples.streaming_web.ws_handlers.asyncio.sleep',
                new=AsyncMock(),
            ),
        ):
            count = await metadata_push_loop(
                cast(Any, ws),
                MagicMock(),
                'metadata-key',
                '127.0.0.1',
                'alice',
            )

        self.assertEqual(count, 0)

    async def test_metadata_push_loop_sends_metadata_update(self) -> None:
        """Exercise this test."""
        ws = self.make_ws()

        async def wait_forever(*_args: object) -> None:
            """Support wait_forever."""
            await asyncio.Event().wait()

        with (
            patch(
                'examples.streaming_web.ws_handlers.'
                '_safe_websocket_receive_text',
                side_effect=wait_forever,
            ),
            patch(
                'examples.streaming_web.ws_handlers.'
                '_safe_websocket_send_json',
                new=AsyncMock(return_value=True),
            ) as send_json,
            patch(
                'examples.streaming_web.ws_handlers.'
                '_is_websocket_connected',
                return_value=True,
            ),
            patch(
                'examples.streaming_web.ws_handlers.'
                'check_and_maybe_close_on_timeout',
                new=AsyncMock(side_effect=[False, True]),
            ),
            patch(
                'examples.streaming_web.ws_handlers.'
                'fetch_latest_metadata_for_key',
                new=AsyncMock(
                    return_value={
                        'id': '1-0',
                        'has_warning': 'on',
                    },
                ),
            ),
        ):
            count = await metadata_push_loop(
                cast(Any, ws),
                MagicMock(),
                'metadata-key',
                '127.0.0.1',
                'alice',
            )

        self.assertEqual(count, 1)
        send_json.assert_awaited_once()

    async def test_handle_metadata_ws_closes_when_user_lookup_fails(
        self,
    ) -> None:
        """Exercise this test."""
        ws = self.make_ws()

        with (
            patch(
                'examples.streaming_web.ws_handlers.'
                'authenticate_ws_or_none',
                new=AsyncMock(return_value=('alice', None)),
            ),
            patch(
                'examples.streaming_web.ws_handlers.'
                'get_user_and_sites',
                new=AsyncMock(side_effect=RuntimeError('missing')),
            ),
        ):
            await handle_metadata_ws(
                websocket=cast(Any, ws),
                label='site-a',
                key='cam-a',
                rds=MagicMock(),
                settings=MagicMock(),
                db=MagicMock(),
            )

        ws.close.assert_awaited_once_with(code=4001, reason='User not found')

    async def test_handle_metadata_ws_closes_on_site_denied(self) -> None:
        """Exercise this test."""
        ws = self.make_ws()

        with (
            patch(
                'examples.streaming_web.ws_handlers.'
                'authenticate_ws_or_none',
                new=AsyncMock(return_value=('alice', None)),
            ),
            patch(
                'examples.streaming_web.ws_handlers.'
                'get_user_and_sites',
                new=AsyncMock(return_value=(None, ['other'], 'user')),
            ),
        ):
            await handle_metadata_ws(
                websocket=cast(Any, ws),
                label='site-a',
                key='cam-a',
                rds=MagicMock(),
                settings=MagicMock(),
                db=MagicMock(),
            )

        ws.close.assert_awaited_once_with(code=4003, reason='Access denied')

    async def test_handle_metadata_ws_logs_disconnect(self) -> None:
        """Exercise this test."""
        ws = self.make_ws()

        with (
            patch(
                'examples.streaming_web.ws_handlers.'
                'authenticate_ws_or_none',
                new=AsyncMock(return_value=('alice', None)),
            ),
            patch(
                'examples.streaming_web.ws_handlers.'
                'metadata_push_loop',
                new=AsyncMock(side_effect=RuntimeError('boom')),
            ),
        ):
            with self.assertRaises(RuntimeError):
                await handle_metadata_ws(
                    websocket=cast(Any, ws),
                    label='site-a',
                    key='cam-a',
                    rds=MagicMock(),
                    settings=MagicMock(),
                )

    async def test_handle_metadata_ws_handles_websocket_disconnect(
        self,
    ) -> None:
        """Exercise this test."""
        ws = self.make_ws()

        with (
            patch(
                'examples.streaming_web.ws_handlers.'
                'authenticate_ws_or_none',
                new=AsyncMock(return_value=('alice', None)),
            ),
            patch(
                'examples.streaming_web.ws_handlers.'
                'metadata_push_loop',
                new=AsyncMock(side_effect=WebSocketDisconnect),
            ),
        ):
            await handle_metadata_ws(
                websocket=cast(Any, ws),
                label='site-a',
                key='cam-a',
                rds=MagicMock(),
                settings=MagicMock(),
            )

    async def test_handle_metadata_stream_id_ws_auth_fail(self) -> None:
        """Exercise this test."""
        ws = self.make_ws()

        with patch(
            'examples.streaming_web.ws_handlers.'
            'authenticate_ws_or_none',
            new=AsyncMock(return_value=(None, None)),
        ):
            await handle_metadata_stream_id_ws(
                websocket=cast(Any, ws),
                label='site-a',
                stream_id='cam-id',
                rds=MagicMock(),
                settings=MagicMock(),
            )

        ws.accept.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
