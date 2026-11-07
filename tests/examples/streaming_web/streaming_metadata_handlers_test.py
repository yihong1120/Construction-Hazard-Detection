from __future__ import annotations

import asyncio
import unittest
from collections.abc import AsyncGenerator
from functools import partial
from typing import Any
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from examples.streaming_web.metadata_fanout import _StreamSubscribers
from examples.streaming_web.metadata_fanout import MetadataFanout
from examples.streaming_web.schemas import FrameOutData
from examples.streaming_web.streaming_metadata_handlers import (
    metadata_push_loop,
)
from examples.streaming_web.streaming_metadata_handlers import (
    metadata_stream_generator,
)
from examples.streaming_web.streaming_metadata_handlers import (
    MetadataOverlayLease,
)
from examples.streaming_web.streaming_metadata_handlers import (
    MetadataSubscription,
)


async def _reader_removes_empty_state_when_blocked_fetcher(
    release: Any,
    started: Any,
    /,
    _redis: object,
    _key: str,
    _last_id: str,
) -> FrameOutData | None:
    """Wait until cancellation after signalling that the fetch began.

    Returns:
        No frame because the task is cancelled before Redis responds.
    """
    started.set()
    await release.wait()
    return None


class _OneReaderBroadcastsLatestFrameToAllCallbacks:
    calls: Any
    release: Any

    async def _call_fetcher(
        self,
        _rds: object,
        _key: str,
        _last_id: str,
    ) -> FrameOutData:

        self.calls += 1
        await self.release.wait()
        if self.calls == 1:
            return {
                'id': '1-0',
                'has_warning': True,
                'key': 'camera',
                'stream_id': 'camera',
                'redis_key': 'metadata-key',
            }
        await asyncio.Event().wait()
        return {
            'id': 'unreachable',
            'has_warning': False,
            'key': 'camera',
            'stream_id': 'camera',
            'redis_key': 'metadata-key',
        }


class _ReaderRetriesEmptyAndFailedRedisReadsCallbacks:
    sleep_calls: Any
    state: Any

    async def _call_remove_subscribers(self, _delay: float) -> None:
        """Clear the subscriber set after both retry paths execute."""

        self.sleep_calls += 1
        if self.sleep_calls == 2:
            self.state.queues.clear()


class MetadataFanoutTests(unittest.IsolatedAsyncioTestCase):
    """Provide MetadataFanoutTests."""

    async def test_one_reader_broadcasts_latest_frame_to_all_subscribers(
        self,
    ) -> None:
        """Test one reader broadcasts latest frame to all subscribers."""
        _callbacks = _OneReaderBroadcastsLatestFrameToAllCallbacks()
        fanout = MetadataFanout()
        _callbacks.release = asyncio.Event()
        _callbacks.calls = 0

        fetcher = _callbacks._call_fetcher

        first = await fanout.subscribe(
            MagicMock(),
            'metadata-key',
            fetcher=fetcher,
        )
        second = await fanout.subscribe(
            MagicMock(),
            'metadata-key',
            fetcher=fetcher,
        )
        _callbacks.release.set()
        first_item = await first.get()
        second_item = await second.get()
        assert not isinstance(first_item, Exception)
        assert not isinstance(second_item, Exception)
        self.assertEqual(first_item['id'], '1-0')
        self.assertEqual(second_item['id'], '1-0')
        self.assertGreaterEqual(_callbacks.calls, 1)
        await first.close()
        await second.close()

    async def test_subscription_close_and_publish_keep_only_latest_item(
        self,
    ) -> None:
        """Subscriptions close idempotently and queues discard stale data.

        Only the newest metadata item remains for a slow client.
        """
        fanout = MetadataFanout()
        subscription = await fanout.subscribe(
            MagicMock(),
            'metadata-key',
            fetcher=AsyncMock(return_value=None),
        )
        await subscription.close()
        await subscription.close()

        queue: asyncio.Queue[FrameOutData | Exception] = asyncio.Queue(
            maxsize=1,
        )
        queue.put_nowait(
            {
                'id': 'old',
                'has_warning': False,
                'key': 'camera',
                'stream_id': 'camera',
                'redis_key': 'metadata-key',
            },
        )
        state = _StreamSubscribers(
            rds=MagicMock(),
            queues={queue},
            fetcher=AsyncMock(return_value=None),
        )
        failure = RuntimeError('offline')
        await fanout._publish(state, failure)
        self.assertIs(await queue.get(), failure)

    async def test_reader_retries_empty_and_failed_redis_reads(self) -> None:
        """The reader yields on idle/error paths until subscribers disappear.

        Retry delays prevent an unavailable Redis server from spinning CPU.
        """
        _callbacks = _ReaderRetriesEmptyAndFailedRedisReadsCallbacks()
        fanout = MetadataFanout()
        queue: asyncio.Queue[FrameOutData | Exception] = asyncio.Queue()
        _callbacks.state = _StreamSubscribers(
            rds=MagicMock(),
            queues={queue},
            fetcher=AsyncMock(side_effect=[RuntimeError('offline'), None]),
        )
        fanout._streams['metadata-key'] = _callbacks.state
        _callbacks.sleep_calls = 0

        remove_subscribers = _callbacks._call_remove_subscribers

        with patch(
            'examples.streaming_web.metadata_fanout._sleep',
            new=remove_subscribers,
        ):
            await fanout._run('metadata-key', _callbacks.state)

        self.assertIsInstance(await queue.get(), RuntimeError)

    async def test_reader_removes_empty_state_when_cancelled_during_fetch(
        self,
    ) -> None:
        """Cancelling a reader removes its empty fan-out stream state.

        No unused Redis reader remains after the final subscriber exits.
        """
        fanout = MetadataFanout()
        queue: asyncio.Queue[FrameOutData | Exception] = asyncio.Queue()
        started = asyncio.Event()
        release = asyncio.Event()

        blocked_fetcher = partial(
            _reader_removes_empty_state_when_blocked_fetcher,
            release,
            started,
        )

        state = _StreamSubscribers(
            rds=MagicMock(),
            queues={queue},
            fetcher=blocked_fetcher,
        )
        fanout._streams['metadata-key'] = state
        reader = asyncio.create_task(fanout._run('metadata-key', state))
        await started.wait()
        state.queues.clear()
        reader.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await reader

        self.assertNotIn('metadata-key', fanout._streams)


class MetadataHandlerTests(unittest.IsolatedAsyncioTestCase):
    """Provide MetadataHandlerTests."""

    async def test_sse_reads_from_subscription_and_closes_it(self) -> None:
        """Test sse reads from subscription and closes it."""
        subscription = MagicMock()
        subscription.get = AsyncMock(
            return_value={'id': '1-0', 'has_warning': True},
        )
        subscription.close = AsyncMock()
        request = MagicMock()
        request.is_disconnected = AsyncMock(side_effect=[False, True])

        with patch(
            'examples.streaming_web.streaming_metadata_handlers.'
            'metadata_fanout.subscribe',
            new=AsyncMock(return_value=subscription),
        ):
            iterator = metadata_stream_generator(
                request,
                MetadataSubscription(
                    MagicMock(),
                    'metadata-key',
                    MetadataOverlayLease(),
                ),
            )
            self.assertEqual(
                await anext(iterator),
                b'retry: 15000\n: connected\n\n',
            )
            event = await anext(iterator)
            self.assertIn(b'event: metadata', event)
            await cast(AsyncGenerator[bytes], iterator).aclose()

        subscription.close.assert_awaited_once()

    async def test_websocket_push_sends_subscription_frame(self) -> None:
        """Test websocket push sends subscription frame."""
        subscription = MagicMock()
        subscription.get = AsyncMock(
            return_value={'id': '1-0', 'has_warning': True},
        )
        subscription.close = AsyncMock()
        websocket = MagicMock()
        websocket.client_state = MagicMock(name='CONNECTED')

        with (
            patch(
                'examples.streaming_web.streaming_metadata_handlers.'
                'metadata_fanout.subscribe',
                new=AsyncMock(return_value=subscription),
            ),
            patch(
                'examples.streaming_web.streaming_metadata_handlers.'
                '_metadata_websocket_is_active',
                new=AsyncMock(side_effect=[True, False]),
            ),
            patch(
                'examples.streaming_web.streaming_metadata_handlers.'
                '_safe_websocket_receive_text',
                new=AsyncMock(return_value=None),
            ),
            patch(
                'examples.streaming_web.streaming_metadata_handlers.'
                '_send_metadata_websocket_frame',
                new=AsyncMock(return_value=True),
            ) as send_frame,
        ):
            count = await metadata_push_loop(
                websocket,
                MagicMock(),
                'metadata-key',
                '127.0.0.1',
                'alice',
            )

        self.assertEqual(count, 1)
        send_frame.assert_awaited_once()
        subscription.close.assert_awaited_once()


class MetadataOverlayErrorTest(unittest.IsolatedAsyncioTestCase):
    async def test_quota_refusal_emits_overlay_error_and_closes_subscription(
        self,
    ):
        from fastapi import HTTPException
        from examples.streaming_web import (
            streaming_metadata_handlers as handlers,
        )

        request = MagicMock()
        request.is_disconnected = AsyncMock(return_value=False)
        subscription = MagicMock()
        subscription.close = AsyncMock()
        with (
            patch.object(
                handlers.metadata_fanout,
                'subscribe',
                new=AsyncMock(return_value=subscription),
            ),
            patch.object(
                handlers,
                '_refresh_overlay_demand_if_due',
                new=AsyncMock(
                    side_effect=HTTPException(
                        429,
                        'overlay_language_limit_reached',
                    ),
                ),
            ),
        ):
            stream = handlers.MetadataSubscription(
                MagicMock(),
                'metadata-key',
            ).events(request)
            self.assertIn(b'connected', await anext(stream))
            error = await anext(stream)
            self.assertIn(b'overlay_error', error)
            self.assertNotIn(b'redis_unavailable', error)
            with self.assertRaises(StopAsyncIteration):
                await anext(stream)
        subscription.close.assert_awaited_once()
