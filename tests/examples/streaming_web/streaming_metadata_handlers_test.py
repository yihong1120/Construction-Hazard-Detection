from __future__ import annotations

import asyncio
import unittest
from collections.abc import AsyncGenerator
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from examples.streaming_web.metadata_fanout import MetadataFanout
from examples.streaming_web.schemas import FrameOutData
from examples.streaming_web.streaming_metadata_handlers import (
    metadata_push_loop,
)
from examples.streaming_web.streaming_metadata_handlers import (
    metadata_stream_generator,
)


class MetadataFanoutTests(unittest.IsolatedAsyncioTestCase):
    """Provide MetadataFanoutTests."""

    async def test_one_reader_broadcasts_latest_frame_to_all_subscribers(
        self,
    ) -> None:
        """Test one reader broadcasts latest frame to all subscribers."""
        fanout = MetadataFanout()
        release = asyncio.Event()
        calls = 0

        async def fetcher(
            _rds: object,
            _key: str,
            _last_id: str,
        ) -> FrameOutData:
            """Perform fetcher.

            Args:
                _rds: Value used by this callable.
                _key: Value used by this callable.
                _last_id: Value used by this callable.

            Returns:
                The callable result.
            """
            nonlocal calls
            calls += 1
            await release.wait()
            if calls == 1:
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

        first = await fanout.subscribe(
            MagicMock(), 'metadata-key', fetcher=fetcher,
        )
        second = await fanout.subscribe(
            MagicMock(), 'metadata-key', fetcher=fetcher,
        )
        release.set()
        first_item = await first.get()
        second_item = await second.get()
        assert not isinstance(first_item, Exception)
        assert not isinstance(second_item, Exception)
        self.assertEqual(first_item['id'], '1-0')
        self.assertEqual(second_item['id'], '1-0')
        self.assertGreaterEqual(calls, 1)
        await first.close()
        await second.close()


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
                MagicMock(),
                'metadata-key',
            )
            self.assertEqual(
                await anext(iterator), b'retry: 15000\n: connected\n\n',
            )
            event = await anext(iterator)
            self.assertIn(b'event: metadata', event)
            await cast(AsyncGenerator[bytes, None], iterator).aclose()

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
