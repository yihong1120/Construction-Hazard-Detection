from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import patch

from examples.streaming_web import redis_service as service
from examples.streaming_web.redis_service import build_metadata_key
from examples.streaming_web.redis_service import (
    fetch_latest_metadata_for_key,
)
from examples.streaming_web.redis_service import (
    get_metadata_keys_for_label,
)


class TestRedisService(unittest.IsolatedAsyncioTestCase):
    """Tests for compact live metadata Redis helpers."""

    def setUp(self) -> None:
        """Prepare test fixtures."""
        self.mock_rds = AsyncMock()

    def test_build_metadata_key_uses_encoded_site_and_stream(self) -> None:
        """Exercise this test."""
        with patch(
            'examples.streaming_web.redis_service.Utils.encode',
            side_effect=lambda value: f'encoded({value})',
        ):
            key = build_metadata_key('site-a', 'cam-1')

        self.assertEqual(key, 'stream_metadata:encoded(site-a)|encoded(cam-1)')

    def test_metadata_key_helpers_handle_cache_and_invalid_values(self) -> None:
        """Metadata display names tolerate malformed or repeated Redis keys."""
        service._stream_name_cache.clear()
        self.assertEqual(service._extract_stream_id('invalid-key'), '')
        self.assertEqual(service._decode_stream_name('invalid-key'), 'Unknown')
        self.assertEqual(
            service._decode_stream_name('stream_metadata:site|not-base64'),
            'Unknown',
        )

        encoded_name = 'Q2FtMQ=='
        service._stream_name_cache[encoded_name] = 'Cached camera'
        self.assertEqual(
            service._decode_stream_name(
                f'stream_metadata:site|{encoded_name}',
            ),
            'Cached camera',
        )

        service._stream_name_cache.clear()
        service._stream_name_cache.update(
            {f'cached-{index}': 'value' for index in range(512)},
        )
        self.assertEqual(
            service._decode_stream_name(
                f'stream_metadata:site|{encoded_name}',
            ),
            'Cam1',
        )
        self.assertEqual(service._stream_name_cache, {encoded_name: 'Cam1'})

    async def test_get_metadata_keys_for_label_empty(self) -> None:
        """Exercise this test."""
        self.mock_rds.scan.side_effect = [(0, [])]

        result = await get_metadata_keys_for_label(self.mock_rds, 'mylabel')

        self.assertEqual(result, [])

    async def test_get_metadata_keys_for_label_non_empty(self) -> None:
        """Exercise this test."""
        self.mock_rds.scan.side_effect = [
            (123, [b'stream_metadata:abcd|key2']),
            (0, [b'stream_metadata:abcd|key1', b'other:key']),
        ]

        result = await get_metadata_keys_for_label(self.mock_rds, 'ignored')

        self.assertEqual(
            result,
            ['stream_metadata:abcd|key1', 'stream_metadata:abcd|key2'],
        )

    async def test_fetch_latest_metadata_for_key_no_messages(self) -> None:
        """Exercise this test."""
        self.mock_rds.xread.return_value = []

        result = await fetch_latest_metadata_for_key(
            self.mock_rds,
            'stream_metadata:bGFiZWw=|Q2FtMQ==',
            '$',
        )

        self.assertIsNone(result)
        self.mock_rds.xread.assert_awaited_once_with(
            {'stream_metadata:bGFiZWw=|Q2FtMQ==': '$'},
            count=1,
            block=2000,
        )

    async def test_fetch_latest_metadata_for_key_empty_stream(self) -> None:
        """Exercise this test."""
        self.mock_rds.xread.return_value = [
            (b'stream_metadata:bGFiZWw=|Q2FtMQ==', []),
        ]

        result = await fetch_latest_metadata_for_key(
            self.mock_rds,
            'stream_metadata:bGFiZWw=|Q2FtMQ==',
            '1678889999-0',
        )

        self.assertIsNone(result)

    async def test_fetch_latest_metadata_for_key_without_frame(self) -> None:
        """Exercise this test."""
        self.mock_rds.xread.return_value = [
            (
                b'stream_metadata:bGFiZWw=|Q2FtMQ==',
                [(b'1678889999-0', {b'has_warning': b'1'})],
            ),
        ]

        result = await fetch_latest_metadata_for_key(
            self.mock_rds,
            'stream_metadata:bGFiZWw=|Q2FtMQ==',
            '0-0',
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result['id'], '1678889999-0')
        self.assertEqual(result['key'], 'Cam1')
        self.assertEqual(result['stream_id'], 'Q2FtMQ==')
        self.assertEqual(result['has_warning'], '1')

    async def test_fetch_latest_metadata_skips_repeated_message_id(self) -> None:
        """The SSE poller ignores the Redis item it has already sent."""
        self.mock_rds.xread.return_value = [
            (
                b'stream_metadata:bGFiZWw=|Q2FtMQ==',
                [(b'1678889999-0', {b'has_warning': b'1'})],
            ),
        ]

        self.assertIsNone(
            await fetch_latest_metadata_for_key(
                self.mock_rds,
                'stream_metadata:bGFiZWw=|Q2FtMQ==',
                '1678889999-0',
            ),
        )


if __name__ == '__main__':
    unittest.main()
