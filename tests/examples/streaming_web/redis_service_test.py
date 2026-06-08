from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import patch

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
        self.mock_rds.xrevrange.return_value = []

        result = await fetch_latest_metadata_for_key(
            self.mock_rds,
            'stream_metadata:bGFiZWw=|Q2FtMQ==',
            '0-0',
        )

        self.assertIsNone(result)

    async def test_fetch_latest_metadata_for_key_unchanged(self) -> None:
        """Exercise this test."""
        self.mock_rds.xrevrange.return_value = [
            (b'1678889999-0', {b'has_warning': b'true'}),
        ]

        result = await fetch_latest_metadata_for_key(
            self.mock_rds,
            'stream_metadata:bGFiZWw=|Q2FtMQ==',
            '1678889999-0',
        )

        self.assertIsNone(result)

    async def test_fetch_latest_metadata_for_key_without_frame(self) -> None:
        """Exercise this test."""
        self.mock_rds.xrevrange.return_value = [
            (b'1678889999-0', {b'has_warning': b'true'}),
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
        self.assertEqual(result['has_warning'], 'true')


if __name__ == '__main__':
    unittest.main()
