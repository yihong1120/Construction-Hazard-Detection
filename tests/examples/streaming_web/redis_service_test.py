from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock

from examples.streaming_web import metadata_keys
from examples.streaming_web import redis_service as service
from examples.streaming_web.metadata_keys import build_metadata_key
from examples.streaming_web.metadata_wire import build_metadata_payload
from examples.streaming_web.metadata_wire import MetadataFrame
from examples.streaming_web.redis_service import (
    fetch_latest_metadata_for_key,
)


class TestRedisService(unittest.IsolatedAsyncioTestCase):
    """Tests for compact live metadata Redis helpers."""

    def setUp(self) -> None:
        """Prepare test fixtures."""
        self.mock_rds = AsyncMock()

    def test_rust_overlay_preserves_coordinates_and_owner(self) -> None:
        snapshot = {
            'owner': 'worker-a', 'generation': 3, 'sequence': 42,
            'frame_width': 1920, 'frame_height': 1080, 'source_pts': 1.5,
            'track_data': [[10, 20, 30, 40, .9, 2, 7, 0]],
        }
        record = service._build_metadata_record(
            build_metadata_key('site', 'camera'), b'1-0',
            {b'has_warning': b'1', b'overlay': json.dumps(snapshot).encode()},
        )
        payload = build_metadata_payload(record)
        self.assertTrue(payload['has_warning'])
        self.assertEqual(payload['overlay'], snapshot)

    def test_metadata_wire_preserves_unicode_and_protocol_ids(self) -> None:
        record = service._build_metadata_record(
            build_metadata_key('site', 'camera'), b'3-2',
            {
                b'has_warning': b'0',
                b'overlay': json.dumps({'label': '安全'}).encode(),
            },
        )
        frame = MetadataFrame(record)

        self.assertEqual(
            frame.sse,
            b'id: 3-2\nevent: metadata\n'
            b'data: {"has_warning":false,"overlay":{"label":"\\u5b89\\u5168"},'
            b'"id":"3-2"}\n\n',
        )
        self.assertEqual(
            frame.websocket, '{"has_warning":false,"overlay":{"label":"安全"}}',
        )

    def test_legacy_warning_payload_has_no_overlay(self) -> None:
        record = service._build_metadata_record(
            build_metadata_key('site', 'camera'), b'1-0',
            {b'has_warning': b'0'},
        )
        self.assertNotIn('overlay', record)

    def test_build_metadata_key_uses_encoded_site_and_stream(self) -> None:
        key = build_metadata_key('site-a', 'cam-1')

        self.assertEqual(key, 'stream_metadata:c2l0ZS1h|Y2FtLTE=')

    def test_metadata_key_helpers_reject_invalid_values(
        self,
    ) -> None:
        """Malformed metadata keys are rejected at the contract boundary."""
        with self.assertRaisesRegex(ValueError, 'invalid_metadata_key'):
            service._extract_stream_id('invalid-key')
        with self.assertRaisesRegex(ValueError, 'invalid_metadata_key'):
            service._decode_stream_name('invalid-key')

    async def test_metadata_key_generations_accept_bytes_and_invalid_values(
        self,
    ) -> None:
        """Generation keys support Redis bytes and malformed counters.

        Invalid values safely fall back to the base key namespace.
        """
        self.mock_rds.get.return_value = b'5'
        self.assertEqual(
            await metadata_keys.get_metadata_site_generation(
                self.mock_rds,
                'Site A',
            ),
            5,
        )
        self.mock_rds.get.return_value = 'not-a-number'
        self.assertEqual(
            await metadata_keys.get_metadata_site_generation(
                self.mock_rds,
                'Site A',
            ),
            0,
        )
        self.assertIn(
            ':g3|',
            metadata_keys.build_metadata_key('Site A', 'Camera A', 3),
        )
        self.assertIn(
            ':g3|stream-id',
            metadata_keys.build_metadata_key_from_stream_id(
                'Site A',
                'stream-id',
                3,
            ),
        )
        self.assertFalse(metadata_keys.is_metadata_key('not-metadata'))

    async def test_fetch_latest_metadata_for_key_no_messages(self) -> None:
        self.mock_rds.xread.return_value = []

        result = await fetch_latest_metadata_for_key(
            self.mock_rds,
            'stream_metadata:bGFiZWw=|Q2FtMQ==',
            '$',
        )

        self.assertIsNone(result)
        self.mock_rds.xread.assert_awaited_once_with(
            {'stream_metadata:bGFiZWw=|Q2FtMQ==': '$'},
            count=64,
            block=2000,
        )

    async def test_fetch_latest_metadata_for_key_empty_stream(self) -> None:
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
        self.assertTrue(result['has_warning'])

    async def test_fetch_latest_metadata_skips_repeated_message_id(
        self,
    ) -> None:
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
