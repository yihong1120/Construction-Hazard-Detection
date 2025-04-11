from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import patch

from examples.streaming_web.backend.redis_service import (
    fetch_latest_frame_for_key,
)
from examples.streaming_web.backend.redis_service import fetch_latest_frames
from examples.streaming_web.backend.redis_service import get_keys_for_label
from examples.streaming_web.backend.redis_service import scan_for_labels
from examples.streaming_web.backend.redis_service import store_to_redis
from examples.streaming_web.backend.utils import Utils


class TestRedisService(unittest.IsolatedAsyncioTestCase):
    """
    Tests for the functions in redis_service.py
    """

    def setUp(self) -> None:
        self.mock_rds = AsyncMock()

    # ----------------------------------------------------------------------
    # Tests for scan_for_labels
    # ----------------------------------------------------------------------
    async def test_scan_for_labels_empty(self):
        """
        When Redis SCAN returns no keys, we get an empty list.
        """
        self.mock_rds.scan.side_effect = [(0, [])]
        result = await scan_for_labels(self.mock_rds)
        self.assertEqual(result, [])

    async def test_scan_for_labels_non_empty(self):
        """
        When Redis SCAN returns keys, we decode them and filter out
        the ones with 'test' in the label.
        """
        self.mock_rds.scan.side_effect = [
            (123, [b'stream_frame:YWJj|ZGVm']),     # 'abc|def' in base64
            (0, [b'stream_frame:dGVzdA|bGFiZWw=']),  # 'test|label' in base64
        ]
        result = await scan_for_labels(self.mock_rds)
        # 'test' label is ignored, so only 'abc' remains
        self.assertEqual(result, ['abc'])

    async def test_scan_for_labels_base64_error(self):
        """
        For invalid base64, we hit the except block and 'continue'
        without crashing.
        """
        # 'not_base64!' is invalid base64 => triggers decode exception
        self.mock_rds.scan.side_effect = [
            (123, [b'stream_frame:not_base64!|ZGVm']),
            (0, []),
        ]
        result = await scan_for_labels(self.mock_rds)
        # Because we skip invalid base64 and there's no 'test' in the label,
        # we end up with an empty list.
        self.assertEqual(result, [])

    # ----------------------------------------------------------------------
    # Tests for get_keys_for_label
    # ----------------------------------------------------------------------
    async def test_get_keys_for_label_empty(self):
        """When no matching keys, returns an empty list."""
        self.mock_rds.scan.side_effect = [(0, [])]
        result = await get_keys_for_label(self.mock_rds, 'mylabel')
        self.assertEqual(result, [])

    async def test_get_keys_for_label_non_empty(self):
        """Should collect all matching keys for a given label."""
        self.mock_rds.scan.side_effect = [
            (456, [b'stream_frame:abcd|key1']),
            (0, [b'stream_frame:abcd|key2']),
        ]
        result = await get_keys_for_label(self.mock_rds, 'ignored')
        self.assertEqual(
            result, [
                'stream_frame:abcd|key1',
                'stream_frame:abcd|key2',
            ],
        )

    # ----------------------------------------------------------------------
    # Tests for fetch_latest_frames
    # ----------------------------------------------------------------------
    async def test_fetch_latest_frames_no_messages(self):
        """
        If xrevrange returns nothing for each key, we get an empty result.
        """
        last_ids = {'somekey': '0-0', 'anotherkey': '0-0'}
        self.mock_rds.xrevrange.return_value = []
        result = await fetch_latest_frames(self.mock_rds, last_ids)
        self.assertEqual(result, [])

    async def test_fetch_latest_frames_has_frame(self):
        """
        If there's frame data, we decode it. 'cone_polygons' is tested,
        but 'pole_polygons' is omitted for brevity.
        """
        key = 'stream_frame:encoded_label|encoded_key'
        last_ids = {key: '0-0'}

        message_id = b'1678889999-0'
        data_map = {
            b'frame': b'\x89PNG fake image bytes',
            b'warnings': b'warning data',
            b'cone_polygons': b'polygon data',
            b'detection_items': b'detection data',
            b'width': b'640',
            b'height': b'480',
        }
        # Mock the xrevrange to return a message with the expected data
        # and the message ID
        self.mock_rds.xrevrange.return_value = [(message_id, data_map)]

        with patch('base64.urlsafe_b64decode', return_value=b'decoded_key'):
            result = await fetch_latest_frames(self.mock_rds, last_ids)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['key'], 'decoded_key')
        self.assertEqual(result[0]['frame_bytes'], b'\x89PNG fake image bytes')
        self.assertEqual(result[0]['warnings'], 'warning data')
        self.assertEqual(result[0]['cone_polygons'], 'polygon data')
        self.assertEqual(result[0]['detection_items'], 'detection data')
        self.assertEqual(result[0]['width'], '640')
        self.assertEqual(result[0]['height'], '480')

    async def test_fetch_latest_frames_bad_split(self):
        """
        If the split fails, we set 'stream_name' to 'Unknown'.
        """
        last_ids = {'invalidkey': '0-0'}
        message_id = b'100-0'
        data_map = {
            b'frame': b'fake_image',
        }
        self.mock_rds.xrevrange.return_value = [(message_id, data_map)]
        result = await fetch_latest_frames(self.mock_rds, last_ids)
        self.assertEqual(result[0]['key'], 'Unknown')

    async def test_fetch_latest_frames_base64_error(self):
        """
        If the base64 decode fails, we set 'stream_name' to 'Unknown'.
        """
        last_ids = {'stream_frame:encoded_label|encoded_key': '0-0'}
        message_id = b'101-0'
        data_map = {b'frame': b'fake_image'}
        self.mock_rds.xrevrange.return_value = [(message_id, data_map)]

        with patch(
            'base64.urlsafe_b64decode',
            side_effect=ValueError('bad base64'),
        ):
            result = await fetch_latest_frames(self.mock_rds, last_ids)
        self.assertEqual(result[0]['key'], 'Unknown')

    # ----------------------------------------------------------------------
    # Tests for fetch_latest_frame_for_key
    # ----------------------------------------------------------------------
    async def test_fetch_latest_frame_for_key_no_messages(self):
        """If xrevrange returns no messages, return None."""
        self.mock_rds.xrevrange.return_value = []
        result = await fetch_latest_frame_for_key(
            self.mock_rds,
            'some_key',
            '0-0',
        )
        self.assertIsNone(result)

    async def test_fetch_latest_frame_for_key_with_frame(self):
        """If there's frame data, decode and return it."""
        message_id = b'999-0'
        data_map = {
            b'frame': b'fake_image',
            b'warnings': b'warn',
            b'cone_polygons': b'polys',
            b'pole_polygons': b'pole-data',
            b'detection_items': b'detects',
            b'width': b'320',
            b'height': b'240',
        }
        self.mock_rds.xrevrange.return_value = [(message_id, data_map)]
        result = await fetch_latest_frame_for_key(
            self.mock_rds,
            'some_key',
            '0-0',
        )
        self.assertIsNotNone(result)
        self.assertEqual(result['id'], '999-0')
        self.assertEqual(result['frame_bytes'], b'fake_image')
        self.assertEqual(result['warnings'], 'warn')
        self.assertEqual(result['cone_polygons'], 'polys')
        self.assertEqual(result['pole_polygons'], 'pole-data')
        self.assertEqual(result['detection_items'], 'detects')
        self.assertEqual(result['width'], '320')
        self.assertEqual(result['height'], '240')

    async def test_fetch_latest_frame_for_key_no_frame_data(self):
        """
        If there's no 'frame' key in the data map, we return None.
        """
        message_id = b'202-0'
        data_map = {
            # no b"frame" key
            b'warnings': b'warn',
        }
        self.mock_rds.xrevrange.return_value = [(message_id, data_map)]
        result = await fetch_latest_frame_for_key(
            self.mock_rds,
            'some_key',
            '0-0',
        )
        self.assertIsNone(result)

    # ----------------------------------------------------------------------
    # Tests for store_to_redis
    # ----------------------------------------------------------------------
    async def test_store_to_redis_no_frame_bytes(self):
        """
        If frame_bytes is None or empty, do nothing.
        """
        await store_to_redis(
            rds=self.mock_rds,
            site='demo_site',
            stream_name='demo_stream',
            frame_bytes=None,
            warnings_json='',
            cone_polygons_json='',
            pole_polygons_json='',
            detection_items_json='',
            width=0,
            height=0,
        )
        self.mock_rds.xadd.assert_not_awaited()

    async def test_store_to_redis_ok(self):
        """
        If frame_bytes is present, we xadd the data with correct fields.
        """
        with patch.object(
            Utils,
            'encode',
            side_effect=lambda x: f"encoded({x})",
        ):
            await store_to_redis(
                rds=self.mock_rds,
                site='demo_site',
                stream_name='demo_stream',
                frame_bytes=b'fake_image',
                warnings_json='{}',
                cone_polygons_json='[]',
                pole_polygons_json='[]',
                detection_items_json='{}',
                width=640,
                height=360,
            )
        expected_key = 'stream_frame:encoded(demo_site)|encoded(demo_stream)'
        self.mock_rds.xadd.assert_awaited_once_with(
            expected_key,
            {
                'frame': b'fake_image',
                'warnings': '{}',
                'cone_polygons': '[]',
                'pole_polygons': '[]',
                'detection_items': '{}',
                'width': 640,
                'height': 360,
            },
            maxlen=10,
        )


if __name__ == '__main__':
    unittest.main()

'''
pytest \
    --cov=examples.streaming_web.backend.redis_service \
    --cov-report=term-missing \
    tests/examples/streaming_web/backend/redis_service_test.py
'''
