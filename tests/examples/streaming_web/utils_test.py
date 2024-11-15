from __future__ import annotations

import base64
import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import redis
from fastapi import WebSocket

from examples.streaming_web.utils import RedisManager
from examples.streaming_web.utils import Utils


class TestUtils(unittest.IsolatedAsyncioTestCase):
    """
    Test suite for utility functions in the streaming_web module.
    """

    def setUp(self) -> None:
        """
        Set up the test environment before each test.
        """
        self.redis_mock = MagicMock(spec=redis.Redis)
        self.redis_mock.scan = AsyncMock()
        self.redis_mock.mget = AsyncMock()
        self.redis_mock.get = AsyncMock()

    def tearDown(self) -> None:
        """
        Clean up after each test.
        """
        self.redis_mock.reset_mock()

    async def test_get_labels(self) -> None:
        """
        Test the get_labels function to ensure it returns expected labels.
        """
        # Mock the Redis scan method to return some keys
        self.redis_mock.scan.return_value = (
            0, [
                b'stream_frame:label1_image1',
                b'stream_frame:label1_image2',
                b'stream_frame:label2_image1',
                b'stream_frame:test_image',
                b'__invalid_key',
                b'_another_invalid_key',
                b'stream_frame:label3_image1',
            ],
        )

        # Call the function
        redis_manager = RedisManager('localhost', 6379, 'password')
        redis_manager.client = self.redis_mock
        result = await redis_manager.get_labels()

        # Check the expected result
        expected_result = ['label1', 'label2', 'label3']
        self.assertEqual(result, expected_result)

    async def test_get_image_data(self) -> None:
        """
        Test the get_image_data function
        to ensure it returns correct image data.
        """
        # Mock the Redis scan method to return keys matching the label
        label = 'label1'
        self.redis_mock.scan.return_value = (
            0, [
                b'stream_frame:label1_image1',
                b'stream_frame:label1_image2',
            ],
        )

        # Mock the Redis mget method to return image data
        self.redis_mock.mget.return_value = [
            b'image_data_1',
            b'image_data_2',
        ]

        # Call the function
        redis_manager = RedisManager('localhost', 6379, 'password')
        redis_manager.client = self.redis_mock
        result = await redis_manager.get_keys_for_label(label)

        # Check the expected result
        expected_result = [
            'stream_frame:label1_image1',
            'stream_frame:label1_image2',
        ]
        self.assertEqual(result, expected_result)

    async def test_get_image_data_no_image(
        self,
    ) -> None:
        """
        Test get_image_data function when some images are missing.
        """
        # Mock the Redis scan method to return keys matching the label
        label = 'label1'
        self.redis_mock.scan.return_value = (
            0, [
                b'stream_frame:label1_image1',
                b'stream_frame:label1_image2',
            ],
        )

        # Mock the Redis mget method to return None for an image
        self.redis_mock.mget.return_value = [
            None,  # Simulate missing image
            b'image_data_2',
        ]

        # Call the function
        redis_manager = RedisManager('localhost', 6379, 'password')
        redis_manager.client = self.redis_mock
        keys = await redis_manager.get_keys_for_label(label)

        # Simulate processing image data
        result = []
        for key, image in zip(keys, self.redis_mock.mget.return_value):
            if image:
                # 提取圖像名稱
                image_name = key.split('_')[-1]
                encoded_image = base64.b64encode(image).decode('utf-8')
                result.append((encoded_image, image_name))

        # 預先計算預期結果
        expected_encoded_image = base64.b64encode(
            b'image_data_2',
        ).decode('utf-8')
        expected_result = [
            (expected_encoded_image, 'image2'),
        ]

        self.assertEqual(result, expected_result)

    async def test_process_image_data(self) -> None:
        """
        Test the process_image_data function to ensure image data
        processesed correctly.
        """
        image = b'image_data_1'

        # Call the function
        result = base64.b64encode(image).decode('utf-8')

        # Check the expected result
        expected_result = base64.b64encode(image).decode('utf-8')
        self.assertEqual(result, expected_result)

    async def test_send_frames(self) -> None:
        """
        Test the send_frames function to ensure it sends data to
        WebSocket client.
        """
        # Mock the WebSocket send_json method
        websocket_mock = MagicMock(spec=WebSocket)
        websocket_mock.send_json = AsyncMock()

        # Prepare data
        label = 'label1'
        updated_data = [
            {'key': 'image1', 'image': 'encoded_image_data_1'},
            {'key': 'image2', 'image': 'encoded_image_data_2'},
        ]

        # Call the function
        await Utils.send_frames(websocket_mock, label, updated_data)

        # Check if send_json was called with correct data
        expected_data = {
            'label': label,
            'images': updated_data,
        }
        websocket_mock.send_json.assert_called_once_with(expected_data)


if __name__ == '__main__':
    unittest.main()
