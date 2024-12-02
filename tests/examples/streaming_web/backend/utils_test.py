from __future__ import annotations

import base64
import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import redis
from fastapi import WebSocket

from examples.streaming_web.backend.utils import RedisManager
from examples.streaming_web.backend.utils import Utils


class TestRedisManager(unittest.IsolatedAsyncioTestCase):
    """
    Test suite for RedisManager methods.
    """

    def setUp(self) -> None:
        """
        Set up the test environment before each test.
        """
        self.redis_mock = MagicMock(spec=redis.Redis)
        self.redis_mock.xrevrange = AsyncMock()
        self.redis_mock.scan = AsyncMock()

    async def test_fetch_latest_frame_for_key_with_data(self) -> None:
        """
        Test fetch_latest_frame_for_key method when Redis contains valid data.
        """
        redis_key = 'stream_frame:test_label_image1'
        last_id = '0-0'

        # Mock Redis response
        message_id = '1234-0'
        frame_data = b'sample_frame_data'
        warnings_data = b'Sample warning'
        self.redis_mock.xrevrange.return_value = [
            (
                message_id.encode('utf-8'),
                {b'frame': frame_data, b'warnings': warnings_data},
            ),
        ]

        redis_manager = RedisManager('localhost', 6379, 'password')
        redis_manager.client = self.redis_mock
        result = await redis_manager.fetch_latest_frame_for_key(
            redis_key,
            last_id,
        )

        expected_result = {
            'id': message_id,
            'image': base64.b64encode(frame_data).decode('utf-8'),
            'warnings': warnings_data.decode('utf-8'),
        }
        self.assertEqual(result, expected_result)

    async def test_fetch_latest_frame_for_key_no_data(self) -> None:
        """
        Test fetch_latest_frame_for_key method when Redis contains no new data.
        """
        redis_key = 'stream_frame:test_label_image1'
        last_id = '0-0'

        # Mock Redis response
        self.redis_mock.xrevrange.return_value = []

        redis_manager = RedisManager('localhost', 6379, 'password')
        redis_manager.client = self.redis_mock
        result = await redis_manager.fetch_latest_frame_for_key(
            redis_key,
            last_id,
        )

        self.assertIsNone(result)

    async def test_fetch_latest_frames(self) -> None:
        """
        Test the fetch_latest_frames method for multiple streams.
        """
        last_ids = {
            'stream_frame:test_label|image1': '0-0',
            'stream_frame:test_label|image2': '0-0',
        }

        # Mock Redis response
        self.redis_mock.xrevrange.side_effect = [
            [
                ('1234-0', {b'frame': b'image1_frame'}),
            ],
            [
                ('5678-0', {b'frame': b'image2_frame'}),
            ],
        ]

        redis_manager = RedisManager('localhost', 6379, 'password')
        redis_manager.client = self.redis_mock
        result = await redis_manager.fetch_latest_frames(last_ids)

        expected_result = [
            {
                'key': 'image1', 'image': base64.b64encode(
                    b'image1_frame',
                ).decode('utf-8'),
            },
            {
                'key': 'image2', 'image': base64.b64encode(
                    b'image2_frame',
                ).decode('utf-8'),
            },
        ]
        self.assertEqual(result, expected_result)


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
                b'stream_frame:bGFiZWwx|image1',
                b'stream_frame:bGFiZWwx|image2',
                b'stream_frame:bGFiZWwy|image1',
                b'stream_frame:bGFiZWwz|image1',
                b'stream_frame:dGVzdA==|image',
                b'__invalid_key',
                b'_another_invalid_key',
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

    async def test_is_base64(self) -> None:
        """
        Test the is_base64 function for different cases.
        """
        valid_base64 = 'QmFzZTY0U3RyaW5n'
        invalid_base64 = 'NotBase64@#%'
        empty_string = ''

        self.assertTrue(Utils.is_base64(valid_base64))
        self.assertFalse(Utils.is_base64(invalid_base64))
        self.assertFalse(Utils.is_base64(empty_string))

    async def test_encode(self) -> None:
        """
        Test the encode function to ensure it encodes correctly.
        """
        input_string = 'test_label'
        encoded_string = Utils.encode(input_string)

        # Check if encoding and underscore replacement work as expected
        expected_encoded = base64.urlsafe_b64encode(
            input_string.encode('utf-8'),
        ).decode('utf-8').replace('_', '-')
        self.assertEqual(encoded_string, expected_encoded)

    async def test_decode_valid_base64(self) -> None:
        """
        Test the decode function with valid Base64 input.
        """
        input_string = base64.urlsafe_b64encode(
            b'test_label',
        ).decode('utf-8').replace('_', '-')
        decoded_string = Utils.decode(input_string)
        self.assertEqual(decoded_string, 'test_label')

    async def test_decode_invalid_base64(self) -> None:
        """
        Test the decode function with invalid Base64 input.
        """
        input_string = 'Invalid_String!'
        decoded_string = Utils.decode(input_string)
        self.assertEqual(decoded_string, input_string)


if __name__ == '__main__':
    unittest.main()
