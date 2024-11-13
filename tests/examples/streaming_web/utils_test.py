from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import redis

from examples.streaming_web.utils import encode_image
from examples.streaming_web.utils import get_image_data
from examples.streaming_web.utils import get_labels
from examples.streaming_web.utils import process_image_data


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
                b'label1_image1',
                b'label1_image2',
                b'label2_image1',
                b'test_image',
                b'__invalid_key',
                b'_another_invalid_key',
                b'label3_image1',
            ],
        )

        # Call the function
        result = await get_labels(self.redis_mock)

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
                b'label1_image1',
                b'label1_image2',
            ],
        )

        # Mock the Redis mget method to return image data
        self.redis_mock.mget.return_value = [
            b'image_data_1',
            b'image_data_2',
        ]

        # Call the function
        result = await get_image_data(self.redis_mock, label)

        # Check the expected result
        expected_result = [
            (await encode_image(b'image_data_1'), 'image1'),
            (await encode_image(b'image_data_2'), 'image2'),
        ]
        self.assertEqual(result, expected_result)

    @patch('examples.streaming_web.utils.encode_image', wraps=encode_image)
    async def test_get_image_data_no_image(
        self,
        mock_encode_image: MagicMock,
    ) -> None:
        """
        Test get_image_data function when some images are missing.
        """
        # Mock the Redis scan method to return keys matching the label
        label = 'label1'
        self.redis_mock.scan.return_value = (
            0, [
                b'label1_image1',
                b'label1_image2',
            ],
        )

        # Mock the Redis mget method to return None for an image
        self.redis_mock.mget.return_value = [
            None,  # Simulate missing image
            b'image_data_2',
        ]

        # Call the function
        result = await get_image_data(self.redis_mock, label)

        # Check that only the existing image is returned
        expected_result = [
            (await encode_image(b'image_data_2'), 'image2'),
        ]
        self.assertEqual(result, expected_result)

        # Ensure encode_image was called exactly once for the valid image
        mock_encode_image.assert_called_once_with(b'image_data_2')

    async def test_process_image_data(self) -> None:
        """
        Test the process_image_data function to ensure image data
        processesed correctly.
        """
        key = b'label1_image1'
        image = b'image_data_1'

        # Call the function
        result = await process_image_data(key, image)

        # Check the expected result
        expected_result = (await encode_image(image), 'image1')
        self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
