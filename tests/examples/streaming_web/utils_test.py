from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
import redis
from examples.streaming_web.utils import get_labels, get_image_data, encode_image


class TestUtils(unittest.TestCase):
    
    def setUp(self):
        # Set up a mock Redis instance
        self.redis_mock = MagicMock(spec=redis.Redis)
        
    def test_get_labels(self):
        # Mock the Redis scan method to return some keys
        self.redis_mock.scan.return_value = (0, [
            b'label1_image1',
            b'label1_image2',
            b'label2_image1',
            b'test_image',
            b'__invalid_key',
            b'_another_invalid_key',
            b'label3_image1',
        ])
        
        # Call the function
        result = get_labels(self.redis_mock)
        
        # Check the expected result
        expected_result = ['label1', 'label2', 'label3']
        self.assertEqual(result, expected_result)
        
    def test_get_image_data(self):
        # Mock the Redis scan method to return keys matching the label
        label = 'label1'
        self.redis_mock.scan.return_value = (0, [
            b'label1_image1',
            b'label1_image2',
        ])
        
        # Mock the Redis get method to return image data
        self.redis_mock.get.side_effect = [
            b'image_data_1',
            b'image_data_2',
        ]
        
        # Call the function
        result = get_image_data(self.redis_mock, label)
        
        # Check the expected result
        expected_result = [
            (encode_image(b'image_data_1'), 'image1'),
            (encode_image(b'image_data_2'), 'image2'),
        ]
        self.assertEqual(result, expected_result)
        
    @patch('examples.streaming_web.utils.encode_image', wraps=encode_image)
    def test_get_image_data_no_image(self, mock_encode_image):
        # Mock the Redis scan method to return keys matching the label
        label = 'label1'
        self.redis_mock.scan.return_value = (0, [
            b'label1_image1',
            b'label1_image2',
        ])
        
        # Mock the Redis get method to return None for an image
        self.redis_mock.get.side_effect = [
            None,  # Simulate missing image
            b'image_data_2',
        ]
        
        # Call the function
        result = get_image_data(self.redis_mock, label)
        
        # Check that only the existing image is returned
        expected_result = [
            (encode_image(b'image_data_2'), 'image2'),
        ]
        self.assertEqual(result, expected_result)
        
        # Ensure encode_image was called exactly once for the valid image
        mock_encode_image.assert_called_once_with(b'image_data_2')


if __name__ == '__main__':
    unittest.main()