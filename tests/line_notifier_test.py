from __future__ import annotations

import unittest
from io import BytesIO
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.line_notifier import LineNotifier


class TestLineNotifier(unittest.TestCase):
    def setUp(self):
        self.line_token = 'test_token'
        self.message = 'Test Message'
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.notifier = LineNotifier(line_token=self.line_token)

    @patch.dict('os.environ', {'LINE_NOTIFY_TOKEN': 'test_env_token'})
    def test_init_with_env_token(self):
        notifier = LineNotifier()
        self.assertEqual(notifier.line_token, 'test_env_token')

    def test_init_with_provided_token(self):
        notifier = LineNotifier(line_token='provided_token')
        self.assertEqual(notifier.line_token, 'provided_token')

    def test_init_without_token(self):
        with self.assertRaises(ValueError):
            LineNotifier(line_token=None)

    @patch('src.line_notifier.requests.post')
    def test_send_notification_without_image(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        status_code = self.notifier.send_notification(self.message)
        self.assertEqual(status_code, 200)
        mock_post.assert_called_once_with(
            'https://notify-api.line.me/api/notify',
            headers={'Authorization': f'Bearer {self.line_token}'},
            params={'message': self.message},
        )

    @patch('src.line_notifier.requests.post')
    def test_send_notification_with_image(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        status_code = self.notifier.send_notification(self.message, self.image)
        self.assertEqual(status_code, 200)
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertIn('files', kwargs)
        self.assertIn('imageFile', kwargs['files'])

        # Check if the image is correctly converted and sent
        image_file = kwargs['files']['imageFile']
        self.assertEqual(image_file[0], 'image.png')
        self.assertEqual(image_file[2], 'image/png')
        image = Image.open(image_file[1])
        self.assertTrue(np.array_equal(np.array(image), self.image))

    @patch('src.line_notifier.requests.post')
    def test_send_notification_with_bytes_image(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        buffer = BytesIO()
        Image.fromarray(self.image).save(buffer, format='PNG')
        buffer.seek(0)
        image_bytes = buffer.read()

        status_code = self.notifier.send_notification(
            self.message, image_bytes,
        )
        self.assertEqual(status_code, 200)
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertIn('files', kwargs)
        self.assertIn('imageFile', kwargs['files'])

        # Check if the image is correctly converted and sent
        image_file = kwargs['files']['imageFile']
        self.assertEqual(image_file[0], 'image.png')
        self.assertEqual(image_file[2], 'image/png')
        image = Image.open(image_file[1])
        self.assertTrue(np.array_equal(np.array(image), self.image))


if __name__ == '__main__':
    unittest.main()
