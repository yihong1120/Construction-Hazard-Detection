from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from src.messenger_notifier import MessengerNotifier


class TestMessengerNotifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.messenger_notifier = MessengerNotifier(
            page_access_token='test_page_access_token',
        )

    def test_init(self):
        """Test if the MessengerNotifier instance is initialised correctly."""
        self.assertEqual(
            self.messenger_notifier.page_access_token,
            'test_page_access_token',
        )

    @patch('requests.post')
    def test_send_notification_no_image(self, mock_post):
        """Test sending a notification without an image."""
        mock_post.return_value.status_code = 200
        recipient_id = 'test_recipient_id'
        message = 'Hello, Messenger!'
        response_code = self.messenger_notifier.send_notification(
            recipient_id, message,
        )
        self.assertEqual(response_code, 200)
        self.assertTrue(mock_post.called)
        args, kwargs = mock_post.call_args
        self.assertEqual(
            kwargs['json'], {
                'message': {
                    'text': message,
                }, 'recipient': {'id': recipient_id},
            },
        )

    @patch('requests.post')
    def test_send_notification_with_image(self, mock_post):
        """Test sending a notification with an image."""
        mock_post.return_value.status_code = 200
        recipient_id = 'test_recipient_id'
        message = 'Hello, Messenger!'
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        response_code = self.messenger_notifier.send_notification(
            recipient_id, message, image=image,
        )
        self.assertEqual(response_code, 200)
        self.assertTrue(mock_post.called)
        args, kwargs = mock_post.call_args
        files = kwargs['files']
        self.assertIn('filedata', files)
        self.assertEqual(files['filedata'][0], 'image.png')
        self.assertEqual(files['filedata'][2], 'image/png')

    def test_missing_page_access_token(self):
        """Test initialisation without a page access token."""
        with self.assertRaises(
            ValueError,
            msg='FACEBOOK_PAGE_ACCESS_TOKEN missing.',
        ):
            MessengerNotifier(page_access_token=None)


if __name__ == '__main__':
    unittest.main()
