from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

from src.notifiers.messenger_notifier import main
from src.notifiers.messenger_notifier import MessengerNotifier


class TestMessengerNotifier(unittest.TestCase):
    """
    Unit tests for the MessengerNotifier class methods.
    """

    messenger_notifier: MessengerNotifier

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up class method to initialise the MessengerNotifier instance.
        """
        cls.messenger_notifier = MessengerNotifier(
            page_access_token='test_page_access_token',
        )

    def test_init(self) -> None:
        """
        Test if the MessengerNotifier instance is initialised correctly.
        """
        self.assertEqual(
            self.messenger_notifier.page_access_token,
            'test_page_access_token',
        )

    @patch('requests.post')
    def test_send_notification_no_image(self, mock_post: Any) -> None:
        """
        Test sending a notification without an image.

        Args:
            mock_post (Any): Mock object for the requests.post method.
        """
        mock_post.return_value.status_code = 200
        recipient_id: str = 'test_recipient_id'
        message: str = 'Hello, Messenger!'
        response_code: int = self.messenger_notifier.send_notification(
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
    def test_send_notification_with_image(self, mock_post: Any) -> None:
        """
        Test sending a notification with an image.

        Args:
            mock_post (Any): Mock object for the requests.post method.
        """
        mock_post.return_value.status_code = 200
        recipient_id: str = 'test_recipient_id'
        message: str = 'Hello, Messenger!'
        image: np.ndarray = np.zeros((100, 100, 3), dtype=np.uint8)
        response_code: int = self.messenger_notifier.send_notification(
            recipient_id, message, image=image,
        )
        self.assertEqual(response_code, 200)
        self.assertTrue(mock_post.called)
        args, kwargs = mock_post.call_args
        files = kwargs['files']
        self.assertIn('filedata', files)
        self.assertEqual(files['filedata'][0], 'image.png')
        self.assertEqual(files['filedata'][2], 'image/png')

    def test_missing_page_access_token(self) -> None:
        """
        Test initialisation without a page access token.
        """
        with self.assertRaises(
            ValueError,
            msg='FACEBOOK_PAGE_ACCESS_TOKEN missing.',
        ):
            MessengerNotifier(page_access_token=None)

    @patch(
        'src.notifiers.messenger_notifier.MessengerNotifier.send_notification',
        return_value=200,
    )
    @patch(
        'src.notifiers.messenger_notifier.os.getenv',
        return_value='test_page_access_token',
    )
    def test_main(
        self,
        mock_getenv: MagicMock,
        mock_send_notification: MagicMock,
    ) -> None:
        """
        Test the main function.
        """
        with patch('builtins.print') as mock_print:
            main()
            mock_send_notification.assert_called_once()
            mock_print.assert_called_with('Response code: 200')


if __name__ == '__main__':
    unittest.main()
