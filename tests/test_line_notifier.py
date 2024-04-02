import unittest
from unittest.mock import patch, MagicMock
from src.line_notifier import LineNotifier

class TestLineNotifier(unittest.TestCase):
    """
    Test cases for LineNotifier class in line_notifier.py.
    """

    @patch('src.line_notifier.requests.post')
    @patch('src.line_notifier.os.getenv', return_value='test_line_notify_token')
    def test_send_notification(self, mock_getenv, mock_post):
        """
        Test the send_notification method to ensure it sends a request to the LINE Notify API
        and returns the correct HTTP status code, both with and without an image.
        """
        # Mock the response object to simulate a successful request
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Instantiate LineNotifier with a mock token
        notifier = LineNotifier()

        # Call the send_notification method with a test message and without an image
        test_message = 'This is a test message without an image.'
        status_code_without_image = notifier.send_notification(test_message)
        
        # Assert for sending notification without an image
        mock_post.assert_called_with(
            'https://notify-api.line.me/api/notify',
            headers={
                'Authorization': 'Bearer test_line_notify_token'
            },
            data={'message': test_message},
            files=None
        )
        self.assertEqual(status_code_without_image, 200, "The function should return HTTP status code 200 for a notification without an image")

        # Reset mock
        mock_post.reset_mock()

        # Call the send_notification method again with a test message and an image
        test_message_with_image = 'This is a test message with an image.'
        status_code_with_image = notifier.send_notification(test_message_with_image, 'test_image.jpg')

        # Assert for sending notification with an image
        # Note: Since we can't predict the exact file object, we check if 'files' in call_args is not None
        self.assertTrue(mock_post.call_args[1]['files'] is not None, "The function should have been called with files argument for an image")
        self.assertEqual(status_code_with_image, 200, "The function should return HTTP status code 200 for a notification with an image")

if __name__ == '__main__':
    unittest.main()