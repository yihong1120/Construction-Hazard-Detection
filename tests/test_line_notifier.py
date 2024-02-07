import unittest
from unittest.mock import patch
from src.line_notifier import send_line_notification

class TestLineNotifier(unittest.TestCase):
    """
    Test cases for line_notifier.py.
    """

    @patch('src.line_notifier.requests.post')
    def test_send_line_notification(self, mock_post):
        """
        Test the send_line_notification function to ensure it sends a request to the LINE Notify API
        and returns the correct HTTP status code.
        """
        # Mock the response object to simulate a successful request
        mock_response = unittest.mock.Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Call the send_line_notification function with a test token and message
        test_token = 'test_line_notify_token'
        test_message = 'This is a test message.'
        status_code = send_line_notification(test_token, test_message)

        # Assert that the requests.post method was called with the correct URL and headers
        mock_post.assert_called_once_with(
            'https://notify-api.line.me/api/notify',
            headers={
                'Authorization': f'Bearer {test_token}',
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            data={'message': test_message}
        )

        # Assert that the function returns the correct status code
        self.assertEqual(status_code, 200, "The function should return HTTP status code 200")

if __name__ == '__main__':
    unittest.main()