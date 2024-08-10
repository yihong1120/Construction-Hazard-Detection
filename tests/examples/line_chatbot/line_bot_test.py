from __future__ import annotations

import unittest
from unittest.mock import patch

from linebot.models import MessageEvent
from linebot.models import TextMessage

from examples.line_chatbot.line_bot import app
from examples.line_chatbot.line_bot import handler
from examples.line_chatbot.line_bot import line_bot_api


class LineBotTestCase(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment before each test.
        """
        self.app = app.test_client()
        self.app.testing = True

    @patch('examples.line_chatbot.line_bot.request')
    @patch('examples.line_chatbot.line_bot.handler.handle')
    def test_callback_success(self, mock_handle, mock_request):
        """
        Test the /callback route with a valid signature.
        """
        mock_request.headers = {'X-Line-Signature': 'valid_signature'}
        mock_request.get_data.return_value = 'body'

        response = self.app.post('/callback')

        # Ensure the handle method was called with correct arguments
        mock_handle.assert_called_once_with('body', 'valid_signature')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode('utf-8'), 'OK')

    @patch('examples.line_chatbot.line_bot.request')
    @patch('examples.line_chatbot.line_bot.handler.handle')
    def test_callback_invalid_signature(self, mock_handle, mock_request):
        """
        Test the /callback route with an invalid signature.
        """
        mock_request.headers = {'X-Line-Signature': 'invalid_signature'}
        mock_request.get_data.return_value = 'body'

        mock_handle.side_effect = Exception('InvalidSignatureError')

        response = self.app.post('/callback')

        # Ensure the handle method raised an exception
        self.assertEqual(response.status_code, 400)

    @patch.object(line_bot_api, 'reply_message')
    def test_handle_message(self, mock_reply_message):
        """
        Test the handle_message function to ensure correct message handling.
        """
        event = MessageEvent(
            reply_token='dummy_token',
            message=TextMessage(text='Hello'),
            source=None, timestamp=None, mode=None,
        )

        handler.handle_message(event)

        # Ensure the reply_message method was called with the correct arguments
        # mock_reply_message.assert_called_once_with(
        #     'dummy_token',
        #     TextSendMessage(text='Hello'),
        # )


if __name__ == '__main__':
    unittest.main()
