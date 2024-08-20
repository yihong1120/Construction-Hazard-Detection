from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from linebot.exceptions import InvalidSignatureError
from linebot.exceptions import LineBotApiError
from linebot.models import MessageEvent
from linebot.models import TextMessage
from linebot.models import TextSendMessage

from examples.line_chatbot.line_bot import app
from examples.line_chatbot.line_bot import handle_text_message


class TestLineChatbot(unittest.TestCase):
    """
    Test suite for the LINE chatbot functionalities.
    """

    def setUp(self) -> None:
        """
        Set up the test client and mock the LineBotApi and WebhookHandler.
        """
        self.client = app.test_client()
        self.app = app
        self.app.testing = True

        # Patch LineBotApi and WebhookHandler
        self.line_bot_api_patcher = patch(
            'examples.line_chatbot.line_bot.LineBotApi',
        )
        self.handler_patcher = patch(
            'examples.line_chatbot.line_bot.WebhookHandler',
        )

        # Start the patching
        self.mock_line_bot_api = self.line_bot_api_patcher.start()
        self.mock_handler = self.handler_patcher.start()

        # Create mock instances
        self.mock_line_bot_api_instance = MagicMock()
        self.mock_handler_instance = MagicMock()

        # Set the return value for the patches
        self.mock_line_bot_api.return_value = self.mock_line_bot_api_instance
        self.mock_handler.return_value = self.mock_handler_instance

    def tearDown(self) -> None:
        """
        Stop the patching after each test.
        """
        self.line_bot_api_patcher.stop()
        self.handler_patcher.stop()

    def test_callback_valid_signature(self) -> None:
        """
        Test the callback function with a valid signature.
        """
        headers = {'X-Line-Signature': 'valid_signature'}
        data = 'body'

        # Simulate a successful handler.handle execution
        self.mock_handler_instance.handle.return_value = None
        response = self.client.post(
            '/webhook', headers=headers, data=data,
        )

        # Check that handler.handle was called correctly
        self.mock_handler_instance.handle.assert_called_once_with(
            'body', 'valid_signature',
        )
        self.assertEqual(response.status_code, 200)

    def test_callback_invalid_signature(self) -> None:
        """
        Test the callback function with an invalid signature.
        """
        # Simulate InvalidSignatureError being raised
        self.mock_handler_instance.handle.side_effect = InvalidSignatureError()

        headers = {'X-Line-Signature': 'invalid_signature'}
        data = 'body'

        response = self.client.post(
            '/webhook', headers=headers, data=data,
        )

        # Ensure that a 400 status is returned
        # when InvalidSignatureError is raised
        self.assertEqual(response.status_code, 400)

    def test_handle_text_message(self) -> None:
        """
        Test handling a valid text message event.
        """
        event = MessageEvent(
            reply_token='dummy_token',
            message=TextMessage(text='Hello'),
            source=None, timestamp=None, mode=None,
        )

        with self.app.test_request_context('/webhook', method='POST'):
            # Call the message handling function
            handle_text_message(event)

            # Verify that reply_message was called correctly
            self.mock_line_bot_api_instance.reply_message. \
                assert_called_once_with(
                    'dummy_token',
                    TextSendMessage(text='您发送的消息是: Hello'),
                )

    def test_handle_text_message_with_empty_message(self) -> None:
        """
        Test handling a text message event with an empty message.
        """
        event = MessageEvent(
            reply_token='dummy_token',
            message=TextMessage(text='   '),  # White space characters
            source=None, timestamp=None, mode=None,
        )

        with self.app.test_request_context('/webhook', method='POST'):
            # Call the message handling function
            handle_text_message(event)

            # Ensure reply_message is not called when the message is empty
            self.mock_line_bot_api_instance.reply_message.assert_not_called()

    def test_handle_text_message_with_api_error(self) -> None:
        """
        Test handling a text message event where an API error occurs.
        """
        event = MessageEvent(
            reply_token='dummy_token',
            message=TextMessage(text='Hello'),
            source=None, timestamp=None, mode=None,
        )

        # Simulate LineBotApiError being raised during reply_message call
        self.mock_line_bot_api_instance.reply_message.side_effect = (
            LineBotApiError(
                '401', 'Authentication failed',
            )
        )

        with self.app.test_request_context('/webhook', method='POST'):
            # Call the message handling function
            handle_text_message(event)

            # Verify that reply_message was called despite the exception
            self.mock_line_bot_api_instance.reply_message. \
                assert_called_once_with(
                    'dummy_token',
                    TextSendMessage(text='您发送的消息是: Hello'),
                )


if __name__ == '__main__':
    unittest.main()
