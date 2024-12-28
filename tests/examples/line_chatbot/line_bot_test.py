from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from examples.line_chatbot.line_bot import app
from examples.line_chatbot.line_bot import handler
from examples.line_chatbot.line_bot import line_bot_api


class TestLineBot(unittest.TestCase):
    """
    Unit tests for the LINE Bot API using FastAPI.
    This class validates webhook handling and message processing behaviours.
    """

    def setUp(self) -> None:
        """
        Set up the test environment before each test.
        Initialise a TestClient for the FastAPI app and mock necessary objects.
        """
        # FastAPI test client
        self.client: TestClient = TestClient(app)

        # Mock the LINE Bot API reply_message method
        self.mock_line_bot_api = patch.object(
            line_bot_api, 'reply_message',
        ).start()

        # Mock the LINE Bot WebhookHandler handle method
        self.mock_handler_handle = patch.object(handler, 'handle').start()

    def tearDown(self) -> None:
        """
        Clean up resources and stop all patches after each test.
        """
        patch.stopall()

    def test_callback_ok(self) -> None:
        """
        Test that the webhook endpoint returns 'OK' when provided with
        a valid signature and body payload.
        """
        # Fake signature and body
        fake_signature: str = 'fake_signature'
        fake_body: dict = {
            'events': [
                {
                    'replyToken': 'reply_token',
                    'type': 'message',
                    'message': {
                        'type': 'text',
                        'text': 'Hello',
                    },
                },
            ],
        }

        # Mock handler.handle to simulate no error
        self.mock_handler_handle.return_value = None

        # Send a POST request to the webhook endpoint
        response = self.client.post(
            '/webhook',
            json=fake_body,
            headers={
                'X-Line-Signature': fake_signature,
                'Content-Type': 'application/json',
            },
        )

        # Validate response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, 'OK')

        # Ensure the handler.handle method
        # was called once with correct arguments
        expected_body = json.dumps(fake_body, separators=(',', ':'))
        self.mock_handler_handle.assert_called_once_with(
            expected_body, fake_signature,
        )

    def test_callback_missing_signature(self) -> None:
        """
        Test that the webhook endpoint returns 400 if no signature is provided.
        """
        # Fake body without signature
        fake_body: dict = {'events': []}

        # Send a POST request without a signature header
        response = self.client.post(
            '/webhook',
            json=fake_body,
            headers={},  # Missing X-Line-Signature
        )

        # Validate response
        self.assertEqual(response.status_code, 400)

    def test_callback_invalid_signature(self) -> None:
        """
        Test that the webhook endpoint returns 400 if the signature is invalid.
        """
        from linebot.exceptions import InvalidSignatureError

        fake_signature: str = 'fake_signature'
        fake_body: dict = {'events': []}

        # Simulate an InvalidSignatureError being raised
        self.mock_handler_handle.side_effect = InvalidSignatureError

        # Send a POST request with an invalid signature
        response = self.client.post(
            '/webhook',
            json=fake_body,
            headers={'X-Line-Signature': fake_signature},
        )

        # Validate response
        self.assertEqual(response.status_code, 400)

    def test_callback_internal_server_error(self) -> None:
        """
        Test that the webhook endpoint returns 500
        for unexpected server errors.
        """
        fake_signature: str = 'fake_signature'
        fake_body: dict = {'events': []}

        # Simulate a generic exception being raised
        self.mock_handler_handle.side_effect = Exception('Some error')

        # Send a POST request
        response = self.client.post(
            '/webhook',
            json=fake_body,
            headers={'X-Line-Signature': fake_signature},
        )

        # Validate response
        self.assertEqual(response.status_code, 500)

    def test_handle_text_message(self) -> None:
        """
        Test that the handle_text_message function processes a text message
        event correctly and sends an appropriate reply.
        """
        # Create a fake TextMessage event
        from linebot.models import MessageEvent, TextMessage
        event: MessageEvent = MessageEvent(
            reply_token='dummy_token',
            message=TextMessage(text='Hello, World!'),
        )

        # Import the function to test
        from examples.line_chatbot.line_bot import handle_text_message

        # Call the function with the fake event
        handle_text_message(event)

        # Validate that line_bot_api.reply_message was called once
        self.mock_line_bot_api.assert_called_once()

        # Extract the arguments passed to reply_message
        args, kwargs = self.mock_line_bot_api.call_args

        # Verify the reply token
        self.assertEqual(args[0], 'dummy_token')

        # Verify the reply message content
        text_send_message = args[1]
        self.assertEqual(text_send_message.text, '您发送的消息是: Hello, World!')


if __name__ == '__main__':
    unittest.main()
