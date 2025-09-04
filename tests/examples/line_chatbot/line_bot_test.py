from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from examples.line_chatbot.line_bot import app
from examples.line_chatbot.line_bot import handler
from examples.line_chatbot.line_bot import messaging_api


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

        # Mock the LINE Messaging API v3 reply_message method
        self.mock_line_bot_api = patch.object(
            messaging_api, 'reply_message',
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
                    'message': {'type': 'text', 'text': 'Hello'},
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

        # Ensure the handler.handle method was called once with compact JSON
        expected_body = json.dumps(fake_body, separators=(',', ':'))
        self.mock_handler_handle.assert_called_once_with(
            expected_body, fake_signature,
        )

    def test_callback_missing_signature(self) -> None:
        """Should return 400 when signature is missing."""
        fake_body: dict = {'events': []}
        response = self.client.post(
            '/webhook', json=fake_body, headers={},
        )
        self.assertEqual(response.status_code, 400)

    def test_callback_invalid_signature(self) -> None:
        """Should return 400 when signature is invalid."""
        from linebot.v3.webhook import InvalidSignatureError

        fake_signature: str = 'fake_signature'
        fake_body: dict = {'events': []}

        self.mock_handler_handle.side_effect = InvalidSignatureError
        response = self.client.post(
            '/webhook', json=fake_body,
            headers={'X-Line-Signature': fake_signature},
        )
        self.assertEqual(response.status_code, 400)

    def test_callback_internal_server_error(self) -> None:
        """Should return 500 on unexpected server errors."""
        fake_signature: str = 'fake_signature'
        fake_body: dict = {'events': []}

        self.mock_handler_handle.side_effect = Exception('Some error')
        response = self.client.post(
            '/webhook',
            json=fake_body,
            headers={'X-Line-Signature': fake_signature},
        )
        self.assertEqual(response.status_code, 500)

    def test_handle_text_message(self) -> None:
        """Text message should trigger a reply with the same text echoed."""
        event = SimpleNamespace(
            reply_token='dummy_token',
            message=SimpleNamespace(text='Hello, World!'),
        )
        from examples.line_chatbot.line_bot import handle_text_message
        handle_text_message(event)

        self.mock_line_bot_api.assert_called_once()
        args, kwargs = self.mock_line_bot_api.call_args
        reply_req = args[0]
        self.assertEqual(reply_req.reply_token, 'dummy_token')
        self.assertEqual(len(reply_req.messages), 1)
        self.assertEqual(reply_req.messages[0].text, '您发送的消息是: Hello, World!')

    def test_handle_text_message_empty_early_return(self) -> None:
        """
        Empty or whitespace-only message should early return without reply.
        """
        event = SimpleNamespace(
            reply_token='dummy_token',
            message=SimpleNamespace(text='   \t'),
        )
        from examples.line_chatbot.line_bot import handle_text_message
        handle_text_message(event)
        self.mock_line_bot_api.assert_not_called()

    def test_handle_text_message_linebot_api_error(self) -> None:
        """ApiException should be handled and not crash the handler."""
        from linebot.v3.messaging.exceptions import ApiException
        event = SimpleNamespace(
            reply_token='dummy_token',
            message=SimpleNamespace(text='boom'),
        )
        from examples.line_chatbot.line_bot import handle_text_message

        class _FakeApiException(ApiException):  # type: ignore[misc]
            def __init__(self) -> None:
                pass

            def __str__(self) -> str:
                return 'fake-api-exception'

        self.mock_line_bot_api.side_effect = _FakeApiException()
        handle_text_message(event)
        self.mock_line_bot_api.assert_called_once()

    def test_handle_text_message_generic_exception(self) -> None:
        """Generic exceptions should be caught and logged without raising."""
        event = SimpleNamespace(
            reply_token='dummy_token',
            message=SimpleNamespace(text='crash'),
        )
        from examples.line_chatbot.line_bot import handle_text_message
        self.mock_line_bot_api.side_effect = Exception('boom')
        handle_text_message(event)
        self.mock_line_bot_api.assert_called_once()


if __name__ == '__main__':
    unittest.main()

"""
pytest \
    --cov=examples.line_chatbot.line_bot \
    --cov-report=term-missing \
    tests/examples/line_chatbot/line_bot_test.py
"""
