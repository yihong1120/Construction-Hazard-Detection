from __future__ import annotations

import logging

from flask import abort
from flask import Flask
from flask import request
from linebot import LineBotApi
from linebot import WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.exceptions import LineBotApiError
from linebot.models import MessageEvent
from linebot.models import TextMessage
from linebot.models import TextSendMessage

app: Flask = Flask(__name__)

# Initialise the LINE Bot API and WebhookHandler with access token and secret
line_bot_api: LineBotApi = LineBotApi('YOUR_LINE_CHANNEL_ACCESS_TOKEN')
handler: WebhookHandler = WebhookHandler('YOUR_LINE_CHANNEL_SECRET')


@app.route('/webhook', methods=['POST'])
def callback() -> str:
    """
    Handle the incoming webhook requests from LINE.

    Returns:
        str: The response msg to indicate successful handling of the request.
    """
    signature: str = request.headers.get('X-Line-Signature', '')
    body: str = request.get_data(as_text=True)

    # Check if signature or body is missing
    if not signature or not body:
        logging.warning('Received invalid request')
        abort(400)  # Bad Request

    try:
        # Handle the request with the provided body and signature
        handler.handle(body, signature)
    except InvalidSignatureError:
        # Log and abort if the signature is invalid
        logging.error('Invalid signature error')
        abort(400)  # Bad Request
    except Exception as e:
        # Log and abort if any other unexpected error occurs
        logging.error(f"Unexpected error: {e}")
        abort(500)  # Internal Server Error

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event: MessageEvent) -> None:
    """
    Handle the incoming text message event from the user.

    Args:
        event (MessageEvent): Event object containing details of the message.

    Returns:
        None
    """
    user_message: str = event.message.text

    # Check if the user's message is empty or contains only whitespace
    if not user_message.strip():
        logging.warning('Received empty user message')
        return

    try:
        # Generate a response based on the user's message
        assistant_response: str = f"您发送的消息是: {user_message}"

        # Send the response back to the user via LINE
        line_bot_api.reply_message(
            event.reply_token, TextSendMessage(text=assistant_response),
        )
    except LineBotApiError as e:
        # Log the error if the LINE Bot API fails
        logging.error(f"Error responding to message: {e}")
    except Exception as e:
        # Log any other unexpected errors
        logging.error(f"Unexpected error: {e}")


if __name__ == '__main__':
    # Run the Flask application on the specified host and port
    app.run(host='0.0.0.0', port=8000)
