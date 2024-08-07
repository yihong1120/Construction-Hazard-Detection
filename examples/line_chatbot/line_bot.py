from __future__ import annotations

import os
from typing import Any

from flask import abort
from flask import Flask
from flask import request
from linebot import LineBotApi
from linebot import WebhookHandler
from linebot.exceptions import (
    InvalidSignatureError,
)
from linebot.models import MessageEvent
from linebot.models import TextMessage
from linebot.models import TextSendMessage

app = Flask(__name__)

# Get the LINE Channel Access Token and Secret from environment variables
LINE_CHANNEL_ACCESS_TOKEN: str = os.getenv(
    'LINE_CHANNEL_ACCESS_TOKEN',
    'YOUR_LINE_CHANNEL_ACCESS_TOKEN',
)
LINE_CHANNEL_SECRET: str = os.getenv(
    'LINE_CHANNEL_SECRET',
    'YOUR_LINE_CHANNEL_SECRET',
)

line_bot_api: LineBotApi = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler: WebhookHandler = WebhookHandler(LINE_CHANNEL_SECRET)


@app.route('/callback', methods=['POST'])
def callback() -> Any:
    """
    Handles the callback from LINE Webhook.

    This function processes incoming webhook requests from LINE. It validates
    the request signature to ensure the request is from LINE. If the signature
    is valid, it handles the event using the handler object.

    Returns:
        Returns 'OK' with 200 status if successful, else aborts with 400.

    Raises:
        InvalidSignatureError: If the request signature is invalid.
    """
    # Obtain the X-Line-Signature header value
    signature: str = request.headers['X-Line-Signature']

    # Get the request body as text
    body: str = request.get_data(as_text=True)
    app.logger.info('Request body: ' + body)

    # Handle the webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event: MessageEvent) -> None:
    """
    Responds to text messages from users.

    This function is called when a text message event is received. It sends a
    text message back to the user with the same text they sent.

    Args:
        event (MessageEvent): The event object for the received message.
    """
    # Reply to the user with the same text message
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=event.message.text),
    )


if __name__ == '__main__':
    # Start the Flask application
    app.run()
