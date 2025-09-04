from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import PlainTextResponse
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import ApiClient
from linebot.v3.messaging import Configuration
from linebot.v3.messaging import MessagingApi
from linebot.v3.messaging import ReplyMessageRequest
from linebot.v3.messaging import TextMessage as V3TextMessage
from linebot.v3.messaging.exceptions import ApiException
from linebot.v3.webhook import WebhookHandler
from linebot.v3.webhooks import MessageEvent
from linebot.v3.webhooks import TextMessageContent

from examples.auth.lifespan import global_lifespan

# Create a FastAPI application instance with shared global lifespan
# This ensures Redis and the rate-limit Lua script are initialised/preloaded.
app = FastAPI(lifespan=global_lifespan)

# Initialise the LINE Messaging API (v3) and WebhookHandler
# with access token and secret
configuration: Configuration = Configuration(
    access_token='YOUR_LINE_CHANNEL_ACCESS_TOKEN',
)
api_client: ApiClient = ApiClient(configuration)
messaging_api: MessagingApi = MessagingApi(api_client)
handler: WebhookHandler = WebhookHandler('YOUR_LINE_CHANNEL_SECRET')


@app.post('/webhook', response_class=PlainTextResponse)
async def callback(request: Request) -> str:
    """
    Handle incoming webhook requests from LINE.

    Args:
        request (Request): The HTTP request object
            containing the webhook data.

    Returns:
        str: A plain text response
            indicating successful handling of the request.

    Raises:
        HTTPException: If the request is invalid
            or an error occurs during processing.
    """
    # Extract the signature and body from the request
    signature: str = request.headers.get('X-Line-Signature', '')
    body: bytes = await request.body()
    body_text: str = body.decode('utf-8') if body else ''

    # Check if signature or body is missing
    if not signature or not body_text:
        logging.warning(
            'Received invalid request: signature or body is missing',
        )
        raise HTTPException(status_code=400, detail='Bad Request')

    try:
        # Process the incoming message using the LINE handler
        handler.handle(body_text, signature)
    except InvalidSignatureError:
        logging.error('Invalid signature error')
        raise HTTPException(status_code=400, detail='Invalid Signature Error')
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail='Internal Server Error')

    # Return success response
    return 'OK'


@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event: MessageEvent) -> None:
    """
    Handle incoming text messages from users.

    Args:
        event (MessageEvent): The LINE message event
            containing the user's message.

    Raises:
        LineBotApiError: If an error occurs
            while sending a reply message.
        Exception: If an unexpected error occurs during processing.

    Returns:
        None
    """
    # Extract the user's message
    user_message: str = event.message.text

    # Check if the message is empty or contains only whitespace
    if not user_message.strip():
        logging.warning('Received empty user message')
        return

    try:
        # Generate a response based on the user's message
        assistant_response: str = f"您发送的消息是: {user_message}"

        # Send the response back to the user via LINE (v3 API)
        messaging_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[V3TextMessage(text=assistant_response)],
            ),
        )
    except ApiException as e:
        logging.error(f"Error responding to message: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
