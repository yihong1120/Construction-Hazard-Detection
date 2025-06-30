from __future__ import annotations

import logging

import firebase_admin
from firebase_admin import credentials
from firebase_admin import messaging

logger = logging.getLogger(__name__)


def init_firebase_app() -> None:
    """
    Initialise the Firebase Admin SDK application.
    """
    if not firebase_admin._apps:
        cred_path = (
            'examples/local_notification_server/'
            'construction-hazard-detection-firebase-adminsdk.json'
        )
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(
            cred,
            {'projectId': 'construction-harzard-detection'},
        )


async def send_fcm_notification_service(
    device_tokens: list[str],
    title: str,
    body: str,
    image_path: str | None = None,
    data: dict[str, str] | None = None,
) -> bool:
    """
    Send FCM notifications to a list of device tokens.

    Args:
        device_tokens (List[str]):
            A list of device tokens to which the notification will be sent.
            These tokens should be valid FCM device tokens.
        title (str):
            The title of the notification.
        body (str):
            The body content of the notification.
        image_path (Optional[str], optional):
            Optional path to an image to include with the notification.
            This can be a URL or a local file path.
            Defaults to None.
        data (Optional[Dict[str, str]], optional):
            Optional additional data to include with the notification.
            This can be used for custom payloads or extra information.
            Defaults to None.

    Returns:
        bool:
            `True` if all notifications were sent successfully;
            otherwise `False`.

    Raises:
        None explicitly, but any exceptions during sending will be caught,
        logged, and the function will return `False`.
    """
    if not device_tokens:
        logger.error('No device tokens provided.')
        return False

    messages = [
        messaging.Message(
            token=token,
            notification=messaging.Notification(
                title=title,
                body=body,
                image=image_path,
            ),
            data=data or {},
        )
        for token in device_tokens
    ]

    try:
        response = messaging.send_each(messages)
        for idx, res in enumerate(response.responses):
            if not res.success:
                logger.error(
                    'Failed to send message to token %s: %s',
                    messages[idx].token,
                    res.exception,
                )
        return response.failure_count == 0
    except Exception as exc:
        logger.error('FCM sending failed: %s', exc)
        return False
