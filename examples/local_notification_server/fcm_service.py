from __future__ import annotations

import logging

import firebase_admin
from firebase_admin import credentials
from firebase_admin import messaging

logger = logging.getLogger(__name__)


def init_firebase_app(cred_path: str, project_id: str) -> None:
    """
    Initialises the Firebase Admin SDK application.

    Args:
        cred_path (str): Path to the Firebase service account key JSON file.
        project_id (str): The GCP/Firebase project ID.

    Raises:
        ValueError: If cred_path or project_id is empty.

    Returns:
        None

    Note:
        This function will only initialise the Firebase app if it has not
        already been initialised.
    """
    # Validate input parameters
    if not cred_path:
        raise ValueError('cred_path must be a non-empty string.')
    if not project_id:
        raise ValueError('project_id must be a non-empty string.')
    # Initialise only if not already done
    if not firebase_admin._apps:
        cred: credentials.Certificate = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(
            cred,
            {'projectId': project_id},
        )


ANDROID_CFG: messaging.AndroidConfig = messaging.AndroidConfig(
    priority='high',
    notification=messaging.AndroidNotification(
        # Must match the Flutter side
        channel_id='high_importance_channel',
        sound='default',
        default_vibrate_timings=True,
    ),
)

APNS_CFG: messaging.APNSConfig = messaging.APNSConfig(
    # General alert; for Critical use 5 + CriticalSound
    headers={'apns-priority': '10'},
    payload=messaging.APNSPayload(
        aps=messaging.Aps(
            # iOS plays default sound
            sound='default',
            badge=1,
        ),
    ),
)


async def send_fcm_notification_service(
    device_tokens: list[str],
    title: str,
    body: str,
    image_path: str | None = None,
    data: dict[str, str] | None = None,
) -> bool:
    """
    Sends FCM notifications to a list of device tokens.

    Args:
        device_tokens (list[str]):
            A list of device tokens to which the notification will be sent.
            These tokens should be valid FCM device tokens.
        title (str):
            The title of the notification.
        body (str):
            The body content of the notification.
        image_path (str | None, optional):
            Optional path to an image to include with the notification.
            This can be a URL or a local file path. Defaults to None.
        data (dict[str, str] | None, optional):
            Optional additional data to include with the notification.
            This can be used for custom payloads or extra information.
            Defaults to None.

    Returns:
        bool: True if all notifications were sent successfully; otherwise,
        False.

    Raises:
        None explicitly. Any exceptions during sending will be caught,
        logged, and the function will return False.

    Notes:
        This function uses the Firebase Admin SDK to send notifications to
        multiple devices. Android and iOS configurations are set for high
        priority and default sounds.
    """
    # Return early if no device tokens are provided
    if not device_tokens:
        logger.error('No device tokens provided.')
        return False

    # Construct FCM messages for each device token
    messages: list[messaging.Message] = []
    for token in device_tokens:
        # Create a message for each device
        msg: messaging.Message = messaging.Message(
            token=token,
            notification=messaging.Notification(
                title=title,
                body=body,
                image=image_path,
            ),
            data=data or {},
            android=ANDROID_CFG,
            apns=APNS_CFG,
        )
        messages.append(msg)

    try:
        # Send all messages using Firebase Admin SDK
        response: messaging.BatchResponse = messaging.send_each(messages)
        for idx, res in enumerate(response.responses):
            if not res.success:
                logger.error(
                    'Failed to send message to token %s: %s',
                    messages[idx].token,
                    res.exception,
                )
        # Return True only if all messages succeeded
        return response.failure_count == 0
    except Exception as exc:
        logger.error('FCM sending failed: %s', exc)
        return False
