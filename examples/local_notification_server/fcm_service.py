from __future__ import annotations

import logging
from dataclasses import dataclass

import firebase_admin
from firebase_admin import credentials
from firebase_admin import messaging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FcmSendResult:
    """Result for one FCM batch send."""

    success_count: int
    failure_count: int
    invalid_tokens: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.success_count > 0 and self.failure_count == 0


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
) -> FcmSendResult:
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
        FcmSendResult:
            Batch result. It is truthy only when all notifications in the
            batch were sent successfully.

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
        return FcmSendResult(success_count=0, failure_count=0)

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
        invalid_tokens: list[str] = []
        for idx, res in enumerate(response.responses):
            if not res.success:
                token = messages[idx].token
                logger.error(
                    'Failed to send message to token %s: %s',
                    token,
                    res.exception,
                )
                if _is_invalid_registration_token_error(res.exception):
                    invalid_tokens.append(token)
        return FcmSendResult(
            success_count=len(messages) - response.failure_count,
            failure_count=response.failure_count,
            invalid_tokens=tuple(invalid_tokens),
        )
    except Exception as exc:
        logger.error('FCM sending failed: %s', exc)
        return FcmSendResult(
            success_count=0,
            failure_count=len(device_tokens),
        )


def _is_invalid_registration_token_error(exc: object) -> bool:
    """Return True only for FCM errors that mark a token as unusable."""
    if isinstance(exc, messaging.UnregisteredError):
        return True
    if isinstance(exc, messaging.SenderIdMismatchError):
        return True

    code = str(getattr(exc, 'code', '') or '').lower()
    if code in {
        'registration-token-not-registered',
        'sender-id-mismatch',
        'unregistered',
        'not-found',
    }:
        return True

    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            'requested entity was not found',
            'registration token is not registered',
            'not registered',
            'sender id mismatch',
            'sender-id mismatch',
        )
    )
