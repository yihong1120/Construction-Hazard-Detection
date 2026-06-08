from __future__ import annotations

import logging
from dataclasses import dataclass

import firebase_admin
from firebase_admin import credentials
from firebase_admin import messaging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FcmSendResult:
    """Result for one FCM batch send.

    Attributes:
        success_count: Number of successfully delivered messages.
        failure_count: Number of failed message deliveries.
        invalid_tokens: Device tokens that Firebase reported as unusable.
    """

    success_count: int
    failure_count: int
    invalid_tokens: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        """Return whether every message in the batch was sent successfully.

        Returns:
            True when at least one message succeeded and none failed.
        """
        return self.success_count > 0 and self.failure_count == 0


def init_firebase_app(cred_path: str, project_id: str) -> None:
    """Initialise the Firebase Admin SDK application once.

    Args:
        cred_path: Path to the Firebase service account key JSON file.
        project_id: GCP/Firebase project ID.

    Raises:
        ValueError: Raised when `cred_path` or `project_id` is empty.
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
    """Send one FCM notification payload to a batch of device tokens.

    Args:
        device_tokens: FCM device tokens receiving the notification.
        title: Notification title.
        body: Notification body text.
        image_path: Optional image URL or path included in the notification.
        data: Optional custom FCM data payload.

    Returns:
        Batch send result. It is truthy only when all messages in the batch
        were sent successfully.
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
    """Return whether an FCM exception marks a token as unusable.

    Args:
        exc: Firebase exception object or compatible error.

    Returns:
        True when the token should be removed from Redis.
    """
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
