from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass

import firebase_admin  # type: ignore[import-untyped]
from firebase_admin import credentials
from firebase_admin import messaging

logger = logging.getLogger(__name__)


def _token_log_id(device_token: str) -> str:
    """Return a non-sensitive FCM token identifier for logs.

    Args:
        device_token: Raw FCM registration token.

    Returns:
        Short deterministic token hash suitable for diagnostic logs.
    """
    return hashlib.sha256(device_token.encode('utf-8')).hexdigest()[:12]


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

    """
    # Firebase keeps a process-wide app registry, so workers initialise once.
    if not firebase_admin._apps:
        cred: credentials.Certificate = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(
            cred,
            {'projectId': project_id},
        )


ANDROID_CFG: messaging.AndroidConfig = messaging.AndroidConfig(
    priority='high',
    notification=messaging.AndroidNotification(
        # This channel identifier must match the Flutter client configuration.
        channel_id='high_importance_channel',
        sound='default',
        default_vibrate_timings=True,
    ),
)

APNS_CFG: messaging.APNSConfig = messaging.APNSConfig(
    # Use a normal alert priority; critical alerts require a distinct payload.
    headers={'apns-priority': '10'},
    payload=messaging.APNSPayload(
        aps=messaging.Aps(
            # iOS clients play their configured default notification sound.
            sound='default',
            badge=1,
        ),
    ),
)


WEBPUSH_CFG: messaging.WebpushConfig = messaging.WebpushConfig(
    headers={
        'Urgency': 'high',
        'TTL': '3600',
    },
)


async def send_fcm_notification_service(
    device_tokens: list[str],
    title: str,
    body: str,
    data: dict[str, str],
    image_path: str | None = None,
) -> FcmSendResult:
    """Send one FCM notification payload to a batch of device tokens.

    Args:
        device_tokens: FCM device tokens receiving the notification.
        title: Notification title.
        body: Notification body text.
        image_path: Optional image URL or path included in the notification.
        data: Custom FCM data payload.

    Returns:
        Batch send result. It is truthy only when all messages in the batch
        were sent successfully.
    """
    # Callers construct batches only from the canonical token cache.
    messages: list[messaging.Message] = []
    for token in device_tokens:
        # Firebase requires one Message object for every target token.
        msg: messaging.Message = messaging.Message(
            token=token,
            notification=messaging.Notification(
                title=title,
                body=body,
                image=image_path,
            ),
            data=data,
            android=ANDROID_CFG,
            apns=APNS_CFG,
            webpush=WEBPUSH_CFG,
        )
        messages.append(msg)

    try:
        # The SDK response preserves message order for precise failure mapping.
        # firebase-admin exposes only a blocking transport.  The dispatcher
        # limits the number of in-flight batches, while this hand-off keeps
        # Redis reads, request cancellation, and other API work responsive.
        response: messaging.BatchResponse = await asyncio.to_thread(
            messaging.send_each,
            messages,
        )
        invalid_tokens: list[str] = []
        for idx, res in enumerate(response.responses):
            if not res.success:
                token = messages[idx].token
                logger.error(
                    'Failed to send message to token_hash=%s: %s',
                    _token_log_id(token),
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
    match exc:
        case messaging.UnregisteredError() | messaging.SenderIdMismatchError():
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
