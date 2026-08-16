from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

from examples.local_notification_server.lang_config import LANGUAGES
from examples.local_notification_server.lang_config import NotificationLanguage
from src.warning_types import MutableWarnings

NotificationType = Literal[
    'signature',
    'violation',
    'document',
    'site_alert',
    'system',
]
NotificationStatus = Literal['unread', 'read', 'all']
ClientPlatform = Literal['android', 'ios', 'web']
NotificationPlatform = Literal['android', 'ios', 'web', 'unknown']
NotificationPermissionStatus = Literal[
    'granted',
    'denied',
    'default',
    'unknown',
]


class DeviceRegistrationRequest(BaseModel):
    """Request payload for registering an FCM device token.

    Attributes:
        device_token: Device token used by Firebase Cloud Messaging.
        device_lang: Canonical preferred device language.
        platform: Client platform that registered the token.
    """

    model_config = ConfigDict(extra='forbid')

    device_token: str
    device_lang: NotificationLanguage
    platform: ClientPlatform


class DeviceUnregistrationRequest(BaseModel):
    """Request payload for removing the current user's FCM device token.

    Attributes:
        device_token: Device token removed from Firebase Cloud Messaging.
    """

    model_config = ConfigDict(extra='forbid')

    device_token: str


class DeviceRegistrationResponse(BaseModel):
    """Response returned after registering an FCM token.

    Attributes:
        ok: Whether registration completed successfully.
        updated: Whether the registration was created or refreshed.
        user_id: Identifier of the token owner.
        device_lang: Canonical language stored for the device.
        registered_at: Original registration timestamp in UTC.
        last_seen_at: Most recent registration timestamp in UTC.
    """

    ok: bool
    updated: bool
    user_id: int
    device_lang: str
    registered_at: str
    last_seen_at: str


class DeviceTokenStatus(BaseModel):
    """Notification state for one registered device token.

    Attributes:
        token_hash: Non-sensitive identifier for the registered token.
        platform: Platform recorded for the token, or ``unknown``.
        device_lang: Canonical language requested by the device.
        permission_status: Permission state recorded for the token, or
            ``unknown``.
        registered_at: Original registration timestamp in UTC.
        last_seen_at: Most recent registration timestamp in UTC.
        last_success_at: Most recent successful delivery timestamp, if any.
        last_failure_at: Most recent failed delivery timestamp, if any.
        failure_reason: Most recent delivery failure reason, if any.
        is_active: Whether the token may receive notification sends.
    """

    token_hash: str
    platform: NotificationPlatform
    device_lang: NotificationLanguage
    permission_status: NotificationPermissionStatus
    registered_at: str
    last_seen_at: str
    last_success_at: str | None = None
    last_failure_at: str | None = None
    failure_reason: str | None = None
    is_active: bool = True


class DeviceStatusResponse(BaseModel):
    """Notification diagnostics for the current user.

    Attributes:
        user_id: Identifier of the authenticated user.
        has_fcm_token: Whether at least one active token is registered.
        token_count: Number of active registered tokens.
        devices: Per-device diagnostic entries.
    """

    user_id: int
    has_fcm_token: bool
    token_count: int
    devices: list[DeviceTokenStatus]


class TestNotificationResponse(BaseModel):
    """Response from sending a test push notification.

    Attributes:
        success: Whether every attempted token was sent successfully.
        message: User-facing result summary.
        attempted_tokens: Number of target tokens.
        success_count: Number of successful token sends.
        failure_count: Number of failed token sends.
        invalid_tokens: Number of invalid tokens disabled after the send.
    """

    success: bool
    message: str
    attempted_tokens: int
    success_count: int = 0
    failure_count: int = 0
    invalid_tokens: int = 0


class SiteNotifyRequest(BaseModel):
    """Request payload for a site-based safety notification.

    Attributes:
        site: Site name or identifier.
        stream_name: Camera stream name associated with the event.
        body: Warning payload grouped by warning key.
        image_path: Optional URL or path to the violation image.
        violation_id: Optional unique violation identifier.
        notification_type: Notification-centre category.
        title: Notification-centre display title.
        deep_link: App route copied into FCM data.
        metadata: Structured context stored with the notification.
    """

    # Clients may use ``type`` while Python retains the unambiguous field name.
    model_config = ConfigDict(populate_by_name=True)

    site: str
    stream_name: str
    body: MutableWarnings = Field(min_length=1)
    image_path: str | None = None
    violation_id: int | None = None
    notification_type: NotificationType = Field(alias='type')
    title: str
    deep_link: str
    metadata: dict[str, object]

    @field_validator('body')
    @classmethod
    def validate_warning_keys(cls, body: MutableWarnings) -> MutableWarnings:
        """Require warning keys that have translations in every locale.

        Args:
            body: Non-empty warning payload received from the event producer.

        Returns:
            The validated warning payload.

        Raises:
            ValueError: If a warning key has no notification translation.
        """
        # A single canonical catalogue prevents partially translatable events.
        unsupported_keys = set(body).difference(LANGUAGES['en-GB'])
        if unsupported_keys:
            keys = ', '.join(sorted(unsupported_keys))
            raise ValueError(f'Unsupported notification warning keys: {keys}')
        return body


class NotificationOut(BaseModel):
    """Single in-app notification returned to a mobile or web client.

    Attributes:
        id: Notification primary key.
        type: Notification category.
        title: Display title.
        body: Display body.
        deep_link: Optional client route associated with the notification.
        is_read: Whether the recipient has read the notification.
        created_at: Creation timestamp.
        metadata: Structured application context for the notification.
    """

    id: int
    type: NotificationType
    title: str
    body: str
    deep_link: str | None = None
    is_read: bool
    created_at: datetime
    metadata: dict[str, object]


class NotificationList(BaseModel):
    """Paginated notification-centre response.

    Attributes:
        total: Number of records matching the filters.
        page: One-based result page.
        page_size: Maximum number of records in the page.
        items: Notification records in descending creation order.
    """

    total: int
    page: int
    page_size: int
    items: list[NotificationOut]


class NotificationUnreadCount(BaseModel):
    """Unread notification badge response.

    Attributes:
        unread_count: Number of unread notifications for the user.
    """

    unread_count: int


class NotificationBulkReadResponse(BaseModel):
    """Result of marking all notifications as read.

    Attributes:
        updated_count: Number of records changed from unread to read.
    """

    updated_count: int


class SiteNotificationPreferenceIn(BaseModel):
    """Requested notification status for one site.

    Attributes:
        site_id: Site identifier.
        is_enabled: Whether notifications should be enabled for this user and
            site.
    """

    site_id: int
    is_enabled: bool


class SiteNotificationPreferenceUpdateRequest(BaseModel):
    """Full notification preference payload from the settings page.

    Attributes:
        preferences: Requested per-site notification preferences.
    """

    preferences: list[SiteNotificationPreferenceIn] = Field(min_length=1)


class SiteNotificationPreferenceOut(BaseModel):
    """Notification subscription status for one site.

    Attributes:
        site_id: Site identifier.
        site_name: Human-readable site name.
        group_name: Optional group name associated with the site.
        is_enabled: Whether notifications are enabled for the current user.
    """

    site_id: int
    site_name: str
    group_name: str | None = None
    is_enabled: bool
