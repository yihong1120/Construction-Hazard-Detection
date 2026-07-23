from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from src.warning_types import MutableWarnings

NotificationType = Literal[
    'signature',
    'violation',
    'document',
    'site_alert',
    'system',
]
NotificationStatus = Literal['unread', 'read', 'all']
NotificationPlatform = Literal['android', 'ios', 'web', 'unknown']
NotificationPermissionStatus = Literal[
    'granted',
    'denied',
    'default',
    'unknown',
]


class TokenRequest(BaseModel):
    """Request payload for registering or deleting an FCM device token.

    Attributes:
        user_id: Unique user identifier.
        device_token: Device token used by Firebase Cloud Messaging.
        device_lang: Preferred device language. Required by `/store_token` and
            ignored by `/delete_token`.
    """

    user_id: int
    device_token: str
    device_lang: str | None = None
    platform: NotificationPlatform | None = None
    permission_status: NotificationPermissionStatus | None = None
    app_version: str | None = None
    web_vapid_key_available: bool | None = None
    web_service_worker_registered: bool | None = None


class TokenStoreResponse(BaseModel):
    """Response returned after registering an FCM token."""

    ok: bool
    updated: bool
    user_id: int
    device_lang: str
    registered_at: str
    last_seen_at: str


class DeviceTokenStatus(BaseModel):
    """Notification state for one registered device token."""

    token_hash: str
    platform: str = 'unknown'
    device_lang: str | None = None
    permission_status: str = 'unknown'
    registered_at: str | None = None
    last_seen_at: str | None = None
    last_success_at: str | None = None
    last_failure_at: str | None = None
    failure_reason: str | None = None
    is_active: bool = True
    web_vapid_key_available: bool | None = None
    web_service_worker_registered: bool | None = None


class DeviceStatusResponse(BaseModel):
    """Notification diagnostics for the current user."""

    user_id: int
    has_fcm_token: bool
    token_count: int
    devices: list[DeviceTokenStatus]


class TestNotificationResponse(BaseModel):
    """Response from sending a test push notification."""

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
        notification_type: Optional notification-center category. If omitted,
            violation notifications default to `violation` and site warnings
            default to `site_alert`.
        title: Optional notification-center title override.
        deep_link: Optional app route. This is also copied into FCM data.
        metadata: Optional structured context stored with the notification.
    """

    model_config = ConfigDict(populate_by_name=True)

    site: str
    stream_name: str
    body: MutableWarnings
    image_path: str | None = None
    violation_id: int | None = None
    notification_type: NotificationType | None = Field(
        default=None,
        alias='type',
    )
    title: str | None = None
    deep_link: str | None = None
    metadata: dict[str, object] | None = None


class NotificationOut(BaseModel):
    """Single in-app notification returned to the mobile/web client."""

    id: int
    type: NotificationType
    title: str
    body: str
    deep_link: str | None = None
    is_read: bool
    created_at: datetime
    metadata: dict[str, object] = Field(default_factory=dict)


class NotificationList(BaseModel):
    """Paginated notification-center response."""

    total: int
    page: int
    page_size: int
    items: list[NotificationOut]


class NotificationUnreadCount(BaseModel):
    """Unread notification badge response."""

    unread_count: int


class NotificationBulkReadResponse(BaseModel):
    """Result of marking all notifications as read."""

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

    preferences: list[SiteNotificationPreferenceIn]


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
