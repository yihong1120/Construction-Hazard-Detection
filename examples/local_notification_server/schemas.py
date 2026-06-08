from __future__ import annotations

from pydantic import BaseModel

from src.warning_types import MutableWarnings


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


class SiteNotifyRequest(BaseModel):
    """Request payload for a site-based safety notification.

    Attributes:
        site: Site name or identifier.
        stream_name: Camera stream name associated with the event.
        body: Warning payload grouped by warning key.
        image_path: Optional URL or path to the violation image.
        violation_id: Optional unique violation identifier.
    """

    site: str
    stream_name: str
    body: MutableWarnings
    image_path: str | None = None
    violation_id: int | None = None


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
