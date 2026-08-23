from __future__ import annotations

from datetime import datetime


def is_expired(expire_date_str: str | None) -> bool:
    """Return whether an ISO 8601 expiry timestamp has elapsed."""
    if not expire_date_str:
        return False
    try:
        return datetime.now() > datetime.fromisoformat(expire_date_str)
    except ValueError:
        return False


def should_notify(
    timestamp: int,
    last_notification_time: int,
    cooldown_period: int = 300,
) -> bool:
    """Return whether the notification cooldown has elapsed."""
    return timestamp - last_notification_time >= cooldown_period
