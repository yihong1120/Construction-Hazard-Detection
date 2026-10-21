from __future__ import annotations

from datetime import datetime
from datetime import timezone


def is_expired(expire_date_str: str | None) -> bool:
    """Return whether an ISO 8601 expiry timestamp has elapsed."""
    if not expire_date_str:
        return False
    try:
        expiry = datetime.fromisoformat(expire_date_str)
    except ValueError:
        return False
    if expiry.tzinfo is None:
        # Naive timestamps keep the local-time interpretation.
        return datetime.now() > expiry
    # Offset-aware timestamps denote an absolute instant, so they are
    # compared against the current UTC time.
    return datetime.now(timezone.utc) > expiry


def should_notify(
    timestamp: int,
    last_notification_time: int,
    cooldown_period: int = 300,
) -> bool:
    """Return whether the notification cooldown has elapsed."""
    return timestamp - last_notification_time >= cooldown_period
