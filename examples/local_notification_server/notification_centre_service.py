from __future__ import annotations

import base64
import json
from datetime import datetime
from typing import Any
from typing import cast

from fastapi import HTTPException
from sqlalchemy import and_
from sqlalchemy import func
from sqlalchemy import or_
from sqlalchemy import select
from sqlalchemy import update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.models import Notification
from examples.auth.models import User
from examples.local_notification_server.schemas import DeviceStatusResponse
from examples.local_notification_server.schemas import DeviceTokenStatus
from examples.local_notification_server.schemas import (
    NotificationBulkReadResponse,
)
from examples.local_notification_server.schemas import NotificationList
from examples.local_notification_server.schemas import NotificationOut
from examples.local_notification_server.schemas import NotificationStatus
from examples.local_notification_server.schemas import NotificationType
from examples.local_notification_server.schemas import NotificationUnreadCount
from examples.local_notification_server.services import list_fcm_device_status


def _notification_to_out(notification: Notification) -> NotificationOut:
    """Convert a notification ORM entity into an API response model.

    Args:
        notification: Persisted notification entity.

    Returns:
        API-safe notification representation.
    """
    return NotificationOut(
        id=notification.id,
        type=cast(NotificationType, notification.type),
        title=notification.title,
        body=notification.body,
        deep_link=notification.deep_link,
        is_read=notification.is_read,
        created_at=notification.created_at,
        metadata=dict(notification.metadata_json),
    )


async def _get_owned_notification(
    notification_id: int,
    user_id: int,
    db: AsyncSession,
) -> Notification:
    """Load a notification that belongs to one user.

    Args:
        notification_id: Notification primary key.
        user_id: Required owner user ID.
        db: Database session used for the lookup.

    Returns:
        Owned notification entity.

    Raises:
        HTTPException: If no owned notification exists.
    """
    # Ownership is part of the query to avoid leaking another user's record.
    notification = await db.scalar(
        select(Notification).where(
            Notification.id == notification_id,
            Notification.user_id == user_id,
        ),
    )
    if notification is None:
        raise HTTPException(status_code=404, detail='Notification not found.')
    return notification


def _decode_notification_cursor(cursor: str) -> tuple[datetime, int]:
    """Decode the exclusive keyset position for one notification page."""
    try:
        padded = cursor + '=' * (-len(cursor) % 4)
        payload = json.loads(
            base64.urlsafe_b64decode(padded.encode('ascii')).decode('utf-8'),
        )
        return datetime.fromisoformat(payload['created_at']), int(payload['id'])
    except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(
            status_code=422, detail='invalid_notification_cursor',
        ) from exc


def _encode_notification_cursor(notification: Notification) -> str:
    """Encode the last returned row as the next exclusive keyset position."""
    payload = json.dumps(
        {'created_at': notification.created_at.isoformat(), 'id': notification.id},
        separators=(',', ':'),
    ).encode('utf-8')
    return base64.urlsafe_b64encode(payload).decode('ascii').rstrip('=')


async def list_notifications(
    status: NotificationStatus | None,
    notification_type: NotificationType | None,
    page_size: int,
    cursor: str | None,
    db: AsyncSession,
    me: User,
) -> NotificationList:
    """Return a page of notification-centre records for the current user.

    Args:
        status: Optional read-status filter.
        notification_type: Optional notification-category filter.
        page_size: Maximum number of items in the page.
        cursor: Optional exclusive cursor from the previous response.
        db: Database session used for queries.
        me: Authenticated user owning the records.

    Returns:
        Paginated notification response.
    """
    conditions = [Notification.user_id == me.id]
    if status == 'unread':
        conditions.append(Notification.is_read.is_(False))
    elif status == 'read':
        conditions.append(Notification.is_read.is_(True))
    if notification_type is not None:
        conditions.append(Notification.type == notification_type)
    if cursor is not None:
        created_at, notification_id = _decode_notification_cursor(cursor)
        conditions.append(
            or_(
                Notification.created_at < created_at,
                and_(
                    Notification.created_at == created_at,
                    Notification.id < notification_id,
                ),
            ),
        )
    items_result = await db.execute(
        select(Notification)
        .where(*conditions)
        .order_by(Notification.created_at.desc(), Notification.id.desc())
        .limit(page_size + 1),
    )
    items = list(items_result.scalars().all())
    has_next_page = len(items) > page_size
    visible_items = items[:page_size]
    return NotificationList(
        items=[_notification_to_out(item) for item in visible_items],
        next_cursor=(
            _encode_notification_cursor(visible_items[-1])
            if has_next_page and visible_items
            else None
        ),
    )


async def get_notification_unread_count(
    db: AsyncSession,
    me: User,
) -> NotificationUnreadCount:
    """Return the unread notification count for one user.

    Args:
        db: Database session used for the aggregate query.
        me: Authenticated user owning the notifications.

    Returns:
        Unread notification count.
    """
    result = await db.execute(
        select(func.count())
        .select_from(Notification)
        .where(Notification.user_id == me.id, Notification.is_read.is_(False)),
    )
    return NotificationUnreadCount(
        unread_count=int(cast(int, result.scalar())),
    )


async def get_notification_device_status(
    db: AsyncSession,
    me: User,
) -> DeviceStatusResponse:
    """Return FCM device diagnostics for the authenticated user.

    Args:
        db: Database session used to load device status.
        me: Authenticated user owning the devices.

    Returns:
        Active-token summary and individual device diagnostics.
    """
    devices = [
        DeviceTokenStatus.model_validate(row)
        for row in await list_fcm_device_status(me.id, db)
    ]
    active_count = sum(device.is_active for device in devices)
    return DeviceStatusResponse(
        user_id=me.id,
        has_fcm_token=active_count > 0,
        token_count=active_count,
        devices=devices,
    )


async def mark_notification_read(
    notification_id: int,
    db: AsyncSession,
    me: User,
) -> NotificationOut:
    """Mark an owned notification as read.

    Args:
        notification_id: Notification primary key.
        db: Database session used for the update.
        me: Authenticated owner of the notification.

    Returns:
        Updated notification response.
    """
    notification = await _get_owned_notification(notification_id, me.id, db)
    notification.is_read = True
    await db.commit()
    await db.refresh(notification)
    return _notification_to_out(notification)


async def mark_all_notifications_read(
    db: AsyncSession,
    me: User,
) -> NotificationBulkReadResponse:
    """Mark every unread notification for the current user as read.

    Args:
        db: Database session used for the bulk update.
        me: Authenticated notification owner.

    Returns:
        Number of updated notifications.
    """
    # A set-based update avoids loading every unread notification into memory.
    result = cast(
        CursorResult[Any], await db.execute(
            update(Notification)
            .where(Notification.user_id == me.id, Notification.is_read.is_(False))
            .values(is_read=True),
        ),
    )
    await db.commit()
    return NotificationBulkReadResponse(
        updated_count=int(cast(int, result.rowcount)),
    )


async def delete_notification(
    notification_id: int,
    db: AsyncSession,
    me: User,
) -> dict[str, str]:
    """Delete an owned notification.

    Args:
        notification_id: Notification primary key.
        db: Database session used for deletion.
        me: Authenticated notification owner.

    Returns:
        Deletion outcome message.
    """
    notification = await _get_owned_notification(notification_id, me.id, db)
    await db.delete(notification)
    await db.commit()
    return {'message': 'Notification deleted.'}
