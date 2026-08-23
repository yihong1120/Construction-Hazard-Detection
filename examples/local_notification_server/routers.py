from __future__ import annotations

import redis.asyncio as redis
from fastapi import APIRouter
from fastapi import Depends
from fastapi import Query
from fastapi import Security
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.models import User
from examples.auth.redis_pool import get_redis_pool
from examples.db_management.deps import get_current_user
from examples.local_notification_server.notification_centre_service import (
    delete_notification as delete_notification_service,
)
from examples.local_notification_server.notification_centre_service import (
    get_notification_device_status as get_notification_device_status_service,
)
from examples.local_notification_server.notification_centre_service import (
    get_notification_unread_count as get_notification_unread_count_service,
)
from examples.local_notification_server.notification_centre_service import (
    list_notifications as list_notifications_service,
)
from examples.local_notification_server.notification_centre_service import (
    mark_all_notifications_read as mark_all_notifications_read_service,
)
from examples.local_notification_server.notification_centre_service import (
    mark_notification_read as mark_notification_read_service,
)
from examples.local_notification_server.notification_delivery_service import (
    register_fcm_device,
)
from examples.local_notification_server.notification_delivery_service import (
    send_site_notification,
)
from examples.local_notification_server.notification_delivery_service import (
    send_test_notification as send_test_notification_service,
)
from examples.local_notification_server.notification_delivery_service import (
    unregister_fcm_device,
)
from examples.local_notification_server.schemas import (
    DeviceRegistrationRequest,
)
from examples.local_notification_server.schemas import (
    DeviceRegistrationResponse,
)
from examples.local_notification_server.schemas import DeviceStatusResponse
from examples.local_notification_server.schemas import (
    DeviceUnregistrationRequest,
)
from examples.local_notification_server.schemas import (
    NotificationBulkReadResponse,
)
from examples.local_notification_server.schemas import NotificationList
from examples.local_notification_server.schemas import NotificationOut
from examples.local_notification_server.schemas import NotificationStatus
from examples.local_notification_server.schemas import NotificationType
from examples.local_notification_server.schemas import NotificationUnreadCount
from examples.local_notification_server.schemas import (
    SiteNotificationPreferenceOut,
)
from examples.local_notification_server.schemas import (
    SiteNotificationPreferenceUpdateRequest,
)
from examples.local_notification_server.schemas import SiteNotifyRequest
from examples.local_notification_server.schemas import TestNotificationResponse
from examples.local_notification_server.site_preference_service import (
    list_site_notification_preferences as list_site_notification_preferences_service,
)
from examples.local_notification_server.site_preference_service import (
    update_site_notification_preferences as update_site_preferences_service,
)

# Route handlers remain thin adapters; workflow logic belongs to domain services.
router = APIRouter()


@router.put('/devices', response_model=DeviceRegistrationResponse)
async def register_device(
    req: DeviceRegistrationRequest,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
    rds: redis.Redis = Depends(get_redis_pool),
) -> DeviceRegistrationResponse:
    """Create or update the authenticated user's FCM device registration.

    Args:
        req: Validated device-token registration payload.
        db: Database session dependency.
        me: Authenticated token owner.
        rds: Redis connection dependency.

    Returns:
        Stored token metadata without exposing the token value.
    """
    return await register_fcm_device(me.id, req, db, rds)


@router.delete('/devices')
async def unregister_device(
    req: DeviceUnregistrationRequest,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
    rds: redis.Redis = Depends(get_redis_pool),
) -> dict[str, str]:
    """Remove the authenticated user's FCM device registration.

    Args:
        req: Validated token-removal payload.
        db: Database session dependency.
        me: Authenticated token owner.
        rds: Redis connection dependency.

    Returns:
        Deletion outcome message.
    """
    return await unregister_fcm_device(me.id, req, db, rds)


@router.post('/send_fcm_notification')
async def send_fcm_notification(
    req: SiteNotifyRequest,
    db: AsyncSession = Depends(get_db),
    _cred: JwtAuthorizationCredentials = Security(jwt_access),
    rds: redis.Redis = Depends(get_redis_pool),
) -> dict[str, object]:
    """Send a notification to users subscribed to one site.

    Args:
        req: Validated notification payload.
        db: Database session dependency.
        _cred: Verified JWT credentials.
        rds: Redis connection dependency.

    Returns:
        Notification delivery outcome and batch statistics.
    """
    return await send_site_notification(req, db, rds)


@router.get('/notifications', response_model=NotificationList)
async def list_notifications(
    status: NotificationStatus | None = Query(default=None),
    notification_type: NotificationType | None = Query(
        default=None,
        alias='type',
    ),
    page_size: int = Query(20, ge=1, le=100),
    cursor: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> NotificationList:
    """List the current user's notification-centre records.

    Args:
        status: Optional read-status filter.
        notification_type: Optional notification category filter.
        page_size: Maximum number of records to return.
        cursor: Exclusive cursor returned by the previous page.
        db: Database session dependency.
        me: Authenticated user.

    Returns:
        Paginated notification records.
    """
    return await list_notifications_service(
        status,
        notification_type,
        page_size,
        cursor,
        db,
        me,
    )


@router.get(
    '/notifications/unread_count',
    response_model=NotificationUnreadCount,
)
async def get_notification_unread_count(
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> NotificationUnreadCount:
    """Return the unread notification count for the current user.

    Args:
        db: Database session dependency.
        me: Authenticated user.

    Returns:
        Unread notification count for the authenticated user.
    """
    return await get_notification_unread_count_service(
        db,
        me,
    )


@router.get(
    '/notifications/device-status',
    response_model=DeviceStatusResponse,
)
async def get_notification_device_status(
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> DeviceStatusResponse:
    """Return current FCM device diagnostics for the authenticated user.

    Args:
        db: Database session dependency.
        me: Authenticated user.

    Returns:
        Active FCM device status and delivery diagnostics.
    """
    return await get_notification_device_status_service(
        db,
        me,
    )


@router.post(
    '/notifications/test',
    response_model=TestNotificationResponse,
)
async def send_test_notification(
    rds: redis.Redis = Depends(get_redis_pool),
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> TestNotificationResponse:
    """Send a test notification to the current user's active devices.

    Args:
        rds: Redis connection dependency.
        db: Database session dependency.
        me: Authenticated user receiving the test.

    Returns:
        Aggregated test delivery result.
    """
    return await send_test_notification_service(
        rds,
        db,
        me,
    )


@router.patch('/notifications/{notification_id}/read')
async def mark_notification_read(
    notification_id: int,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> NotificationOut:
    """Mark one notification as read for its owner.

    Args:
        notification_id: Notification primary key.
        db: Database session dependency.
        me: Authenticated notification owner.

    Returns:
        Updated notification record.
    """
    return await mark_notification_read_service(
        notification_id,
        db,
        me,
    )


@router.patch(
    '/notifications/read_all',
    response_model=NotificationBulkReadResponse,
)
async def mark_all_notifications_read(
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> NotificationBulkReadResponse:
    """Mark all unread notifications as read for the current user.

    Args:
        db: Database session dependency.
        me: Authenticated notification owner.

    Returns:
        Count of records changed to read.
    """
    return await mark_all_notifications_read_service(
        db,
        me,
    )


@router.delete('/notifications/{notification_id}')
async def delete_notification(
    notification_id: int,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> dict[str, str]:
    """Delete one notification owned by the current user.

    Args:
        notification_id: Notification primary key.
        db: Database session dependency.
        me: Authenticated notification owner.

    Returns:
        Deletion outcome message.
    """
    return await delete_notification_service(
        notification_id,
        db,
        me,
    )


@router.get(
    '/notifications/site_preferences',
    response_model=list[SiteNotificationPreferenceOut],
)
async def list_site_notification_preferences(
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> list[SiteNotificationPreferenceOut]:
    """List site notification preferences visible to the current user.

    Args:
        db: Database session dependency.
        me: Authenticated user.

    Returns:
        Notification preference state for each manageable site.
    """
    return await list_site_notification_preferences_service(
        db,
        me,
    )


@router.put(
    '/notifications/site_preferences',
    response_model=list[SiteNotificationPreferenceOut],
)
async def update_site_notification_preferences(
    payload: SiteNotificationPreferenceUpdateRequest,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
    rds: redis.Redis = Depends(get_redis_pool),
) -> list[SiteNotificationPreferenceOut]:
    """Replace explicit site notification preferences for the current user.

    Args:
        payload: Requested per-site notification settings.
        db: Database session dependency.
        me: Authenticated user.
        rds: Redis connection dependency.

    Returns:
        Current notification preference state for each manageable site.
    """
    return await update_site_preferences_service(
        payload,
        db,
        me,
        rds,
    )
