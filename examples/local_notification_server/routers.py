from __future__ import annotations

import hashlib
import json
import time
from typing import cast

import redis.asyncio as redis
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query
from fastapi import Security
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.models import Notification
from examples.auth.models import Site
from examples.auth.models import SiteNotificationPreference
from examples.auth.models import User
from examples.auth.redis_pool import get_redis_pool
from examples.auth.user_service import list_effective_sites_for_user
from examples.db_management.deps import get_current_user
from examples.db_management.deps import is_super_admin
from examples.db_management.services.site_services import list_sites
from examples.local_notification_server.fcm_service import (
    FcmSendResult,
)
from examples.local_notification_server.fcm_service import (
    send_fcm_notification_service,
)
from examples.local_notification_server.lang_config import normalize_language
from examples.local_notification_server.schemas import \
    DeviceStatusResponse
from examples.local_notification_server.schemas import \
    DeviceTokenStatus
from examples.local_notification_server.schemas import \
    NotificationBulkReadResponse
from examples.local_notification_server.schemas import NotificationList
from examples.local_notification_server.schemas import NotificationOut
from examples.local_notification_server.schemas import NotificationStatus
from examples.local_notification_server.schemas import NotificationType
from examples.local_notification_server.schemas import \
    NotificationUnreadCount
from examples.local_notification_server.schemas import \
    SiteNotificationPreferenceOut
from examples.local_notification_server.schemas import \
    SiteNotificationPreferenceUpdateRequest
from examples.local_notification_server.schemas import SiteNotifyRequest
from examples.local_notification_server.schemas import TestNotificationResponse
from examples.local_notification_server.schemas import TokenRequest
from examples.local_notification_server.schemas import TokenStoreResponse
from examples.local_notification_server.services import \
    _execute_push_tasks_bounded_streaming
from examples.local_notification_server.services import \
    _iter_push_tasks_streaming
from examples.local_notification_server.services import \
    create_notification_records_for_users
from examples.local_notification_server.services import delete_fcm_token_metadata
from examples.local_notification_server.services import diagnose_push_preflight
from examples.local_notification_server.services import \
    get_site_notification_user_ids_cached
from examples.local_notification_server.services import list_fcm_device_status
from examples.local_notification_server.services import \
    load_active_fcm_device_tokens
from examples.local_notification_server.services import \
    mark_fcm_tokens_failure
from examples.local_notification_server.services import \
    mark_fcm_tokens_success
from examples.local_notification_server.services import \
    mark_invalid_fcm_tokens_for_users
from examples.local_notification_server.services import \
    record_fcm_token_registration
from examples.local_notification_server.services import \
    refresh_fcm_token_cache_for_users
from examples.local_notification_server.services import \
    refresh_site_notification_user_cache

router: APIRouter = APIRouter()
_dedupe_ttl_with_violation_id: int = 600
_dedupe_ttl_without_violation_id: int = 15


def _notification_dedupe_key(req: SiteNotifyRequest) -> str:
    """Build a stable Redis dedupe key for a notification request.

    Args:
        req: Validated notification request.

    Returns:
        Redis key used to suppress duplicate notification sends.
    """
    if req.violation_id is not None:
        return (
            'fcm_notification_dedupe:'
            f'{req.site}:{req.stream_name}:{req.violation_id}'
        )

    payload_hash = hashlib.sha256(
        json.dumps(
            {
                'site': req.site,
                'stream_name': req.stream_name,
                'body': req.body,
                'image_path': req.image_path,
            },
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8'),
    ).hexdigest()
    return f'fcm_notification_dedupe:{payload_hash}'


def _notification_dedupe_ttl(req: SiteNotifyRequest) -> int:
    """Return the dedupe TTL for a notification request.

    Args:
        req: Validated notification request.

    Returns:
        TTL in seconds for the request's dedupe key.
    """
    if req.violation_id is not None:
        return _dedupe_ttl_with_violation_id
    return _dedupe_ttl_without_violation_id


async def _claim_notification_send(
    req: SiteNotifyRequest,
    rds: redis.Redis,
) -> bool:
    """Claim a notification send slot across all notification servers.

    Args:
        req: Validated notification request.
        rds: Redis connection used to claim the dedupe key.

    Returns:
        True when this server owns the send slot.
    """
    return bool(
        await rds.set(
            _notification_dedupe_key(req),
            '1',
            ex=_notification_dedupe_ttl(req),
            nx=True,
        ),
    )


@router.post('/store_token', response_model=TokenStoreResponse)
async def store_fcm_token(
    req: TokenRequest,
    db: AsyncSession = Depends(get_db),
    rds: redis.Redis = Depends(get_redis_pool),
) -> TokenStoreResponse:
    """Store an FCM device token in DB and refresh Redis send cache.

    Args:
        req: Payload containing the user ID, device token, and language.
        db: Async database session dependency.
        rds: Redis connection dependency.

    Raises:
        HTTPException: Raised when the user does not exist or the requested
            device language is unsupported.

    Returns:
        Success message indicating that the token was stored.
    """
    # Validate user existence
    stmt_user = select(User.id).where(User.id == req.user_id)
    result = await db.execute(stmt_user)
    if not result.scalar():
        raise HTTPException(status_code=404, detail='User not found')

    device_lang = normalize_language(req.device_lang)
    if device_lang is None:
        raise HTTPException(status_code=422, detail='unsupported_device_lang')

    meta = await record_fcm_token_registration(req, device_lang, db, rds)

    return TokenStoreResponse(
        ok=True,
        updated=True,
        user_id=req.user_id,
        device_lang=device_lang,
        registered_at=meta['registered_at'],
        last_seen_at=meta['last_seen_at'],
    )


@router.delete('/delete_token')
async def delete_fcm_token(
    req: TokenRequest,
    db: AsyncSession = Depends(get_db),
    rds: redis.Redis = Depends(get_redis_pool),
) -> dict[str, str]:
    """Disable an FCM device token in DB and remove it from Redis cache.

    Args:
        req: Payload containing the user ID and device token.
        db: Async database session dependency.
        rds: Redis connection dependency.

    Returns:
        Message indicating whether the token was deleted.
    """
    # Validate user existence
    stmt_user = select(User.id).where(User.id == req.user_id)
    result = await db.execute(stmt_user)
    if not result.scalar():
        return {'message': 'User not found.'}

    deleted = await delete_fcm_token_metadata(
        req.user_id,
        req.device_token,
        db,
        rds,
    )

    # Use Redis pipeline for batch operations. Redis is only a send cache, so
    # the API result is based on the DB update above.
    pipe = rds.pipeline()
    key: str = f"fcm_tokens:{req.user_id}"

    pipe.hdel(key, req.device_token)
    pipe.hlen(key)  # Check remaining token count

    results = await pipe.execute()
    remaining_tokens: int = results[1]

    # Delete key if no tokens remain
    if remaining_tokens == 0:
        await rds.delete(key)

    if not deleted:
        return {'message': 'Token not found.'}

    return {'message': 'Token deleted.'}


@router.post('/send_fcm_notification')
async def send_fcm_notification(
    req: SiteNotifyRequest,
    db: AsyncSession = Depends(get_db),
    _cred: JwtAuthorizationCredentials = Security(jwt_access),
    rds: redis.Redis = Depends(get_redis_pool),
) -> dict[str, object]:
    """Send an FCM notification to subscribed users for one site.

    Args:
        req: Notification request including site, stream name, warning body,
            optional image path, and optional violation ID.
        db: Async database session dependency.
        _cred: JWT credentials required for authentication.
        rds: Redis connection dependency.

    Returns:
        Response payload containing success status, message, and batch stats.
    """
    if not req.body:
        return {'success': False, 'message': 'Body is empty, nothing to send.'}

    if not await _claim_notification_send(req, rds):
        return {
            'success': True,
            'message': 'Duplicate notification skipped.',
        }

    user_ids: list[int] | None = await get_site_notification_user_ids_cached(
        req.site,
        db,
        rds,
    )
    if user_ids is None:
        return {
            'success': False,
            'message': f"Site '{req.site}' not found.",
        }

    if not user_ids:
        return {
            'success': False,
            'message': f"Site '{req.site}' has no subscribed users.",
        }

    notification_record_count = await create_notification_records_for_users(
        req,
        user_ids,
        db,
    )
    await refresh_fcm_token_cache_for_users(user_ids, db, rds)

    start_time: float = time.time()
    push_tasks = _iter_push_tasks_streaming(req, user_ids, rds)
    translation_time: float = time.time() - start_time
    print(f"Notification preparation time: {translation_time:.3f}s")

    fcm_start_time: float = time.time()

    ok, total_batches, successful_batches, error_msg = (
        await _execute_push_tasks_bounded_streaming(
            push_tasks,
            timeout=30.0,
            invalid_token_handler=(
                lambda invalid_tokens: mark_invalid_fcm_tokens_for_users(
                    user_ids,
                    invalid_tokens,
                    rds,
                    db=db,
                )
            ),
        )
    )
    if not ok:
        # Log internal detail on server, but return a generic message to client
        if error_msg and error_msg != 'FCM notification sending timed out.':
            print(f"FCM sending failed: {error_msg}")
        user_message = (
            'FCM notification sending timed out.'
            if error_msg == 'FCM notification sending timed out.'
            else 'Failed to send FCM notifications.'
        )
        return {'success': False, 'message': user_message}

    fcm_time: float = time.time() - fcm_start_time
    assert total_batches is not None
    assert successful_batches is not None
    if total_batches == 0:
        preflight_stats = await diagnose_push_preflight(req, user_ids, rds)
        if preflight_stats['unique_tokens'] == 0:
            skip_message = f"Site '{req.site}' has no device tokens."
        else:
            skip_message = (
                f"Site '{req.site}' has no sendable device tokens."
            )
        print(
            'FCM sending skipped: no sendable batches; '
            f"site={req.site!r}, stream={req.stream_name!r}, "
            f"subscribed_users={len(user_ids)}. "
            f"diagnostics={preflight_stats!r}.",
        )
        return {
            'success': False,
            'message': skip_message,
            'stats': {
                'translation_time': translation_time,
                'fcm_time': fcm_time,
                'total_batches': total_batches,
                'successful_batches': successful_batches,
                'notification_records': notification_record_count,
                'preflight': preflight_stats,
            },
        }

    overall_success: bool = total_batches == successful_batches

    print(
        f"FCM sending time: {fcm_time:.3f}s, successful batches: "
        f"{successful_batches}/{total_batches}",
    )

    return {
        'success': overall_success,
        'message': (
            f'FCM notification processed. '
            f'{successful_batches}/{total_batches} batches succeeded.'
        ),
        'stats': {
            'translation_time': translation_time,
            'fcm_time': fcm_time,
            'total_batches': total_batches,
            'successful_batches': successful_batches,
            'notification_records': notification_record_count,
        },
    }


def _notification_to_out(notification: Notification) -> NotificationOut:
    """Convert a notification ORM object into its API response shape."""
    return NotificationOut(
        id=notification.id,
        type=cast(NotificationType, notification.type),
        title=notification.title,
        body=notification.body,
        deep_link=notification.deep_link,
        is_read=notification.is_read,
        created_at=notification.created_at,
        metadata=dict(notification.metadata_json or {}),
    )


async def _get_owned_notification(
    notification_id: int,
    user_id: int,
    db: AsyncSession,
) -> Notification:
    """Load one notification belonging to the current user."""
    notification = await db.scalar(
        select(Notification).where(
            Notification.id == notification_id,
            Notification.user_id == user_id,
        ),
    )
    if notification is None:
        raise HTTPException(status_code=404, detail='Notification not found.')
    return notification


@router.get('/notifications', response_model=NotificationList)
async def list_notifications(
    status: NotificationStatus | None = Query(default=None),
    notification_type: NotificationType | None = Query(
        default=None,
        alias='type',
    ),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> NotificationList:
    """List current user's notification-center records."""
    conditions = [Notification.user_id == me.id]
    if status == 'unread':
        conditions.append(Notification.is_read.is_(False))
    elif status == 'read':
        conditions.append(Notification.is_read.is_(True))

    if notification_type is not None:
        conditions.append(Notification.type == notification_type)

    total_result = await db.execute(
        select(func.count()).select_from(Notification).where(*conditions),
    )
    total = int(total_result.scalar() or 0)
    offset = (page - 1) * page_size
    item_result = await db.execute(
        select(Notification)
        .where(*conditions)
        .order_by(Notification.created_at.desc(), Notification.id.desc())
        .offset(offset)
        .limit(page_size),
    )
    items = [
        _notification_to_out(notification)
        for notification in item_result.scalars().all()
    ]
    return NotificationList(
        total=total,
        page=page,
        page_size=page_size,
        items=items,
    )


@router.get(
    '/notifications/unread_count',
    response_model=NotificationUnreadCount,
)
async def get_notification_unread_count(
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> NotificationUnreadCount:
    """Return the current user's unread notification count."""
    result = await db.execute(
        select(func.count())
        .select_from(Notification)
        .where(
            Notification.user_id == me.id,
            Notification.is_read.is_(False),
        ),
    )
    return NotificationUnreadCount(unread_count=int(result.scalar() or 0))


@router.get(
    '/notifications/device-status',
    response_model=DeviceStatusResponse,
)
async def get_notification_device_status(
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> DeviceStatusResponse:
    """Return diagnostic metadata for the current user's notification tokens."""
    rows = await list_fcm_device_status(me.id, db)
    devices = [DeviceTokenStatus.model_validate(row) for row in rows]
    active_count = sum(1 for device in devices if device.is_active)
    return DeviceStatusResponse(
        user_id=me.id,
        has_fcm_token=active_count > 0,
        token_count=active_count,
        devices=devices,
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
    """Send a test push notification to the current user's active tokens."""
    await refresh_fcm_token_cache_for_users([me.id], db, rds)
    device_tokens = await load_active_fcm_device_tokens(me.id, db)
    if not device_tokens:
        return TestNotificationResponse(
            success=False,
            message='No FCM token registered for this user.',
            attempted_tokens=0,
        )

    result = await send_fcm_notification_service(
        device_tokens=device_tokens,
        title='Visionnaire test notification',
        body='This is a test notification.',
        data={'type': 'system', 'test': 'true'},
    )
    if isinstance(result, FcmSendResult):
        invalid_tokens = set(result.invalid_tokens)
        if result.success_count > 0 and result.failure_count == 0:
            await mark_fcm_tokens_success(me.id, device_tokens, rds, db=db)
        elif result.failure_count > 0:
            await mark_fcm_tokens_failure(
                me.id,
                device_tokens,
                rds,
                'fcm_error',
                db=db,
            )
        if invalid_tokens:
            await mark_invalid_fcm_tokens_for_users(
                [me.id],
                invalid_tokens,
                rds,
                db=db,
            )
        return TestNotificationResponse(
            success=bool(result),
            message=(
                'Test notification sent.'
                if bool(result)
                else 'Test notification failed.'
            ),
            attempted_tokens=len(device_tokens),
            success_count=result.success_count,
            failure_count=result.failure_count,
            invalid_tokens=len(invalid_tokens),
        )

    success = bool(result)
    if success:
        await mark_fcm_tokens_success(me.id, device_tokens, rds, db=db)
    else:
        await mark_fcm_tokens_failure(
            me.id,
            device_tokens,
            rds,
            'fcm_error',
            db=db,
        )
    return TestNotificationResponse(
        success=success,
        message=(
            'Test notification sent.'
            if success
            else 'Test notification failed.'
        ),
        attempted_tokens=len(device_tokens),
        success_count=len(device_tokens) if success else 0,
        failure_count=0 if success else len(device_tokens),
    )


@router.patch('/notifications/{notification_id}/read')
async def mark_notification_read(
    notification_id: int,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> NotificationOut:
    """Mark one notification as read."""
    notification = await _get_owned_notification(notification_id, me.id, db)
    notification.is_read = True
    await db.commit()
    await db.refresh(notification)
    return _notification_to_out(notification)


@router.patch(
    '/notifications/read_all',
    response_model=NotificationBulkReadResponse,
)
async def mark_all_notifications_read(
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> NotificationBulkReadResponse:
    """Mark all current-user notifications as read."""
    result = await db.execute(
        update(Notification)
        .where(
            Notification.user_id == me.id,
            Notification.is_read.is_(False),
        )
        .values(is_read=True),
    )
    await db.commit()
    return NotificationBulkReadResponse(
        updated_count=int(result.rowcount or 0),
    )


@router.delete('/notifications/{notification_id}')
async def delete_notification(
    notification_id: int,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> dict[str, str]:
    """Delete one notification owned by the current user."""
    notification = await _get_owned_notification(notification_id, me.id, db)
    await db.delete(notification)
    await db.commit()
    return {'message': 'Notification deleted.'}


async def _list_notification_scope_sites(
    db: AsyncSession,
    me: User,
) -> list[Site]:
    """Return sites the current user may manage notification settings for.

    Args:
        db: Async database session dependency.
        me: Current authenticated user.

    Returns:
        Sites visible for notification preference management.

    Raises:
        HTTPException: Raised when a non-super-admin user has no group.
    """
    if is_super_admin(me):
        return await list_sites(db)
    if me.group_id is None:
        raise HTTPException(status_code=403, detail='User without group.')
    return await list_sites(db, group_id=me.group_id)


def _serialize_site_preferences(
    sites: list[Site],
    pref_map: dict[int, bool],
    access_site_ids: set[int],
) -> list[SiteNotificationPreferenceOut]:
    """Build the unified response payload for notification preferences.

    Args:
        sites: Sites in the current user's notification-management scope.
        pref_map: Explicit notification preference values by site ID.
        access_site_ids: Sites enabled by effective site access.

    Returns:
        Serialised notification preference records.
    """
    return [
        SiteNotificationPreferenceOut(
            site_id=site.id,
            site_name=site.name,
            group_name=site.groups[0].name if site.groups else None,
            is_enabled=pref_map.get(site.id, site.id in access_site_ids),
        )
        for site in sites
    ]


@router.get(
    '/notifications/site_preferences',
    response_model=list[SiteNotificationPreferenceOut],
)
async def list_site_notification_preferences(
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> list[SiteNotificationPreferenceOut]:
    """List notification preferences visible to the current user.

    Args:
        db: Async database session dependency.
        me: Current authenticated user.

    Returns:
        Per-site notification preference records.
    """
    sites = await _list_notification_scope_sites(db, me)
    if not sites:
        return []

    site_ids = [site.id for site in sites]
    pref_rows = await db.execute(
        select(
            SiteNotificationPreference.site_id,
            SiteNotificationPreference.is_enabled,
        ).where(
            SiteNotificationPreference.user_id == me.id,
            SiteNotificationPreference.site_id.in_(site_ids),
        ),
    )
    pref_pairs = [
        cast(tuple[int, bool], (row[0], row[1]))
        for row in pref_rows.all()
    ]
    pref_map: dict[int, bool] = {
        site_id: is_enabled
        for site_id, is_enabled in pref_pairs
    }

    access_site_ids = {
        site.id
        for site in await list_effective_sites_for_user(me, db)
        if site.id in site_ids
    }

    return _serialize_site_preferences(sites, pref_map, access_site_ids)


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
    """Replace the current user's site notification preferences.

    Args:
        payload: Full preference replacement payload.
        db: Async database session dependency.
        me: Current authenticated user.
        rds: Redis connection used to refresh recipient caches.

    Returns:
        Updated per-site notification preference records.

    Raises:
        HTTPException: Raised when the user requests a site outside their
            management scope.
    """
    sites = await _list_notification_scope_sites(db, me)
    allowed_site_ids = {site.id for site in sites}
    requested_site_ids = {item.site_id for item in payload.preferences}

    if not requested_site_ids:
        return await list_site_notification_preferences(db, me)

    invalid_site_ids = requested_site_ids - allowed_site_ids
    if invalid_site_ids:
        raise HTTPException(
            status_code=403,
            detail='Cannot subscribe to sites outside your scope.',
        )

    pref_result = await db.execute(
        select(SiteNotificationPreference).where(
            SiteNotificationPreference.user_id == me.id,
            SiteNotificationPreference.site_id.in_(allowed_site_ids),
        ),
    )
    existing_prefs = {
        pref.site_id: pref for pref in pref_result.scalars().all()
    }

    access_site_ids = {
        site.id
        for site in await list_effective_sites_for_user(me, db)
        if site.id in allowed_site_ids
    }

    changed_site_names: list[str] = []
    requested_pref_map = {
        item.site_id: item.is_enabled for item in payload.preferences
    }
    for site in sites:
        if site.id not in requested_pref_map:
            continue

        desired_enabled = requested_pref_map[site.id]
        pref = existing_prefs.get(site.id)
        current_enabled = (
            pref.is_enabled if pref is not None else site.id in access_site_ids
        )

        if pref is None:
            db.add(
                SiteNotificationPreference(
                    user_id=me.id,
                    site_id=site.id,
                    is_enabled=desired_enabled,
                ),
            )
        else:
            pref.is_enabled = desired_enabled

        if current_enabled != desired_enabled:
            changed_site_names.append(site.name)

    await db.commit()
    if changed_site_names:
        for site_name in changed_site_names:
            await refresh_site_notification_user_cache(site_name, db, rds)

    return await list_site_notification_preferences(db, me)
