from __future__ import annotations

import logging
import time

import redis.asyncio as redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.models import Site
from examples.auth.models import User
from examples.local_notification_server.event_delivery import FcmEventDelivery
from examples.local_notification_server.fcm_service import (
    send_fcm_notification_service,
)
from examples.local_notification_server.notification_claim import (
    notification_event_key as _notification_dedupe_key,
)
from examples.local_notification_server.notification_claim import (
    NotificationClaim,
)
from examples.local_notification_server.notification_claim import (
    NotificationLeaseLost,
)
from examples.local_notification_server.push_dispatch import (
    create_notification_records_for_users,
)
from examples.local_notification_server.push_dispatch import (
    execute_push_tasks_bounded_streaming,
)
from examples.local_notification_server.push_dispatch import (
    iter_push_tasks_streaming,
)
from examples.local_notification_server.push_dispatch import (
    preflight_from_token_stats,
)
from examples.local_notification_server.push_dispatch import PushTokenStats
from examples.local_notification_server.schemas import (
    DeviceRegistrationRequest,
)
from examples.local_notification_server.schemas import (
    DeviceRegistrationResponse,
)
from examples.local_notification_server.schemas import (
    DeviceUnregistrationRequest,
)
from examples.local_notification_server.schemas import SiteNotifyRequest
from examples.local_notification_server.schemas import TestNotificationResponse
from examples.local_notification_server.services import (
    delete_fcm_token_metadata,
)
from examples.local_notification_server.services import (
    ensure_fcm_token_cache_for_users,
)
from examples.local_notification_server.services import (
    load_active_fcm_device_tokens,
)
from examples.local_notification_server.services import (
    record_fcm_token_registration,
)
from examples.local_notification_server.site_recipient_cache import (
    get_site_notification_user_ids_cached,
)
from examples.local_notification_server.token_delivery_state import (
    FcmDeliveryState,
)

_dedupe_ttl_with_violation_id = 600
_dedupe_ttl_without_violation_id = 15
logger = logging.getLogger(__name__)


async def register_fcm_device(
    user_id: int,
    req: DeviceRegistrationRequest,
    db: AsyncSession,
    rds: redis.Redis,
) -> DeviceRegistrationResponse:
    """Register an FCM token for the authenticated user.

    Args:
        user_id: Authenticated identifier that owns the device token.
        req: Validated token registration request.
        db: Database session used to store the token.
        rds: Redis connection used to refresh the send cache.

    Returns:
        Non-sensitive registration metadata.
    """
    metadata = await record_fcm_token_registration(user_id, req, db, rds)
    return DeviceRegistrationResponse(
        ok=True,
        updated=True,
        user_id=user_id,
        device_lang=req.device_lang,
        registered_at=metadata['registered_at'],
        last_seen_at=metadata['last_seen_at'],
    )


async def unregister_fcm_device(
    user_id: int,
    req: DeviceUnregistrationRequest,
    db: AsyncSession,
    rds: redis.Redis,
) -> dict[str, str]:
    """Disable an FCM token and remove it from the Redis send cache.

    Args:
        user_id: Authenticated identifier that owns the device token.
        req: Validated token deletion request.
        db: Database session used to update token state.
        rds: Redis connection used to remove the send cache entry.

    Returns:
        Deletion outcome message.
    """
    deleted = await delete_fcm_token_metadata(
        user_id,
        req.device_token,
        db,
        rds,
    )
    if not deleted:
        return {'message': 'Token not found.'}
    return {'message': 'Token deleted.'}


async def send_site_notification(
    req: SiteNotifyRequest,
    db: AsyncSession,
    rds: redis.Redis,
) -> dict[str, object]:
    if req.site_id is not None:
        current_name = await db.scalar(
            select(Site.name).where(Site.id == req.site_id),
        )
        if current_name is None:
            return {
                'success': False,
                'message': 'Notification site not found.',
                'outcome': 'permanent_failure',
            }
        req = req.model_copy(update={'site': current_name})
    ttl = (
        _dedupe_ttl_with_violation_id
        if req.violation_id is not None
        else _dedupe_ttl_without_violation_id
    )
    claim = NotificationClaim(rds, _notification_dedupe_key(req), ttl)
    try:
        if not await claim.acquire():
            if await claim.is_completed():
                return {
                    'success': True,
                    'message': 'Duplicate notification skipped.',
                    'outcome': 'skipped',
                }
            return {
                'success': False,
                'message': (
                    'Notification delivery is in progress; retry later.'
                ),
                'outcome': 'retryable',
            }
        async with claim:
            result = await claim.run(
                _deliver_site_notification(
                    req,
                    db,
                    rds,
                    claim,
                ),
            )
            if result['success']:
                await claim.finish()
            return result
    except NotificationLeaseLost:
        return {
            'success': False,
            'outcome': 'retryable',
            'message': 'Notification delivery lease expired.',
        }


async def _deliver_site_notification(
    req: SiteNotifyRequest,
    db: AsyncSession,
    rds: redis.Redis,
    claim: NotificationClaim,
) -> dict[str, object]:
    """Store and deliver a site notification to subscribed recipients.

    Args:
        req: Validated notification payload.
        db: Database session used for recipient and record operations.
        rds: Redis connection used for coordination and token cache access.

    Returns:
        Delivery outcome with batch statistics when execution completed.
    """
    user_ids = await get_site_notification_user_ids_cached(req.site, db, rds)
    if user_ids is None:
        return {
            'success': False,
            'outcome': 'permanent_failure',
            'message': f"Site '{req.site}' not found.",
        }
    if not user_ids:
        return {
            'success': True,
            'outcome': 'skipped',
            'message': f"Site '{req.site}' has no subscribed users.",
        }

    # Persist the in-app record before FCM; push delivery is an external
    # effect.
    record_count = await create_notification_records_for_users(
        req,
        user_ids,
        db,
    )
    await ensure_fcm_token_cache_for_users(user_ids, db, rds)
    delivery_state = FcmDeliveryState(db, rds)
    await delivery_state.capture(user_ids)
    event_delivery = FcmEventDelivery(delivery_state, claim)
    preparation_started_at = time.monotonic()
    token_stats = PushTokenStats()
    push_tasks = iter_push_tasks_streaming(
        req,
        user_ids,
        rds,
        token_stats=token_stats,
        build_push_task_fn=event_delivery.send_batch,
    )
    preparation_seconds = time.monotonic() - preparation_started_at
    delivery_started_at = time.monotonic()
    (
        ok,
        total_batches,
        successful_batches,
        error_message,
    ) = await execute_push_tasks_bounded_streaming(
        push_tasks,
        timeout=30.0,
        invalid_token_handler=(delivery_state.invalidate),
    )
    if not ok:
        if (
            error_message
            and error_message != 'FCM notification sending timed out.'
        ):
            logger.error('FCM sending failed: %s', error_message)
        message = (
            'FCM notification sending timed out.'
            if error_message == 'FCM notification sending timed out.'
            else 'Failed to send FCM notifications.'
        )
        return {'success': False, 'outcome': 'retryable', 'message': message}

    assert total_batches is not None
    assert successful_batches is not None
    stats: dict[str, object] = {
        'translation_time': preparation_seconds,
        'fcm_time': time.monotonic() - delivery_started_at,
        'total_batches': total_batches,
        'successful_batches': successful_batches,
        'notification_records': record_count,
    }
    if total_batches == 0:
        # The streaming dispatch already read these hashes; reuse its counts
        # instead of making a second full Redis pass on this failure path.
        preflight = preflight_from_token_stats(req, user_ids, token_stats)
        stats['preflight'] = preflight
        message = (
            f"Site '{req.site}' has no device tokens."
            if preflight['unique_tokens'] == 0
            else f"Site '{req.site}' has no sendable device tokens."
        )
        return {
            'success': True,
            'outcome': 'skipped',
            'message': message,
            'stats': stats,
        }
    return {
        'success': total_batches == successful_batches,
        'outcome': (
            'completed' if total_batches == successful_batches else 'retryable'
        ),
        'message': (
            'FCM notification processed. '
            f"{successful_batches}/{total_batches} batches succeeded."
        ),
        'stats': stats,
    }


async def send_test_notification(
    rds: redis.Redis,
    db: AsyncSession,
    me: User,
) -> TestNotificationResponse:
    """Send a test notification and record per-token delivery state.

    Args:
        rds: Redis connection used for token cache maintenance.
        db: Database session used for token state updates.
        me: Authenticated user receiving the test notification.

    Returns:
        Aggregated test-notification delivery result.
    """
    await ensure_fcm_token_cache_for_users([me.id], db, rds)
    delivery_state = FcmDeliveryState(db, rds)
    await delivery_state.capture([me.id])
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
    invalid_tokens = set(result.invalid_tokens)
    success = result.success_count > 0 and result.failure_count == 0
    if success:
        await delivery_state.success(device_tokens)
    elif result.failure_count > 0:
        await delivery_state.failure(device_tokens)
    if invalid_tokens:
        await delivery_state.invalidate(invalid_tokens)
    return TestNotificationResponse(
        success=success,
        message=(
            'Test notification sent.'
            if success
            else 'Test notification failed.'
        ),
        attempted_tokens=len(device_tokens),
        success_count=result.success_count,
        failure_count=result.failure_count,
        invalid_tokens=len(invalid_tokens),
    )
