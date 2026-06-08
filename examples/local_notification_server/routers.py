from __future__ import annotations

import hashlib
import json
import time
from typing import cast

import redis.asyncio as redis
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Security
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.models import Site
from examples.auth.models import SiteNotificationPreference
from examples.auth.models import User
from examples.auth.redis_pool import get_redis_pool
from examples.auth.user_service import list_effective_sites_for_user
from examples.db_management.deps import get_current_user
from examples.db_management.deps import is_super_admin
from examples.db_management.services.site_services import list_sites
from examples.local_notification_server.lang_config import normalize_language
from examples.local_notification_server.schemas import \
    SiteNotificationPreferenceOut
from examples.local_notification_server.schemas import \
    SiteNotificationPreferenceUpdateRequest
from examples.local_notification_server.schemas import SiteNotifyRequest
from examples.local_notification_server.schemas import TokenRequest
from examples.local_notification_server.services import \
    _execute_push_tasks_bounded_streaming
from examples.local_notification_server.services import \
    _iter_push_tasks_streaming
from examples.local_notification_server.services import \
    get_site_notification_user_ids_cached
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


@router.post('/store_token')
async def store_fcm_token(
    req: TokenRequest,
    db: AsyncSession = Depends(get_db),
    rds: redis.Redis = Depends(get_redis_pool),
) -> dict[str, str]:
    """Store an FCM device token in Redis.

    A Redis hash is used to store token-language pairs:
    - Key: "fcm_tokens:{user_id}"
    - Field: Device token
    - Value: Language code (e.g., 'en-GB')

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

    key: str = f"fcm_tokens:{req.user_id}"
    device_lang = normalize_language(req.device_lang)
    if device_lang is None:
        raise HTTPException(status_code=422, detail='unsupported_device_lang')

    # Use Redis pipeline for batch operations
    pipe = rds.pipeline()

    # Set token and expiration
    pipe.hset(key, req.device_token, device_lang)
    pipe.expire(key, 86400 * 30)  # 30 days expiration

    await pipe.execute()

    return {'message': 'Token stored successfully.'}


@router.delete('/delete_token')
async def delete_fcm_token(
    req: TokenRequest,
    db: AsyncSession = Depends(get_db),
    rds: redis.Redis = Depends(get_redis_pool),
) -> dict[str, str]:
    """Delete an FCM device token from Redis.

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

    # Use Redis pipeline for batch operations
    pipe = rds.pipeline()
    key: str = f"fcm_tokens:{req.user_id}"

    pipe.hdel(key, req.device_token)
    pipe.hlen(key)  # Check remaining token count

    results = await pipe.execute()
    removed: int = results[0]
    remaining_tokens: int = results[1]

    # Delete key if no tokens remain
    if remaining_tokens == 0:
        await rds.delete(key)

    if removed == 0:
        return {'message': 'Token not found in Redis hash.'}

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

    start_time: float = time.time()
    push_tasks = _iter_push_tasks_streaming(req, user_ids, rds)
    translation_time: float = time.time() - start_time
    print(f"Notification preparation time: {translation_time:.3f}s")

    fcm_start_time: float = time.time()

    ok, total_batches, successful_batches, error_msg = (
        await _execute_push_tasks_bounded_streaming(
            push_tasks,
            timeout=30.0,
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
        print(
            'FCM sending skipped: no sendable batches; '
            f"site={req.site!r}, stream={req.stream_name!r}, "
            f"subscribed_users={len(user_ids)}. "
            'Check fcm_tokens, token languages, and notification body keys.',
        )
        return {
            'success': False,
            'message': f"Site '{req.site}' has no device tokens.",
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
        },
    }


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
