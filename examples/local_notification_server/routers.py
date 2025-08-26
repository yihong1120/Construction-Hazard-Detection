from __future__ import annotations

import time
from collections.abc import Awaitable
from typing import DefaultDict

import redis.asyncio as redis
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi_jwt import JwtAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.models import User
from examples.auth.redis_pool import get_redis_pool
from examples.local_notification_server.schemas import SiteNotifyRequest
from examples.local_notification_server.schemas import TokenRequest
from examples.local_notification_server.services import _build_push_tasks
from examples.local_notification_server.services import _execute_push_tasks
from examples.local_notification_server.services import _get_lang_to_tokens
from examples.local_notification_server.services import get_site_users_cached

router: APIRouter = APIRouter()


@router.post('/store_token')
async def store_fcm_token(
    req: TokenRequest,
    db: AsyncSession = Depends(get_db),
    rds: redis.Redis = Depends(get_redis_pool),
) -> dict[str, str]:
    """
    Store an FCM device token in Redis.

    A Redis hash is used to store token-language pairs:
    - Key: "fcm_tokens:{user_id}"
    - Field: Device token
    - Value: Language code (e.g., 'en-GB')

    Args:
        req (TokenRequest):
            Data model containing the user's ID and device token.
        db (AsyncSession):
            Async database session dependency for querying user data.
        rds (redis.Redis):
            Redis connection dependency for storing and retrieving token data.

    Raises:
        HTTPException:
            Raised with status code 404 if the specified user does not exist.

    Returns:
        dict[str, str]: A success message indicating the token was stored.
    """
    # Validate user existence
    stmt_user = select(User.id).where(User.id == req.user_id)
    result = await db.execute(stmt_user)
    if not result.scalar():
        raise HTTPException(status_code=404, detail='User not found')

    # Use Redis pipeline for batch operations
    pipe = rds.pipeline()
    key: str = f"fcm_tokens:{req.user_id}"
    device_lang: str = req.device_lang or 'en'

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
    """
    Delete an FCM device token from Redis with optimised validation.

    Args:
        req (TokenRequest):
            Data model containing the user's ID and device token.
        db (AsyncSession):
            Async database session dependency for querying user data.
        rds (redis.Redis):
            Redis connection dependency for storing and retrieving token data.

    Returns:
        dict[str, str]: A message indicating the result of the deletion.
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
    _cred: JwtAuthorizationCredentials = Depends(jwt_access),
    rds: redis.Redis = Depends(get_redis_pool),
) -> dict[str, object]:
    """
    Send a Firebase Cloud Messaging (FCM) notification to all users of a site.

    Args:
        req (SiteNotifyRequest):
            The notification request, including site, stream name, body, image
            path, and violation ID.
        db (AsyncSession):
            Async database session dependency for querying site and user data.
        _cred (JwtAuthorizationCredentials):
            JWT credentials for authentication (not used directly).
        rds (redis.Redis):
            Redis connection dependency for retrieving device tokens.

    Returns:
        dict[str, object]:
            A dictionary indicating success and a message about
            the notification process.
    """
    if not req.body:
        return {'success': False, 'message': 'Body is empty, nothing to send.'}

    users: list[User] | None = await get_site_users_cached(req.site, db)
    if not users:
        return {
            'success': False,
            'message': f"Site '{req.site}' not found or has no users.",
        }

    # Group tokens by language
    lang_to_tokens: DefaultDict[str, list[str]] = await _get_lang_to_tokens(
        users, rds,
    )

    if not lang_to_tokens:
        return {
            'success': False,
            'message': f"Site '{req.site}' has no device tokens.",
        }

    start_time: float = time.time()
    push_tasks: list[Awaitable[bool]] = _build_push_tasks(req, lang_to_tokens)
    translation_time: float = time.time() - start_time
    print(f"Translation and preparation time: {translation_time:.3f}s")

    fcm_start_time: float = time.time()
    ok, results, error_msg = await _execute_push_tasks(
        push_tasks, timeout=30.0,
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
    assert results is not None
    overall_success: bool = all(results)
    successful_batches: int = sum(results)

    print(
        f"FCM sending time: {fcm_time:.3f}s, successful batches: "
        f"{successful_batches}/{len(results)}",
    )

    return {
        'success': overall_success,
        'message': (
            f'FCM notification processed. '
            f'{successful_batches}/{len(results)} batches succeeded.'
        ),
        'stats': {
            'translation_time': translation_time,
            'fcm_time': fcm_time,
            'total_batches': len(results),
            'successful_batches': successful_batches,
        },
    }
