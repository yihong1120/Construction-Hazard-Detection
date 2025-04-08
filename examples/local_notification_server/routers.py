from __future__ import annotations

from typing import Any

import redis.asyncio as redis
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi_jwt import JwtAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.models import Site
from examples.auth.models import User
from examples.auth.redis_pool import get_redis_pool
from examples.local_notification_server.fcm_service import (
    send_fcm_notification_service,
)
from examples.local_notification_server.lang_config import Translator
from examples.local_notification_server.schemas import SiteNotifyRequest
from examples.local_notification_server.schemas import TokenRequest

router = APIRouter()


@router.post('/store_token')
async def store_fcm_token(
    req: TokenRequest,
    db: AsyncSession = Depends(get_db),
    rds: redis.Redis = Depends(get_redis_pool),
) -> dict[str, str]:
    """Store an FCM device token in Redis.

    A Redis hash is used to store token-language pairs:
    - key: "fcm_tokens:{user_id}"
    - field: device token
    - value: language code (e.g. 'en-GB')

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
    stmt_user = select(User).where(User.id == req.user_id)
    result_user = await db.execute(stmt_user)
    user_obj = result_user.scalar_one_or_none()
    if not user_obj:
        raise HTTPException(status_code=404, detail='User not found')

    key = f"fcm_tokens:{req.user_id}"
    device_lang = req.device_lang or 'en'
    await rds.hset(key, req.device_token, device_lang)

    return {'message': 'Token stored successfully.'}


@router.delete('/delete_token')
async def delete_fcm_token(
    req: TokenRequest,
    db: AsyncSession = Depends(get_db),
    rds: redis.Redis = Depends(get_redis_pool),
) -> dict[str, str]:
    """Delete an FCM device token from Redis.

    A Redis hash is used to store token-language pairs. If the user or token
    does not exist, the method returns an informational message.

    Args:
        req (TokenRequest):
            Data model containing the user's ID and device token.
        db (AsyncSession):
            Async database session dependency for querying user data.
        rds (redis.Redis):
            Redis connection dependency for removing token data.

    Returns:
        dict[str, str]:
            A message indicating whether the user does not exist, the token
            was not found in Redis, or it was successfully deleted.
    """
    stmt_user = select(User).where(User.id == req.user_id)
    result_user = await db.execute(stmt_user)
    user_obj = result_user.scalar_one_or_none()
    if not user_obj:
        # User not found in the database
        # Return a message instead of raising an exception
        # to avoid exposing user information.
        return {'message': 'User not found.'}

    key = f"fcm_tokens:{req.user_id}"
    removed = await rds.hdel(key, req.device_token)
    if removed == 0:
        return {'message': 'Token not found in Redis hash.'}

    return {'message': 'Token deleted.'}


@router.post('/send_fcm_notification')
async def send_fcm_notification(
    req: SiteNotifyRequest,
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Depends(jwt_access),
    rds: redis.Redis = Depends(get_redis_pool),
) -> dict[str, Any]:
    """Send an FCM notification to all users associated with a specified site.

    Steps:
      1. Query the database for the Site matching req.site.
      2. Gather all associated users. If none, return a descriptive message.
      3. For each user, fetch tokens and language preferences from Redis.
      4. Group tokens by language and call the translation service.
      5. Invoke the FCM sending service for each language group.

    Args:
        req (SiteNotifyRequest):
            Data model containing the site name, stream name, violation ID,
            body (message content), and optional image path.
        db (AsyncSession):
            Async database session dependency for querying site and user data.
        credentials (JwtAuthorizationCredentials):
            Decoded JWT credentials with user info. (e.g., jti, sub).
        rds (redis.Redis):
            Redis connection dependency for retrieving user tokens.

    Returns:
        dict[str, Any]:
            A dictionary containing:
            {
                "success": bool,
                "message": "FCM notification has been processed."
            }
            If the site does not exist or has no tokens, "success" is False.
    """
    stmt = select(Site).where(Site.name == req.site)
    res_site = await db.execute(stmt)
    site_obj = res_site.scalar_one_or_none()
    if not site_obj:
        return {'success': False, 'message': f"Site '{req.site}' not found."}

    await db.refresh(site_obj, ['users'])
    if not site_obj.users:
        return {
            'success': False,
            'message': f"Site '{req.site}' has no users.",
        }

    # Collect user tokens by language: {lang: [token, ...]}
    lang_to_tokens: dict[str, list[str]] = {}
    for user in site_obj.users:
        key = f"fcm_tokens:{user.id}"
        # Each entry is {token_bytes: lang_bytes}
        tokens_map = await rds.hgetall(key)
        decoded_map = {
            token_bytes.decode('utf-8'): lang_bytes.decode('utf-8')
            for token_bytes, lang_bytes in tokens_map.items()
        }

        if tokens_map:
            for token, lang in decoded_map.items():
                lang_to_tokens.setdefault(lang, []).append(token)

    if not lang_to_tokens:
        return {
            'success': False,
            'message': f"Site '{req.site}' has no user tokens in Redis.",
        }

    overall_success = True

    # Perform the FCM send operation for each language group
    for lang, tokens in lang_to_tokens.items():
        translated_messages = Translator.translate_from_dict(req.body, lang)
        title_str = '[警示通知]'
        # title_str = (
        #    Translator.LANGUAGES[lang].get(
        #       "warning_notification", "[Warning Notification]")
        #    )
        # )
        message_str = f"{req.site} - {req.stream_name}\n" + \
            '\n'.join(translated_messages)

        sent_success = await send_fcm_notification_service(
            device_tokens=tokens,
            title=title_str,
            body=message_str,
            image_path=req.image_path,
            data={
                'navigate': 'violation_list_page',
                'violation_id': str(req.violation_id or ''),
            },
        )
        if not sent_success:
            overall_success = False

    return {
        'success': overall_success,
        'message': 'FCM notification has been processed.',
    }
