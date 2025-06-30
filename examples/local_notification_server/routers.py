from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable
from typing import DefaultDict

import redis.asyncio as redis
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi_jwt import JwtAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

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
    # Query the user by ID
    stmt_user = select(User).where(User.id == req.user_id)
    result_user = await db.execute(stmt_user)
    user_obj: User | None = result_user.unique().scalar_one_or_none()
    if not user_obj:
        raise HTTPException(status_code=404, detail='User not found')

    # Store the device token and language in Redis
    key: str = f"fcm_tokens:{req.user_id}"
    device_lang: str = req.device_lang or 'en'
    await rds.hset(key, req.device_token, device_lang)

    return {'message': 'Token stored successfully.'}


@router.delete('/delete_token')
async def delete_fcm_token(
    req: TokenRequest,
    db: AsyncSession = Depends(get_db),
    rds: redis.Redis = Depends(get_redis_pool),
) -> dict[str, str]:
    """
    Delete an FCM device token from Redis.

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
    # Query the user by ID
    stmt_user = select(User).where(User.id == req.user_id)
    result_user = await db.execute(stmt_user)
    user_obj: User | None = result_user.unique().scalar_one_or_none()
    if not user_obj:
        # User not found in the database
        # Return a message instead of raising an exception
        # to avoid exposing user information.
        return {'message': 'User not found.'}

    # Attempt to remove the device token from Redis
    key: str = f"fcm_tokens:{req.user_id}"
    removed: int = await rds.hdel(key, req.device_token)
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
    # ---------- 0. Early exit if body is empty ----------
    if not req.body:
        return {'success': False, 'message': 'Body is empty, nothing to send.'}

    # ---------- 1. Query Site and Users ----------
    stmt = (
        select(Site)
        .options(selectinload(Site.users))
        .where(Site.name == req.site)
    )
    site_obj: Site | None = (
        (await db.execute(stmt)).unique().scalar_one_or_none()
    )
    if not site_obj or not site_obj.users:
        return {
            'success': False,
            'message': (
                f"Site '{req.site}' not found or has no users."
            ),
        }

    # ---------- 2. Batch fetch Redis tokens using pipeline ----------
    pipe = rds.pipeline()
    key_to_userid: dict[str, int] = {}
    for user in site_obj.users:
        key = f"fcm_tokens:{user.id}"
        key_to_userid[key] = user.id
        pipe.hgetall(key)  # Add hgetall command to pipeline (not awaited)

    # Single round-trip
    redis_results: list[dict[bytes, bytes]] = await pipe.execute()
    lang_to_tokens: DefaultDict[str, list[str]] = defaultdict(list)

    # Convert bytes to str and group tokens by language
    for raw_map in redis_results:
        for token_b, lang_b in raw_map.items():
            token: str = token_b.decode()
            lang: str = (lang_b.decode() or 'en-GB')
            lang_to_tokens[lang].append(token)

    if not lang_to_tokens:
        return {
            'success': False,
            'message': f"Site '{req.site}' has no device tokens.",
        }

    # ---------- 3. Prepare notification content ----------
    push_tasks: list[Awaitable[bool]] = []
    title: str = '[警示通知]'
    for lang, tokens in lang_to_tokens.items():
        translated_lines: list[str] = Translator.translate_from_dict(
            req.body, lang,
        )
        body: str = f"{req.site} - {req.stream_name}\n" + \
            '\n'.join(translated_lines)
        # Debug print for notification body
        print(f"lang: {lang}, body: {body}")
        push_tasks.append(
            send_fcm_notification_service(
                device_tokens=tokens,
                title=title,
                body=body,
                image_path=req.image_path,
                data={
                    'navigate': 'violation_list_page',
                    'violation_id': str(req.violation_id or ''),
                },
            ),
        )

    # ---------- 4. Send FCM notifications in parallel ----------
    results: list[bool] = await asyncio.gather(
        *push_tasks, return_exceptions=False,
    )
    overall_success: bool = all(results)

    return {
        'success': overall_success,
        'message': 'FCM notification has been processed.',
    }
