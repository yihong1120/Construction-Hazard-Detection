from __future__ import annotations

import asyncio
import time
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
from examples.local_notification_server.lang_config import LANGUAGES
from examples.local_notification_server.lang_config import Translator
from examples.local_notification_server.schemas import SiteNotifyRequest
from examples.local_notification_server.schemas import TokenRequest

router: APIRouter = APIRouter()

# Cache mechanism for storing site user information
_site_users_cache: dict[str, tuple[list[User], float]] = {}
_cache_ttl: int = 300  # Cache time-to-live in seconds (5 minutes)


async def get_site_users_cached(
    site_name: str,
    db: AsyncSession,
) -> list[User] | None:
    """
    Retrieve the list of users for a site, with caching support.

    Args:
        site_name (str): The name of the site.
        db (AsyncSession): The database session.

    Returns:
        Union[list[User], None]:
            A list of users or None if the site does not exist.
    """
    current_time: float = time.time()

    # Check cache
    if site_name in _site_users_cache:
        cached_users, cached_time = _site_users_cache[site_name]
        if current_time - cached_time < _cache_ttl:
            return cached_users

    # Query the database
    stmt = (
        select(Site)
        .options(selectinload(Site.users))
        .where(Site.name == site_name)
    )
    site_obj: Site | None = (
        (await db.execute(stmt)).unique().scalar_one_or_none()
    )

    if not site_obj:
        return None

    site_users: list[User] = list(site_obj.users) if site_obj.users else []

    # Update cache
    _site_users_cache[site_name] = (site_users, current_time)

    return site_users


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

    pipe = rds.pipeline()
    key_to_userid: dict[str, int] = {}
    for user in users:
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

    start_time: float = time.time()
    push_tasks: list[Awaitable[bool]] = []

    # Batch process translations for efficiency
    translation_cache: dict[str, list[str]] = {}
    title_cache: dict[str, str] = {}

    for lang, tokens in lang_to_tokens.items():
        # Translate title
        if lang not in title_cache:
            lang_map = LANGUAGES.get(lang, LANGUAGES['en-GB'])
            title_cache[lang] = lang_map.get(
                'warning_notification', '[Warning Notification]',
            )

        # Translate content
        if lang not in translation_cache:
            translation_cache[lang] = Translator.translate_from_dict(
                req.body, lang,
            )

        title: str = title_cache[lang]
        translated_lines: list[str] = translation_cache[lang]
        body: str = (
            f"{req.site} - {req.stream_name}\n"
            + '\n'.join(translated_lines)
        )

        # Debug print for notification body
        print(f"lang: {lang}, tokens: {len(tokens)}, body: {body}")

        # Send tokens in batches to avoid FCM limits
        batch_size: int = 100  # Recommended batch size by FCM
        for i in range(0, len(tokens), batch_size):
            token_batch: list[str] = tokens[i:i + batch_size]
            push_tasks.append(
                send_fcm_notification_service(
                    device_tokens=token_batch,
                    title=title,
                    body=body,
                    image_path=req.image_path,
                    data={
                        'navigate': 'violation_list_page',
                        'violation_id': str(req.violation_id or ''),
                    },
                ),
            )

    translation_time: float = time.time() - start_time
    print(f"Translation and preparation time: {translation_time:.3f}s")

    fcm_start_time: float = time.time()
    try:
        # Set timeout to prevent long waits
        results: list[bool] = await asyncio.wait_for(
            asyncio.gather(*push_tasks, return_exceptions=False),
            timeout=30.0,  # 30 seconds timeout
        )
    except asyncio.TimeoutError:
        print('FCM notification sending timed out')
        return {
            'success': False,
            'message': 'FCM notification sending timed out.',
        }
    except Exception as e:
        print(f"Error during FCM notification sending: {e}")
        return {
            'success': False,
            'message': f'Error sending FCM notifications: {str(e)}',
        }

    fcm_time: float = time.time() - fcm_start_time
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
