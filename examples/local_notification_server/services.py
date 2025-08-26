from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from collections.abc import Awaitable
from typing import DefaultDict
from typing import Final

import redis.asyncio as redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from examples.auth.models import Site
from examples.auth.models import User
from examples.local_notification_server.fcm_service import (
    send_fcm_notification_service,
)
from examples.local_notification_server.lang_config import LANGUAGES
from examples.local_notification_server.lang_config import Translator
from examples.local_notification_server.schemas import SiteNotifyRequest

# In-memory cache for site users to avoid repeated database lookups within a
# short time window. Key is the site name, value is a tuple of (users, epoch).
_site_users_cache: dict[str, tuple[list[User], float]] = {}

# Cache time-to-live (seconds). Keep small to minimise staleness.
_cache_ttl: Final[int] = 300  # 5 minutes


async def get_site_users_cached(
    site_name: str,
    db: AsyncSession,
) -> list[User] | None:
    """
    Get users of a site with a tiny in-memory cache.

    Args:
        site_name: The site name to look up.
        db: An async SQLAlchemy session used to query the database.

    Returns:
        A list of users if the site exists; otherwise ``None`` if the site is
        not found.
    """
    current_time: float = time.time()

    # Check cache
    if site_name in _site_users_cache:
        cached_users, cached_time = _site_users_cache[site_name]
        if current_time - cached_time < _cache_ttl:
            return cached_users

    # Query the database when cache is missing or stale.
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

    # Update cache with fresh users and retrieval timestamp.
    _site_users_cache[site_name] = (site_users, current_time)

    return site_users


def _decode_lang_token_map(
    raw_maps: list[dict[bytes, bytes]],
) -> DefaultDict[str, list[str]]:
    """
    Decode Redis HGETALL results into a language-to-tokens map.

    Args:
        raw_maps: A list of byte dictionaries from Redis, each representing a
            user's token-to-language mapping.

    Returns:
        A mapping from BCP 47 language code to a list of device tokens.
    """
    lang_to_tokens: DefaultDict[str, list[str]] = defaultdict(list)
    for raw_map in raw_maps:
        for token_b, lang_b in raw_map.items():
            token: str = token_b.decode()
            lang: str = (lang_b.decode() or 'en-GB')
            lang_to_tokens[lang].append(token)
    return lang_to_tokens


async def _get_lang_to_tokens(
    users: list[User], rds: redis.Redis,
) -> DefaultDict[str, list[str]]:
    """
    Fetch device tokens for users and group them by language.

    Args:
        users: The users to fetch tokens for.
        rds: An asyncio Redis client instance.

    Returns:
        A mapping from language code to the list of tokens in that language.
    """
    pipe = rds.pipeline()
    for user in users:
        key = f"fcm_tokens:{user.id}"
        pipe.hgetall(key)
    redis_results: list[dict[bytes, bytes]] = await pipe.execute()
    return _decode_lang_token_map(redis_results)


def _translate_title(lang: str) -> str:
    """
    Translate notification title by language with a sensible default.

    Args:
        lang: A BCP 47 language tag.

    Returns:
        The translated title string. Falls back to English (GB) phrase when the
        language is unknown.
    """
    lang_map = LANGUAGES.get(lang, LANGUAGES['en-GB'])
    return lang_map.get('warning_notification', '[Warning Notification]')


def _translate_body_lines(
    body_dict: dict[str, dict[str, int]],
    lang: str,
) -> list[str]:
    """
    Translate body lines using the given language.

    Args:
        body_dict: Mapping from language to message spec dictionary.
        lang: Target language to translate into.

    Returns:
        A list of translated message lines.
    """
    return Translator.translate_from_dict(body_dict, lang)


def _build_push_tasks(
    req: SiteNotifyRequest,
    lang_to_tokens: DefaultDict[str, list[str]],
) -> list[Awaitable[bool]]:
    """
    Build push tasks for sending notifications, batching tokens as needed.

    Args:
        req: Validated site notification request.
        lang_to_tokens: Mapping of language codes to device tokens.

    Returns:
        A list of awaitable tasks (each returns ``True`` on success, ``False``
        otherwise) ready to be awaited by the caller.
    """
    push_tasks: list[Awaitable[bool]] = []

    # Batch process translations for efficiency (simple caches)
    translation_cache: dict[str, list[str]] = {}
    title_cache: dict[str, str] = {}

    for lang, tokens in lang_to_tokens.items():
        if lang not in title_cache:
            title_cache[lang] = _translate_title(lang)

        if lang not in translation_cache:
            translation_cache[lang] = _translate_body_lines(req.body, lang)

        title: str = title_cache[lang]
        translated_lines: list[str] = translation_cache[lang]
        body: str = f"{req.site} - {req.stream_name}\n" + \
            '\n'.join(translated_lines)

        # Debug print for notification body (useful for local diagnostics).
        print(f"lang: {lang}, tokens: {len(tokens)}, body: {body}")

        # Send tokens in batches to avoid FCM limits and request timeouts.
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

    return push_tasks


async def _execute_push_tasks(
    push_tasks: list[Awaitable[bool]], timeout: float = 30.0,
) -> tuple[bool, list[bool] | None, str | None]:
    """
    Execute push tasks with a timeout and return results.

    Args:
        push_tasks: List of awaitable tasks created by ``_build_push_tasks``.
        timeout: Maximum time in seconds to wait for all tasks to complete.

    Returns:
        A tuple ``(ok, results, error_message)`` where:
        - ``ok`` is ``True`` when execution completes without timeout or
          unexpected exception.
        - ``results`` is a list of booleans for each batch when ``ok`` is
          ``True``; otherwise ``None``.
        - ``error_message`` contains a user-safe message when ``ok`` is
          ``False``; otherwise ``None``.
    """
    try:
        results: list[bool] = await asyncio.wait_for(
            asyncio.gather(*push_tasks, return_exceptions=False),
            timeout=timeout,
        )
        return True, results, None
    except asyncio.TimeoutError:
        # Return a generic timeout message (safe to surface to clients).
        return False, None, 'FCM notification sending timed out.'
    except Exception:
        # Do not surface internal exception details to clients.
        # Return a generic error indicator; log details at the call site.
        return False, None, 'internal_error'
