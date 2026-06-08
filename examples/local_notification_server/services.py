from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterable
from collections.abc import AsyncIterator
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from typing import cast
from typing import DefaultDict
from typing import Final

import redis.asyncio as redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.models import Site
from examples.auth.models import SiteNotificationPreference
from examples.auth.models import User
from examples.auth.models import USER_STATUS_ACTIVE
from examples.local_notification_server.fcm_service import (
    FcmSendResult,
)
from examples.local_notification_server.fcm_service import (
    send_fcm_notification_service,
)
from examples.local_notification_server.lang_config import LANGUAGES
from examples.local_notification_server.lang_config import normalize_language
from examples.local_notification_server.lang_config import Translator
from examples.local_notification_server.schemas import SiteNotifyRequest
from src.warning_types import Warnings

# Redis recipient index settings.
_recipient_index_ready_value: Final[str] = '1'
_recipient_index_lock_seconds: Final[int] = 30
_recipient_index_wait_attempts: Final[int] = 5
_recipient_index_wait_seconds: Final[float] = 0.05
_token_fetch_chunk_size: Final[int] = 500
_fcm_batch_size: Final[int] = 100
_fcm_max_concurrency: Final[int] = 8

PushTaskResult = bool | FcmSendResult


def _site_user_cache_key(site_name: str) -> str:
    """Build the Redis set key for site notification recipients.

    Args:
        site_name: Site name used by notification requests.

    Returns:
        Redis set key containing recipient user IDs.
    """
    return f'site_notification_users:{site_name}'


def _site_user_cache_ready_key(site_name: str) -> str:
    """Build the Redis readiness key for a site recipient index.

    Args:
        site_name: Site name used by notification requests.

    Returns:
        Redis key indicating that the recipient index is ready.
    """
    return f'site_notification_users_ready:{site_name}'


def _site_user_cache_lock_key(site_name: str) -> str:
    """Build the Redis lock key used while rebuilding a site index.

    Args:
        site_name: Site name used by notification requests.

    Returns:
        Redis lock key for recipient index rebuilds.
    """
    return f'site_notification_users_lock:{site_name}'


async def _fetch_site_notification_user_ids_from_db(
    site_name: str,
    db: AsyncSession,
) -> list[int] | None:
    """Load current recipient user IDs for a site from the database.

    Args:
        site_name: Site name to look up.
        db: Async database session dependency.

    Returns:
        Active recipient user IDs, or None when the site does not exist.
    """
    stmt = select(Site.id).where(Site.name == site_name)
    site_id_row = (await db.execute(stmt)).first()
    if site_id_row is None:
        return None
    site_id = site_id_row[0]

    users_stmt = (
        select(SiteNotificationPreference.user_id)
        .join(User, User.id == SiteNotificationPreference.user_id)
        .where(
            SiteNotificationPreference.site_id == site_id,
            SiteNotificationPreference.is_enabled.is_(True),
            User.status == USER_STATUS_ACTIVE,
        )
    )
    return list((await db.execute(users_stmt)).scalars().all())


async def refresh_site_notification_user_cache(
    site_name: str,
    db: AsyncSession,
    rds: redis.Redis,
) -> list[int] | None:
    """Rebuild the Redis recipient index for a site from the database.

    Args:
        site_name: Site name to rebuild.
        db: Async database session dependency.
        rds: Redis connection used to write the recipient index.

    Returns:
        Active recipient user IDs, or None when the site does not exist.
    """
    user_ids = await _fetch_site_notification_user_ids_from_db(site_name, db)
    if user_ids is None:
        await invalidate_site_notification_user_cache([site_name], rds)
        return None

    pipe = rds.pipeline()
    cache_key = _site_user_cache_key(site_name)
    ready_key = _site_user_cache_ready_key(site_name)
    pipe.delete(cache_key)
    if user_ids:
        pipe.sadd(cache_key, *user_ids)
    pipe.set(ready_key, _recipient_index_ready_value)
    await pipe.execute()
    return user_ids


async def _get_site_user_index_members(
    site_name: str,
    rds: redis.Redis,
) -> list[int]:
    """Read recipient IDs from the Redis set for a site.

    Args:
        site_name: Site name to read.
        rds: Redis connection used to read the recipient index.

    Returns:
        Recipient user IDs from Redis.
    """
    members = cast(
        Awaitable[set[bytes | str]],
        rds.smembers(_site_user_cache_key(site_name)),
    )
    return [int(member) for member in await members]


async def invalidate_site_notification_user_cache(
    site_names: list[str],
    rds: redis.Redis,
) -> None:
    """Delete Redis recipient indexes for the given sites.

    Args:
        site_names: Site names whose indexes should be removed.
        rds: Redis connection used to delete cache keys.
    """
    keys: list[str] = []
    for site_name in site_names:
        keys.extend([
            _site_user_cache_key(site_name),
            _site_user_cache_ready_key(site_name),
            _site_user_cache_lock_key(site_name),
        ])
    if keys:
        await rds.delete(*keys)


async def get_site_notification_user_ids_cached(
    site_name: str,
    db: AsyncSession,
    rds: redis.Redis,
) -> list[int] | None:
    """
    Get notification recipient user IDs for a site using a Redis index.

    Args:
        site_name: The site name to look up.
        db: An async SQLAlchemy session used for cold rebuilds.
        rds: Redis connection used as the live recipient index.

    Returns:
        A list of user IDs if the site exists; otherwise ``None``.
    """
    ready_key = _site_user_cache_ready_key(site_name)
    lock_key = _site_user_cache_lock_key(site_name)

    if await rds.exists(ready_key):
        return await _get_site_user_index_members(site_name, rds)

    lock_acquired = await rds.set(
        lock_key,
        _recipient_index_ready_value,
        ex=_recipient_index_lock_seconds,
        nx=True,
    )
    if lock_acquired:
        try:
            return await refresh_site_notification_user_cache(
                site_name, db, rds,
            )
        finally:
            await rds.delete(lock_key)

    for _ in range(_recipient_index_wait_attempts):
        await asyncio.sleep(_recipient_index_wait_seconds)
        if await rds.exists(ready_key):
            return await _get_site_user_index_members(site_name, rds)

    return await refresh_site_notification_user_cache(site_name, db, rds)


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
    seen_tokens: set[str] = set()
    for raw_map in raw_maps:
        for token_b, lang_b in raw_map.items():
            token: str = token_b.decode()
            if token in seen_tokens:
                continue
            seen_tokens.add(token)
            lang = normalize_language(lang_b.decode() if lang_b else None)
            if lang is None:
                continue
            lang_to_tokens[lang].append(token)
    return lang_to_tokens


async def _get_lang_to_tokens(
    user_ids: list[int], rds: redis.Redis,
) -> DefaultDict[str, list[str]]:
    """
    Fetch device tokens for users and group them by language.

    Args:
        user_ids: The user IDs to fetch tokens for.
        rds: An asyncio Redis client instance.

    Returns:
        A mapping from language code to the list of tokens in that language.
    """
    lang_to_tokens: DefaultDict[str, list[str]] = defaultdict(list)
    for start in range(0, len(user_ids), _token_fetch_chunk_size):
        pipe = rds.pipeline()
        for user_id in user_ids[start:start + _token_fetch_chunk_size]:
            pipe.hgetall(f'fcm_tokens:{user_id}')
        redis_results: list[dict[bytes, bytes]] = await pipe.execute()
        chunk_tokens = _decode_lang_token_map(redis_results)
        for lang, tokens in chunk_tokens.items():
            lang_to_tokens[lang].extend(tokens)
    return lang_to_tokens


def _translate_title(lang: str) -> str:
    """
    Translate notification title by language.

    Args:
        lang: A BCP 47 language tag.

    Returns:
        The translated title string, or an empty string for unsupported
        languages.
    """
    language = normalize_language(lang)
    if language is None:
        return ''
    return LANGUAGES[language].get('warning_notification', '')


def _translate_body_lines(
    body_dict: Warnings,
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


def _iter_push_tasks(
    req: SiteNotifyRequest,
    lang_to_tokens: DefaultDict[str, list[str]],
) -> Iterator[Awaitable[PushTaskResult]]:
    """
    Yield push tasks for sending notifications, batching tokens as needed.

    Args:
        req: Validated site notification request.
        lang_to_tokens: Mapping of language codes to device tokens.

    Returns:
        Awaitable tasks (each returns ``True`` on success, ``False``
        otherwise), yielded one batch at a time.
    """
    for lang, tokens in lang_to_tokens.items():
        for i in range(0, len(tokens), _fcm_batch_size):
            task = _build_push_task(req, lang, tokens[i:i + _fcm_batch_size])
            if task is not None:
                yield task


def _notification_data(req: SiteNotifyRequest) -> dict[str, str]:
    """Build stable FCM data fields for notification navigation.

    Args:
        req: Validated notification request.

    Returns:
        String-only FCM data payload.
    """
    return {
        'navigate': 'violation_list_page',
        'violation_id': str(req.violation_id or ''),
    }


def _build_push_task(
    req: SiteNotifyRequest,
    lang: str,
    tokens: list[str],
) -> Awaitable[PushTaskResult] | None:
    """Build one FCM batch task for a supported language.

    Args:
        req: Validated notification request.
        lang: Language code for the target tokens.
        tokens: Device tokens in this batch.

    Returns:
        Awaitable send task, or None when the language/body cannot be sent.
    """
    if not tokens:
        return None

    language = normalize_language(lang)
    if language is None:
        print(
            f"FCM notification skipped: unsupported language {lang!r}, "
            f"tokens: {len(tokens)}",
        )
        return None

    title: str = _translate_title(language)
    translated_lines: list[str] = _translate_body_lines(req.body, language)
    if not title or not translated_lines:
        print(
            'FCM notification skipped: no translated notification lines '
            f"for language {language}, body keys: {list(req.body.keys())}",
        )
        return None

    body: str = f"{req.site} - {req.stream_name}\n" + \
        '\n'.join(translated_lines)

    # Debug print for notification body (useful for local diagnostics).
    print(f"lang: {language}, tokens: {len(tokens)}, body: {body}")

    return send_fcm_notification_service(
        device_tokens=tokens,
        title=title,
        body=body,
        image_path=req.image_path,
        data=_notification_data(req),
    )


def _push_task_succeeded(result: PushTaskResult) -> bool:
    """Return whether a push task result represents success.

    Args:
        result: Result returned by a push task.

    Returns:
        True when the push task succeeded.
    """
    return bool(result)


def _invalid_tokens_from_push_result(
    result: PushTaskResult,
) -> tuple[str, ...]:
    """Return invalid FCM tokens collected from a push result.

    Args:
        result: Result returned by a push task.

    Returns:
        Invalid tokens reported by Firebase.
    """
    if isinstance(result, FcmSendResult):
        return result.invalid_tokens
    return ()


async def _iter_push_tasks_streaming(
    req: SiteNotifyRequest,
    user_ids: list[int],
    rds: redis.Redis,
) -> AsyncIterator[Awaitable[PushTaskResult]]:
    """Stream Redis token chunks into FCM batch tasks.

    This avoids materialising all device tokens in a single
    ``lang_to_tokens`` map. Memory is bounded by the Redis chunk, one partial
    FCM batch per active language, and the executor's active tasks.

    Args:
        req: Validated notification request.
        user_ids: Recipient user IDs to fetch tokens for.
        rds: Redis connection used to read token hashes.

    Yields:
        Awaitable FCM batch send tasks.
    """
    pending_batches: DefaultDict[str, list[str]] = defaultdict(list)

    for start in range(0, len(user_ids), _token_fetch_chunk_size):
        pipe = rds.pipeline()
        for user_id in user_ids[start:start + _token_fetch_chunk_size]:
            pipe.hgetall(f'fcm_tokens:{user_id}')

        redis_results: list[dict[bytes, bytes]] = await pipe.execute()
        chunk_tokens = _decode_lang_token_map(redis_results)

        for lang, tokens in chunk_tokens.items():
            batch = pending_batches[lang]
            for token in tokens:
                batch.append(token)
                if len(batch) >= _fcm_batch_size:
                    task = _build_push_task(req, lang, list(batch))
                    batch.clear()
                    if task is not None:
                        yield task

    for lang, tokens in pending_batches.items():
        if not tokens:
            continue
        task = _build_push_task(req, lang, list(tokens))
        if task is not None:
            yield task


def _build_push_tasks(
    req: SiteNotifyRequest,
    lang_to_tokens: DefaultDict[str, list[str]],
) -> list[Awaitable[PushTaskResult]]:
    """Build push tasks for compatibility with existing direct callers.

    Request handlers should prefer ``_iter_push_tasks_streaming`` with
    ``_execute_push_tasks_bounded_streaming`` so large recipient lists do not
    materialize every device token or FCM coroutine at once.

    Args:
        req: Validated notification request.
        lang_to_tokens: Tokens grouped by language.

    Returns:
        Awaitable FCM batch send tasks.
    """
    return list(_iter_push_tasks(req, lang_to_tokens))


async def _execute_push_tasks_bounded(
    push_tasks: Iterable[Awaitable[PushTaskResult]],
    timeout: float = 30.0,
    max_concurrency: int = _fcm_max_concurrency,
    invalid_token_handler: (
        Callable[[tuple[str, ...]], Awaitable[object]] | None
    ) = None,
) -> tuple[bool, int | None, int | None, str | None]:
    """Execute push tasks with bounded concurrency and aggregate counts.

    Args:
        push_tasks: Finite iterable of awaitable FCM batch send tasks.
        timeout: Maximum execution time in seconds.
        max_concurrency: Maximum number of active send tasks.
        invalid_token_handler: Optional callback receiving invalid tokens.

    Returns:
        Tuple of `(ok, total_batches, successful_batches, error_message)`.
    """
    pending: set[asyncio.Future[PushTaskResult]] = set()

    async def run_window() -> tuple[int, int]:
        """Run the bounded task window for a finite iterable."""
        total_batches = 0
        successful_batches = 0
        invalid_tokens: set[str] = set()
        task_iter = iter(push_tasks)

        def schedule_next() -> bool:
            """Schedule the next task when the window has capacity."""
            try:
                awaitable = next(task_iter)
            except StopIteration:
                return False
            pending.add(asyncio.ensure_future(awaitable))
            return True

        for _ in range(max(1, max_concurrency)):
            if not schedule_next():
                break

        try:
            while pending:
                done, _ = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                pending.difference_update(done)
                for task in done:
                    total_batches += 1
                    result = await task
                    successful_batches += int(_push_task_succeeded(result))
                    invalid_tokens.update(
                        _invalid_tokens_from_push_result(result),
                    )
                while len(pending) < max(1, max_concurrency):
                    if not schedule_next():
                        break
        finally:
            for task in pending:
                task.cancel()

        if invalid_tokens and invalid_token_handler is not None:
            await invalid_token_handler(tuple(sorted(invalid_tokens)))

        return total_batches, successful_batches

    window = run_window()
    try:
        total, successful = await asyncio.wait_for(window, timeout=timeout)
        return True, total, successful, None
    except asyncio.TimeoutError:
        window.close()
        return False, None, None, 'FCM notification sending timed out.'
    except Exception:
        window.close()
        return False, None, None, 'internal_error'


async def _execute_push_tasks_bounded_streaming(
    push_tasks: AsyncIterable[Awaitable[PushTaskResult]],
    timeout: float = 30.0,
    max_concurrency: int = _fcm_max_concurrency,
    invalid_token_handler: (
        Callable[[tuple[str, ...]], Awaitable[object]] | None
    ) = None,
) -> tuple[bool, int | None, int | None, str | None]:
    """Execute streamed push tasks with bounded concurrency.

    The async iterable may fetch Redis chunks while producing tasks, so this
    executor pulls only enough batches to fill the concurrency window.

    Args:
        push_tasks: Async iterable that yields awaitable FCM batch send tasks.
        timeout: Maximum execution time in seconds.
        max_concurrency: Maximum number of active send tasks.
        invalid_token_handler: Optional callback receiving invalid tokens.

    Returns:
        Tuple of `(ok, total_batches, successful_batches, error_message)`.
    """
    pending: set[asyncio.Future[PushTaskResult]] = set()
    max_workers = max(1, max_concurrency)

    async def run_window() -> tuple[int, int]:
        """Run the bounded task window for a streaming iterable."""
        total_batches = 0
        successful_batches = 0
        invalid_tokens: set[str] = set()
        task_iter = push_tasks.__aiter__()
        exhausted = False

        async def schedule_next() -> bool:
            """Schedule the next streamed task when capacity is available."""
            nonlocal exhausted
            if exhausted:
                return False
            try:
                awaitable = await task_iter.__anext__()
            except StopAsyncIteration:
                exhausted = True
                return False
            pending.add(asyncio.ensure_future(awaitable))
            return True

        try:
            for _ in range(max_workers):
                if not await schedule_next():
                    break

            while pending:
                done, _ = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                pending.difference_update(done)
                for task in done:
                    total_batches += 1
                    result = await task
                    successful_batches += int(_push_task_succeeded(result))
                    invalid_tokens.update(
                        _invalid_tokens_from_push_result(result),
                    )
                while len(pending) < max_workers:
                    if not await schedule_next():
                        break
        finally:
            for task in pending:
                task.cancel()
            aclose = getattr(task_iter, 'aclose', None)
            if aclose is not None:
                await aclose()

        if invalid_tokens and invalid_token_handler is not None:
            await invalid_token_handler(tuple(sorted(invalid_tokens)))

        return total_batches, successful_batches

    window = run_window()
    try:
        total, successful = await asyncio.wait_for(window, timeout=timeout)
        return True, total, successful, None
    except asyncio.TimeoutError:
        window.close()
        return False, None, None, 'FCM notification sending timed out.'
    except Exception:
        window.close()
        return False, None, None, 'internal_error'


async def _execute_push_tasks(
    push_tasks: list[Awaitable[PushTaskResult]], timeout: float = 30.0,
) -> tuple[bool, list[PushTaskResult] | None, str | None]:
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
        results = list(
            await asyncio.wait_for(
                asyncio.gather(*push_tasks, return_exceptions=False),
                timeout=timeout,
            ),
        )
        return True, results, None
    except asyncio.TimeoutError:
        # Return a generic timeout message (safe to surface to clients).
        return False, None, 'FCM notification sending timed out.'
    except Exception:
        # Do not surface internal exception details to clients.
        # Return a generic error indicator; log details at the call site.
        return False, None, 'internal_error'
