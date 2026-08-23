from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import AsyncIterable
from collections.abc import AsyncIterator
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Coroutine
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import cast
from typing import DefaultDict
from typing import Final

import redis.asyncio as redis
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.models import Notification
from examples.local_notification_server.fcm_service import FcmSendResult
from examples.local_notification_server.fcm_service import (
    send_fcm_notification_service,
)
from examples.local_notification_server.lang_config import LANGUAGES
from examples.local_notification_server.lang_config import NotificationLanguage
from examples.local_notification_server.lang_config import Translator
from examples.local_notification_server.schemas import SiteNotifyRequest
from src.warning_types import Warnings

_token_fetch_chunk_size: Final[int] = 500
_fcm_batch_size: Final[int] = 100
_fcm_max_concurrency: Final[int] = 8
_notification_record_language: Final[NotificationLanguage] = 'zh-TW'
_notification_insert_chunk_size: Final[int] = 500
logger = logging.getLogger(__name__)
PushTaskResult = FcmSendResult


@dataclass
class PushTokenStats:
    """Token counts collected while streaming Redis recipient hashes."""

    users_with_tokens: int = 0
    token_entries: int = 0
    tokens_by_language: DefaultDict[NotificationLanguage, int] = field(
        default_factory=lambda: defaultdict(int),
    )

    def add(self, raw_maps: Iterable[Mapping[bytes, bytes]]) -> None:
        """Accumulate diagnostics from one bounded Redis result chunk."""
        for raw_map in raw_maps:
            if raw_map:
                self.users_with_tokens += 1
            for raw_language in raw_map.values():
                self.token_entries += 1
                language = cast(
                    NotificationLanguage,
                    _decode_redis_string(raw_language),
                )
                self.tokens_by_language[language] += 1


def _decode_redis_string(value: bytes) -> str:
    """Decode one Redis byte value used by the fan-out cache."""
    return value.decode()


def _decode_lang_token_map(
    raw_maps: Iterable[Mapping[bytes, bytes]],
) -> DefaultDict[NotificationLanguage, list[str]]:
    """Decode Redis hash results into a language-to-tokens map.

    Args:
        raw_maps: Byte mappings representing users' token-to-language entries.

    Returns:
        Tokens grouped by canonical BCP 47 language code.
    """
    lang_to_tokens: DefaultDict[NotificationLanguage, list[str]] = defaultdict(
        list,
    )
    for raw_map in raw_maps:
        for token_b, lang_b in raw_map.items():
            token: str = _decode_redis_string(token_b)
            lang = cast(NotificationLanguage, _decode_redis_string(lang_b))
            lang_to_tokens[lang].append(token)
    return lang_to_tokens


async def diagnose_push_preflight(
    req: SiteNotifyRequest,
    user_ids: list[int],
    rds: redis.Redis,
) -> dict[str, object]:
    """Return diagnostics for why notification recipients did not send.

    Args:
        req: Validated notification request.
        user_ids: Recipient user IDs to inspect.
        rds: Redis connection used to read token hashes.

    Returns:
        JSON-serialisable diagnostics for log and API responses.
    """
    diagnostics = await _collect_push_token_diagnostics(user_ids, rds)
    return preflight_from_token_stats(req, user_ids, diagnostics)


def preflight_from_token_stats(
    req: SiteNotifyRequest,
    user_ids: list[int],
    token_stats: PushTokenStats,
) -> dict[str, object]:
    """Build API diagnostics from already-streamed recipient token counts."""
    tokens_by_language = token_stats.tokens_by_language
    translated_languages, sendable_tokens = _sendable_push_languages(
        req.body,
        tokens_by_language,
    )

    return {
        'recipient_users': len(user_ids),
        'users_with_tokens': token_stats.users_with_tokens,
        'token_entries': token_stats.token_entries,
        'unique_tokens': token_stats.token_entries,
        'sendable_tokens': sendable_tokens,
        'tokens_by_language': dict(
            sorted(tokens_by_language.items()),
        ),
        'body_keys': list(req.body.keys()),
        'translated_languages': sorted(translated_languages),
    }


async def _collect_push_token_diagnostics(
    user_ids: list[int],
    rds: redis.Redis,
) -> PushTokenStats:
    """Collect recipient-token counts in Redis-sized chunks.

    Args:
        user_ids: Recipient identifiers whose cached tokens are inspected.
        rds: Redis connection used to read token-to-language hashes.

    Returns:
        Aggregate counts and per-language token totals.
    """
    token_stats = PushTokenStats()
    for start in range(0, len(user_ids), _token_fetch_chunk_size):
        pipe = rds.pipeline()
        for user_id in user_ids[start: start + _token_fetch_chunk_size]:
            pipe.hgetall(f"fcm_tokens:{user_id}")
        results: list[dict[bytes, bytes]] = await pipe.execute()
        token_stats.add(results)
    return token_stats


def _sendable_push_languages(
    body: Warnings,
    tokens_by_language: Mapping[NotificationLanguage, int],
) -> tuple[list[NotificationLanguage], int]:
    """Return languages with complete translations and their token count.

    Args:
        body: Validated warning payload to translate.
        tokens_by_language: Token count for every recipient language.

    Returns:
        Languages that can render the payload and their total token count.
    """
    translated_languages: list[NotificationLanguage] = []
    sendable_tokens = 0
    for language, token_count in tokens_by_language.items():
        Translator.translate_from_dict(body, language)
        translated_languages.append(language)
        sendable_tokens += token_count
    return translated_languages, sendable_tokens


async def create_notification_records_for_users(
    req: SiteNotifyRequest,
    user_ids: list[int],
    db: AsyncSession,
) -> int:
    """Persist one notification-centre record for each recipient user.

    Args:
        req: Validated site-notification request.
        user_ids: Potential recipient user identifiers.
        db: Database session used to create notification records.

    Returns:
        Number of recipient records persisted.
    """
    record_body = f"{req.site} - {req.stream_name}\n" + '\n'.join(
        Translator.translate_from_dict(
            req.body,
            _notification_record_language,
        ),
    )
    for start in range(0, len(user_ids), _notification_insert_chunk_size):
        values = [
            {
                'user_id': user_id,
                'type': req.notification_type,
                'title': req.title,
                'body': record_body,
                'deep_link': req.deep_link,
                'metadata_json': req.metadata,
            }
            for user_id in user_ids[
                start: start + _notification_insert_chunk_size
            ]
        ]
        await db.execute(insert(Notification), values)
    await db.commit()
    return len(user_ids)


def build_push_task(
    req: SiteNotifyRequest,
    lang: NotificationLanguage,
    tokens: list[str],
    *,
    send_notification: Callable[..., Awaitable[PushTaskResult]] | None = None,
) -> Awaitable[PushTaskResult]:
    """Build one FCM batch task for one canonical language.

    Args:
        req: Validated notification request.
        lang: Canonical language code for the target tokens.
        tokens: Device tokens in this batch.

    Returns:
        Awaitable send task.
    """
    title = LANGUAGES[lang]['warning_notification']
    translated_lines = Translator.translate_from_dict(req.body, lang)
    data = {
        'navigate': 'violation_list_page',
        'violation_id': (
            str(req.violation_id) if req.violation_id is not None else ''
        ),
        'deep_link': req.deep_link,
        'type': req.notification_type,
    }

    body: str = f"{req.site} - {req.stream_name}\n" + '\n'.join(
        translated_lines,
    )

    logger.info(
        'FCM notification batch prepared lang=%s tokens=%d body_lines=%d '
        'data_keys=%s',
        lang,
        len(tokens),
        len(translated_lines),
        sorted(data),
    )

    sender = send_notification or send_fcm_notification_service
    return sender(
        device_tokens=tokens,
        title=title,
        body=body,
        image_path=req.image_path,
        data=data,
    )


async def iter_push_tasks_streaming(
    req: SiteNotifyRequest,
    user_ids: list[int],
    rds: redis.Redis,
    *,
    token_fetch_chunk_size: int | None = None,
    fcm_batch_size: int | None = None,
    token_stats: PushTokenStats | None = None,
    build_push_task_fn: (
        Callable[
            [SiteNotifyRequest, NotificationLanguage, list[str]],
            Awaitable[PushTaskResult],
        ]
        | None
    ) = None,
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
    chunk_size = token_fetch_chunk_size or _token_fetch_chunk_size
    batch_size = fcm_batch_size or _fcm_batch_size
    task_builder = build_push_task_fn or build_push_task
    pending_batches: DefaultDict[NotificationLanguage, list[str]] = (
        defaultdict(list)
    )

    for start in range(0, len(user_ids), chunk_size):
        # Read bounded Redis chunks before forming language-specific FCM
        # batches.
        pipe = rds.pipeline()
        for user_id in user_ids[start: start + chunk_size]:
            pipe.hgetall(f"fcm_tokens:{user_id}")

        redis_results: list[dict[bytes, bytes]] = await pipe.execute()
        if token_stats is not None:
            token_stats.add(redis_results)
        chunk_tokens = _decode_lang_token_map(redis_results)

        for lang, tokens in chunk_tokens.items():
            batch = pending_batches[lang]
            for token in tokens:
                batch.append(token)
                if len(batch) >= batch_size:
                    task = task_builder(req, lang, list(batch))
                    batch.clear()
                    yield task

    for lang, tokens in pending_batches.items():
        if not tokens:
            continue
        yield task_builder(req, lang, list(tokens))


async def execute_push_tasks_bounded_streaming(
    push_tasks: AsyncIterable[Awaitable[PushTaskResult]],
    invalid_token_handler: Callable[[tuple[str, ...]], Awaitable[object]],
    timeout: float = 30.0,
    max_concurrency: int = _fcm_max_concurrency,
) -> tuple[bool, int | None, int | None, str | None]:
    """Execute streamed push tasks with bounded concurrency.

    The async iterable may fetch Redis chunks while producing tasks, so this
    executor pulls only enough batches to fill the concurrency window.

    Args:
        push_tasks: Async iterable that yields awaitable FCM batch send tasks.
        timeout: Maximum execution time in seconds.
        max_concurrency: Maximum number of active send tasks.
        invalid_token_handler: Callback receiving invalid tokens.

    Returns:
        Tuple of `(ok, total_batches, successful_batches, error_message)`.
    """
    window = _run_streaming_push_task_window(
        push_tasks,
        max_concurrency,
        invalid_token_handler,
    )
    return await _complete_push_task_window(window, timeout)


async def _fill_pending_streaming_push_tasks(
    task_iter: AsyncIterator[Awaitable[PushTaskResult]],
    pending: set[asyncio.Future[PushTaskResult]],
    max_workers: int,
) -> None:
    """Fill a streaming FCM task window until capacity or iterator exhaustion.

    Args:
        task_iter: Async iterator yielding unscheduled FCM batch tasks.
        pending: Mutable set of currently scheduled tasks.
        max_workers: Maximum number of concurrently scheduled tasks.
    """
    while len(pending) < max_workers:
        try:
            awaitable = await task_iter.__anext__()
        except StopAsyncIteration:
            return
        pending.add(asyncio.ensure_future(awaitable))


async def _collect_completed_push_tasks(
    pending: set[asyncio.Future[PushTaskResult]],
) -> tuple[int, int, set[str]]:
    """Await completed tasks and aggregate their FCM result details.

    Args:
        pending: Currently scheduled FCM batch tasks.

    Returns:
        Completed count, successful count, and invalid tokens from the window.
    """
    done, _ = await asyncio.wait(
        pending,
        return_when=asyncio.FIRST_COMPLETED,
    )
    pending.difference_update(done)
    successful_batches = 0
    invalid_tokens: set[str] = set()
    for task in done:
        result = await task
        successful_batches += int(
            result.success_count > 0 and result.failure_count == 0,
        )
        invalid_tokens.update(result.invalid_tokens)
    return len(done), successful_batches, invalid_tokens


async def _run_streaming_push_task_window(
    push_tasks: AsyncIterable[Awaitable[PushTaskResult]],
    max_workers: int,
    invalid_token_handler: Callable[[tuple[str, ...]], Awaitable[object]],
) -> tuple[int, int]:
    """Run lazily streamed push tasks with bounded concurrency.

    Args:
        push_tasks: Async iterable yielding FCM batch tasks on demand.
        max_workers: Maximum number of concurrently scheduled tasks.
        invalid_token_handler: Callback for invalid FCM tokens.

    Returns:
        Total and successful FCM batch counts.
    """
    pending: set[asyncio.Future[PushTaskResult]] = set()
    total_batches = 0
    successful_batches = 0
    invalid_tokens: set[str] = set()
    task_iter = push_tasks.__aiter__()
    try:
        await _fill_pending_streaming_push_tasks(
            task_iter,
            pending,
            max_workers,
        )
        while pending:
            total, successful, invalid = await _collect_completed_push_tasks(
                pending,
            )
            total_batches += total
            successful_batches += successful
            invalid_tokens.update(invalid)
            await _fill_pending_streaming_push_tasks(
                task_iter,
                pending,
                max_workers,
            )
    finally:
        # Cancel scheduled work and release the stream's Redis resources.
        for task in pending:
            task.cancel()
    if invalid_tokens:
        await invalid_token_handler(tuple(sorted(invalid_tokens)))
    return total_batches, successful_batches


async def _complete_push_task_window(
    window: Coroutine[Any, Any, tuple[int, int]],
    timeout: float,
) -> tuple[bool, int | None, int | None, str | None]:
    """Apply the public timeout and error contract to one task window.

    Args:
        window: FCM task-window coroutine aggregation operation.
        timeout: Maximum time allowed for all work in the window.

    Returns:
        Completion flag, total batches, successful batches, and error code.
    """
    try:
        total, successful = await asyncio.wait_for(window, timeout=timeout)
        return True, total, successful, None
    except asyncio.TimeoutError:
        window.close()
        return False, None, None, 'FCM notification sending timed out.'
    except Exception:
        window.close()
        return False, None, None, 'internal_error'
