from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
import time
from types import TracebackType
from typing import Literal

from redis.asyncio import Redis
from redis.exceptions import RedisError

from examples.local_notification_server.schemas import SiteNotifyRequest
from src.async_tasks import finish_on_cancel

logger = logging.getLogger(__name__)
_FINISH = '''
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
if ARGV[2] == 'done' then
    return redis.call('SET', KEYS[1], 'done', 'EX', ARGV[3])
end
return redis.call('DEL', KEYS[1])
'''
_RENEW = '''
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
return redis.call('EXPIRE', KEYS[1], ARGV[2])
'''


class NotificationLeaseLost(RuntimeError):
    pass


class NotificationClaim:
    def __init__(self, redis: Redis, key: str, completed_ttl: int) -> None:
        self.redis = redis
        self.key = key
        self.completed_ttl = completed_ttl
        self.owner = 'processing:' + secrets.token_urlsafe(24)
        self.acquired = False
        self.completed = False
        self.lease_seconds = 120
        self._deadline = 0.0
        self._heartbeat: asyncio.Task | None = None

    io_timeout_seconds = 5.0

    async def _io(self, operation, *, owned=False):
        limit = time.monotonic() + self.io_timeout_seconds
        if owned:
            limit = min(limit, self._deadline)
        if limit <= time.monotonic():
            operation.close()
            raise NotificationLeaseLost('Notification delivery lease expired')
        try:
            async with asyncio.timeout_at(limit):
                return await operation
        except (TimeoutError, RedisError) as error:
            raise NotificationLeaseLost(
                'Notification delivery lease unconfirmed',
            ) from error

    async def acquire(self) -> bool:
        started = time.monotonic()
        self.acquired = bool(
            await self._io(
                self.redis.set(
                    self.key,
                    self.owner,
                    ex=self.lease_seconds,
                    nx=True,
                ),
            ),
        )
        self._deadline = started + self.lease_seconds
        if self.acquired and self._deadline <= time.monotonic():
            raise NotificationLeaseLost('Notification delivery lease expired')
        return self.acquired

    async def is_completed(self) -> bool:
        return await self._io(self.redis.get(self.key)) in {b'done', 'done'}

    async def finish(self) -> None:
        if not await self._io(
            self.redis.eval(
                _FINISH,
                1,
                self.key,
                self.owner,
                'done',
                self.completed_ttl,
            ),
            owned=True,
        ):
            raise NotificationLeaseLost('Notification delivery lease expired')
        self.completed = True

    async def guard(self) -> None:
        if await self._io(self.redis.get(self.key), owned=True) not in (
            self.owner,
            self.owner.encode(),
        ):
            raise NotificationLeaseLost('Notification delivery lease expired')

    async def run(self, operation):
        if self._heartbeat is None:
            operation.close()
            raise RuntimeError('Notification claim is not running')
        task = asyncio.create_task(operation)
        try:
            done, _ = await asyncio.wait(
                (task, self._heartbeat),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if self._heartbeat in done:
                raise NotificationLeaseLost(
                    'Notification delivery lease expired',
                )
            return await task
        finally:
            if not task.done():
                task.cancel()
            await finish_on_cancel(
                asyncio.gather(task, return_exceptions=True),
            )

    async def _renew(self) -> None:
        while True:
            await asyncio.sleep(self.lease_seconds / 3)
            started = time.monotonic()
            if not await self._io(
                self.redis.eval(
                    _RENEW,
                    1,
                    self.key,
                    self.owner,
                    self.lease_seconds,
                ),
                owned=True,
            ):
                raise NotificationLeaseLost(
                    'Notification delivery lease expired',
                )
            self._deadline = started + self.lease_seconds

    async def __aenter__(self) -> NotificationClaim:
        if not self.acquired:
            raise RuntimeError('Notification claim was not acquired')
        self._heartbeat = asyncio.create_task(self._renew())
        return self

    async def _release(self):
        try:
            await self._io(
                self.redis.eval(
                    _FINISH,
                    1,
                    self.key,
                    self.owner,
                    'release',
                    self.completed_ttl,
                ),
            )
        except Exception:
            logger.warning(
                'Notification lease release deferred key=%s',
                self.key,
            )

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:

        await finish_on_cancel(self._cleanup())
        return False

    async def _cleanup(self):
        if self._heartbeat is not None:
            self._heartbeat.cancel()
            await asyncio.gather(self._heartbeat, return_exceptions=True)
        if self.acquired and not self.completed:
            await self._release()


def notification_event_key(req: SiteNotifyRequest) -> str:
    """Build the Redis key used to suppress a duplicate send.

    Args:
        req: Validated site notification request.

    Returns:
        Stable Redis deduplication key.
    """
    if req.violation_id is not None:
        if req.site_id is not None:
            return f'fcm_notification_dedupe:{req.site_id}:{req.violation_id}'
        return (
            'fcm_notification_dedupe:'
            f"{req.site}:{req.stream_name}:{req.violation_id}"
        )
    # Events without a durable violation ID deduplicate on their semantic
    # payload.
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
    return f"fcm_notification_dedupe:{payload_hash}"
