from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from collections.abc import Callable
from dataclasses import dataclass

import redis.asyncio as redis

from examples.streaming_web.redis_service import fetch_latest_metadata_for_key
from examples.streaming_web.schemas import FrameOutData


_subscriber_queue_size = 1
_retry_delay_seconds = 1.0
_idle_retry_delay_seconds = 0.05
_sleep = asyncio.sleep


@dataclass
class _StreamSubscribers:
    """State shared by all local subscribers to one Redis Stream key."""

    rds: redis.Redis
    queues: set[asyncio.Queue[FrameOutData | Exception]]
    fetcher: Callable[
        [redis.Redis, str, str], Awaitable[FrameOutData | None],
    ]
    task: asyncio.Task[None] | None = None


class MetadataSubscription:
    """A latest-only queue subscription that can be closed idempotently."""

    def __init__(
        self,
        fanout: MetadataFanout,
        redis_key: str,
        queue: asyncio.Queue[FrameOutData | Exception],
    ) -> None:
        """Perform init.

        Args:
            fanout: Value used by this callable.
            redis_key: Value used by this callable.
            queue: Value used by this callable.
        """
        self._fanout = fanout
        self._redis_key = redis_key
        self.queue = queue
        self._closed = False

    async def get(self) -> FrameOutData | Exception:
        """Wait for the newest metadata event or reader error."""
        return await self.queue.get()

    async def close(self) -> None:
        """Remove this subscriber without affecting other browser clients."""
        if self._closed:
            return
        self._closed = True
        await self._fanout.unsubscribe(self._redis_key, self.queue)


class MetadataFanout:
    """Share one blocking Redis read per metadata key and process."""

    def __init__(self) -> None:
        """Perform init.
        """
        self._streams: dict[str, _StreamSubscribers] = {}
        self._lock = asyncio.Lock()

    async def subscribe(
        self,
        rds: redis.Redis,
        redis_key: str,
        *,
        fetcher: Callable[
            [redis.Redis, str, str], Awaitable[FrameOutData | None],
        ] = fetch_latest_metadata_for_key,
    ) -> MetadataSubscription:
        """Subscribe a browser connection using a bounded latest-only queue."""
        queue: asyncio.Queue[FrameOutData | Exception] = asyncio.Queue(
            maxsize=_subscriber_queue_size,
        )
        async with self._lock:
            state = self._streams.get(redis_key)
            if state is None:
                state = _StreamSubscribers(
                    rds=rds,
                    queues=set(),
                    fetcher=fetcher,
                )
                self._streams[redis_key] = state
            state.queues.add(queue)
            if state.task is None or state.task.done():
                state.task = asyncio.create_task(self._run(redis_key, state))
        return MetadataSubscription(self, redis_key, queue)

    async def unsubscribe(
        self,
        redis_key: str,
        queue: asyncio.Queue[FrameOutData | Exception],
    ) -> None:
        """Drop a subscriber and let an idle reader exit after its XREAD."""
        async with self._lock:
            state = self._streams.get(redis_key)
            if state is not None:
                state.queues.discard(queue)

    async def _run(self, redis_key: str, state: _StreamSubscribers) -> None:
        """Read one Redis Stream and broadcast only the newest value."""
        last_id = '$'
        try:
            while True:
                async with self._lock:
                    if not state.queues:
                        self._streams.pop(redis_key, None)
                        return
                try:
                    frame_data = await state.fetcher(
                        state.rds,
                        redis_key,
                        last_id,
                    )
                except Exception as exc:
                    await self._publish(state, exc)
                    await _sleep(_retry_delay_seconds)
                    continue
                if frame_data is None:
                    # XREAD normally blocks.  Yield defensively when a Redis
                    # client returns immediately so an idle stream cannot spin
                    # a CPU core or starve subscriber cancellation.
                    await _sleep(_idle_retry_delay_seconds)
                    continue
                last_id = str(frame_data['id'])
                await self._publish(state, frame_data)
                # A well-behaved Redis XREAD blocks on the next pass.  Yield
                # once regardless so a misconfigured client cannot monopolise
                # the event loop by returning frames immediately.
                await _sleep(0)
        finally:
            async with self._lock:
                current = self._streams.get(redis_key)
                if current is state and not state.queues:
                    self._streams.pop(redis_key, None)

    @staticmethod
    async def _publish(
        state: _StreamSubscribers,
        item: FrameOutData | Exception,
    ) -> None:
        """Offer one item to every client without a slow client growing RAM."""
        for queue in tuple(state.queues):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(item)


metadata_fanout = MetadataFanout()
