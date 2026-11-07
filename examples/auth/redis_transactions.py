from __future__ import annotations

from collections.abc import Awaitable
from collections.abc import Callable
from typing import TypeVar

from fastapi import HTTPException
from redis.asyncio import Redis
from redis.asyncio.client import Pipeline
from redis.exceptions import WatchError

T = TypeVar('T')


async def optimistic(
    redis: Redis,
    update: Callable[[Pipeline], Awaitable[T]],
) -> T:
    """Retry a watched read/modify/write without reviving deleted state."""
    for _ in range(5):
        async with redis.pipeline(transaction=True) as pipeline:
            try:
                return await update(pipeline)
            except WatchError:
                continue
    raise HTTPException(503, 'session_update_busy')
