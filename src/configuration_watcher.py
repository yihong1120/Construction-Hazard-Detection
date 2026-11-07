from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from asyncpg.pool import Pool


class ConfigurationWatcher:
    def __init__(
        self,
        get_pool: Callable[[], Awaitable[Pool | None]],
        reload: Callable[[], Awaitable[None]],
        logger: logging.Logger,
    ) -> None:
        self.get_pool = get_pool
        self.reload_callback = reload
        self.logger = logger
        self.pool: Pool | None = None
        self.connection: Any | None = None
        self.task: asyncio.Task[None] | None = None
        self.reload_requested = False

    async def start(self) -> None:
        """Listen for committed stream-config changes between poll cycles."""
        if self.connection is not None:
            return

        self.pool = await self.get_pool()
        if self.pool is None:
            return
        connection = await self.pool.acquire()
        try:
            await connection.add_listener(
                'stream_config_changed',
                self.notify,
            )
        except Exception:
            await self.pool.release(connection)
            raise
        self.connection = connection
        self.logger.info(
            '[database] Listening for stream configuration changes',
        )

    def notify(
        self,
        _connection: object,
        _pid: int,
        _channel: str,
        _payload: str,
    ) -> None:
        """Schedule a reload when PostgreSQL signals a config change."""
        self.reload_requested = True
        if self.task is None or self.task.done():
            reload_task = asyncio.create_task(
                self.reload(),
            )
            reload_task.add_done_callback(
                self.log_failure,
            )
            self.task = reload_task

    async def reload(self) -> None:
        """Coalesce notifications that arrive during one configuration read."""
        while self.reload_requested:
            self.reload_requested = False
            await self.reload_callback()

    def log_failure(self, task: asyncio.Task[None]) -> None:
        """Log callback reload failures instead of losing the task error."""
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            self.logger.error(f"[config] Immediate reload failed: {error}")

    async def cancel_reload(self) -> None:
        """Cancel an in-flight callback reload before process shutdown."""
        task = self.task
        self.task = None
        if task is None or task.done() or task is asyncio.current_task():
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def close(self) -> None:
        """Release the reserved LISTEN connection before closing its pool."""
        connection = self.connection
        self.connection = None
        if connection is None:
            return

        target_pool = self.pool
        if target_pool is None:
            return
        with suppress(Exception):
            await connection.remove_listener(
                'stream_config_changed',
                self.notify,
            )
        with suppress(Exception):
            await target_pool.release(connection)
