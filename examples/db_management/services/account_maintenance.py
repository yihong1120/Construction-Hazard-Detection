from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from datetime import timezone
from typing import TYPE_CHECKING

from sqlalchemy import select

from examples.auth.models import AccountOperation
from src.async_tasks import finish_on_cancel

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from examples.db_management.services.account_lifecycle import (
        AccountLifecycle,
    )


class AccountMaintenance:
    """Give synchronization and retention independent failure boundaries."""

    def __init__(self, lifecycle: AccountLifecycle):
        self.lifecycle = lifecycle
        self.settings = lifecycle.settings

    async def sweep(self):
        await _AccountSweep(self.lifecycle).run()

    async def cleanup(self):
        async with asyncio.timeout(self.settings.cleanup_budget_seconds):
            await self.lifecycle.purge_completed()

    async def _run(self, operation, interval):
        while not self.lifecycle.stop.is_set():
            try:
                await asyncio.wait_for(self.lifecycle.stop.wait(), interval)
            except TimeoutError:
                try:
                    await operation()
                except Exception as error:
                    logger.warning(
                        'Account maintenance deferred task=%s error=%s',
                        operation.__name__,
                        type(error).__name__,
                    )

    @asynccontextmanager
    async def lifespan(self):
        self.lifecycle.stop.clear()
        tasks = [
            asyncio.create_task(
                self._run(
                    self.sweep,
                    self.settings.sync_interval_seconds,
                ),
            ),
            asyncio.create_task(
                self._run(
                    self.cleanup,
                    self.settings.cleanup_interval_seconds,
                ),
            ),
        ]
        try:
            yield self.lifecycle
        finally:
            self.lifecycle.stop.set()
            for task in tasks:
                task.cancel()
            await finish_on_cancel(
                asyncio.gather(*tasks, return_exceptions=True),
            )


class _AccountSweep:
    def __init__(self, lifecycle: AccountLifecycle):
        self.lifecycle = lifecycle
        self.settings = lifecycle.settings

    async def run(self):
        self.deadline = time.monotonic() + self.settings.sync_budget_seconds
        async with asyncio.timeout_at(
            self.deadline,
        ), self.lifecycle.sessions() as db:
            ids = list(
                await db.scalars(
                    select(AccountOperation.id)
                    .where(
                        AccountOperation.phase.not_in(
                            ('completed', 'uncertain'),
                        ),
                        AccountOperation.next_attempt
                        <= datetime.now(timezone.utc),
                    )
                    .order_by(
                        AccountOperation.next_attempt,
                        AccountOperation.created_at,
                    )
                    .limit(self.settings.sync_batch_size),
                ),
            )
        self.capacity = asyncio.Semaphore(self.settings.sync_concurrency)

        async with asyncio.TaskGroup() as tasks:
            for operation_id in ids:
                tasks.create_task(self.process(operation_id))

    async def process(self, operation_id):
        async with self.capacity:
            if (
                self.lifecycle.stop.is_set()
                or time.monotonic() >= self.deadline
            ):
                return
            try:
                async with asyncio.timeout_at(self.deadline):
                    await self.lifecycle.process(operation_id)
            except TimeoutError:
                await self.lifecycle.defer(operation_id, 'TimeoutError')
            except Exception as error:
                await self.lifecycle.defer(operation_id, type(error).__name__)
