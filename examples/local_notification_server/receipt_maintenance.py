from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from pydantic import Field
from sqlalchemy import delete
from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy import text
from sqlalchemy import tuple_

from examples.auth.database import AsyncSessionLocal
from examples.auth.models import NotificationDeliveryReceipt as Receipt
from src.settings.base import EnvironmentSettings

logger = logging.getLogger(__name__)


class ReceiptMaintenanceSettings(EnvironmentSettings):
    model_config = {'env_prefix': 'NOTIFICATION_RECEIPTS_'}
    interval_seconds: float = Field(60, gt=0, allow_inf_nan=False)
    batch_size: int = Field(500, ge=1, le=5000)
    max_batches: int = Field(4, ge=1, le=32)
    budget_seconds: float = Field(2, gt=0, le=30, allow_inf_nan=False)


class ReceiptMaintenance:
    def __init__(self, sessions=AsyncSessionLocal, settings=None):
        self.sessions = sessions
        self.settings = settings or ReceiptMaintenanceSettings()
        self.stop = asyncio.Event()

    async def sweep(self) -> int:
        removed = 0
        async with asyncio.timeout(self.settings.budget_seconds):
            for _ in range(self.settings.max_batches):
                async with self.sessions() as db, db.begin():
                    await db.execute(
                        text("SET LOCAL statement_timeout='1000ms'"),
                    )
                    keys = (
                        Receipt.event_digest,
                        Receipt.user_id,
                        Receipt.token_hash,
                        Receipt.registration_at,
                    )
                    expired = (
                        select(*keys)
                        .where(Receipt.expires_at <= func.now())
                        .order_by(Receipt.expires_at)
                        .limit(self.settings.batch_size)
                        .with_for_update(skip_locked=True)
                        .cte('expired_receipts')
                    )
                    rows = await db.execute(
                        delete(Receipt)
                        .where(tuple_(*keys).in_(select(expired)))
                        .returning(Receipt.event_digest),
                    )
                    count = len(rows.all())
                    removed += count
                if count < self.settings.batch_size or self.stop.is_set():
                    break
        return removed

    async def run(self) -> None:
        while not self.stop.is_set():
            try:
                await asyncio.wait_for(
                    self.stop.wait(),
                    self.settings.interval_seconds,
                )
            except TimeoutError:
                try:
                    await self.sweep()
                except Exception:
                    logger.warning(
                        'Receipt maintenance deferred',
                        exc_info=True,
                    )

    @asynccontextmanager
    async def lifespan(self):
        task = asyncio.create_task(self.run())
        try:
            yield self
        finally:
            self.stop.set()
            await task
