from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from datetime import timezone

from pydantic import Field
from sqlalchemy import delete
from sqlalchemy import select
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.models import AccountDeletionReceipt
from examples.auth.models import AccountOperation
from src.settings.base import EnvironmentSettings


def now():
    return datetime.now(timezone.utc)


class AccountMaintenanceSettings(EnvironmentSettings):
    model_config = {'env_prefix': 'ACCOUNT_OPERATIONS_'}
    completed_days: int = Field(30, ge=1)
    deletion_receipt_days: int = Field(365, ge=1)
    cleanup_batch_size: int = Field(128, ge=1, le=4096)
    sync_interval_seconds: float = Field(5, gt=0, allow_inf_nan=False)
    cleanup_interval_seconds: float = Field(60, gt=0, allow_inf_nan=False)
    sync_budget_seconds: float = Field(30, gt=0, allow_inf_nan=False)
    cleanup_budget_seconds: float = Field(2, gt=0, allow_inf_nan=False)
    sync_batch_size: int = Field(16, ge=1, le=256)
    sync_concurrency: int = Field(2, ge=1, le=8)


class AccountRetention:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        settings: AccountMaintenanceSettings,
    ):
        self.sessions = sessions
        self.settings = settings

    async def complete(self, db, record):
        record.phase = 'completed'
        record.last_error = None
        record.completed_at = now()
        if record.kind == 'delete':
            receipt = await db.get(AccountDeletionReceipt, record.user_id)
            if receipt is None:
                receipt = AccountDeletionReceipt(user_id=record.user_id)
                db.add(receipt)
            for name in ('tenant_id', 'group_id', 'username', 'role'):
                setattr(receipt, name, getattr(record, name))
            receipt.completed_at = record.completed_at
            receipt.expires_at = now() + timedelta(
                days=self.settings.deletion_receipt_days,
            )

    async def purge(self):
        async with self.sessions() as db, db.begin():
            await db.execute(text("SET LOCAL statement_timeout='1000ms'"))
            expired = (
                select(AccountOperation.id)
                .where(
                    AccountOperation.phase == 'completed',
                    AccountOperation.completed_at
                    < now() - timedelta(days=self.settings.completed_days),
                )
                .order_by(AccountOperation.completed_at)
                .limit(self.settings.cleanup_batch_size)
                .with_for_update(skip_locked=True)
            )
            await db.execute(
                delete(AccountOperation).where(
                    AccountOperation.id.in_(expired),
                ),
            )
            receipts = (
                select(AccountDeletionReceipt.user_id)
                .where(AccountDeletionReceipt.expires_at <= now())
                .limit(self.settings.cleanup_batch_size)
                .with_for_update(skip_locked=True)
            )
            await db.execute(
                delete(AccountDeletionReceipt).where(
                    AccountDeletionReceipt.user_id.in_(receipts),
                ),
            )
