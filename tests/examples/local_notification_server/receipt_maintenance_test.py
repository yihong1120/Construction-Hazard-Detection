from __future__ import annotations

import asyncio
from datetime import datetime
from datetime import timedelta
from datetime import timezone

from sqlalchemy import func
from sqlalchemy import select

from examples.auth.models import NotificationDeliveryReceipt as Receipt
from examples.auth.models import Tenant
from examples.auth.models import User
from examples.local_notification_server.receipt_maintenance import (
    ReceiptMaintenance,
)
from examples.local_notification_server.receipt_maintenance import (
    ReceiptMaintenanceSettings,
)
from tests.database_fixtures_test import OrmDatabase


class ReceiptMaintenanceTest(OrmDatabase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        async with self.sessions() as db:
            tenant = Tenant(name='cleanup')
            user = User(username='cleanup', role='user', tenant=tenant)
            db.add(user)
            await db.flush()
            now = datetime.now(timezone.utc)
            db.add_all(
                [
                    Receipt(
                        event_digest=str(index),
                        user_id=user.id,
                        token_hash='token',
                        registration_at=now,
                        delivered_at=now,
                        expires_at=now
                        + timedelta(days=1 if index == 7 else -1),
                    )
                    for index in range(8)
                ],
            )
            await db.commit()

    async def count(self):
        async with self.sessions() as db:
            return await db.scalar(select(func.count()).select_from(Receipt))

    async def test_each_sweep_has_a_bounded_batch_budget(self):
        worker = ReceiptMaintenance(
            self.sessions,
            ReceiptMaintenanceSettings(batch_size=2, max_batches=2),
        )
        self.assertEqual(await worker.sweep(), 4)
        self.assertEqual(await self.count(), 4)
        self.assertEqual(await worker.sweep(), 3)
        self.assertEqual(await self.count(), 1)

    async def test_concurrent_workers_skip_locked_receipts(self):
        worker = ReceiptMaintenance(
            self.sessions,
            ReceiptMaintenanceSettings(batch_size=2, max_batches=1),
        )
        async with self.sessions() as db:
            await db.execute(
                select(Receipt)
                .where(Receipt.event_digest == '0')
                .with_for_update(),
            )
            self.assertEqual(await asyncio.wait_for(worker.sweep(), 1), 2)
            await db.rollback()
        self.assertEqual(await worker.sweep(), 2)

    async def test_lifecycle_stop_does_not_wait_for_the_interval(self):
        worker = ReceiptMaintenance(self.sessions)
        async with worker.lifespan():
            pass
        self.assertTrue(worker.stop.is_set())
