from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import patch

from sqlalchemy import select

from examples.auth.models import EvidenceWriteOperation
from examples.violation_records import evidence_storage as module
from tests.database_fixtures_test import OrmDatabase


async def _round_budget_stops_scheduling_and_recover(*args):
    await asyncio.sleep(0.01)
    return True


class EvidenceRecoveryTest(OrmDatabase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.recovery = module.EvidenceStorageMaintenance(
            self.engine,
            self.root,
            module.EvidenceRecoverySettings(batch_size=2, min_age_seconds=0),
        )

    async def insert(self, paths):
        stamp = datetime.now(timezone.utc) - timedelta(minutes=2)
        async with self.sessions() as db:
            for index, path in enumerate(paths):
                db.add(
                    EvidenceWriteOperation(
                        id=str(index),
                        path=path,
                        created_at=stamp,
                        next_attempt=stamp,
                    ),
                )
            await db.commit()

    async def test_poisoned_old_jobs_back_off_and_later_jobs_progress(self):
        await self.insert(['../outside-one', '../outside-two', 'valid.png'])
        orphan = self.root / 'valid.png'
        orphan.write_bytes(b'orphan')
        await self.recovery.sweep()
        async with self.sessions() as db:
            records = list(
                await db.scalars(
                    select(EvidenceWriteOperation).order_by(
                        EvidenceWriteOperation.id,
                    ),
                ),
            )
            self.assertEqual([r.attempts for r in records], [1, 1, 0])
            self.assertEqual(records[0].last_error, 'ValueError')
        await self.recovery.sweep()
        self.assertFalse(orphan.exists())
        async with self.sessions() as db:
            self.assertIsNone(await db.get(EvidenceWriteOperation, '2'))

    async def test_locked_writer_is_rescheduled_without_counting_as_failure(
        self,
    ):
        await self.insert(['owned.png'])
        async with self.engine.begin() as connection:
            from sqlalchemy import text

            await connection.execute(
                text(
                    "SELECT pg_advisory_xact_lock(hashtextextended('0',"
                    '4103))',
                ),
            )
            await self.recovery.sweep()
        async with self.sessions() as db:
            row = await db.get(EvidenceWriteOperation, '0')
            self.assertEqual(row.attempts, 0)
            self.assertEqual(row.last_error, 'writer_active')
            self.assertGreater(row.next_attempt, datetime.now(timezone.utc))

    async def test_cleanup_failure_retains_intent_until_a_successful_retry(
        self,
    ):
        await self.insert(['orphan.png'])
        orphan = self.root / 'orphan.png'
        orphan.write_bytes(b'orphan')
        with patch.object(
            module,
            'sync_directory',
            side_effect=OSError('filesystem unavailable'),
        ):
            await self.recovery.sweep()
        async with self.sessions() as db:
            row = await db.get(EvidenceWriteOperation, '0')
            self.assertEqual(row.attempts, 1)
            row.next_attempt = datetime.now(timezone.utc) - timedelta(
                seconds=1,
            )
            await db.commit()
        await self.recovery.sweep()
        async with self.sessions() as db:
            self.assertIsNone(await db.get(EvidenceWriteOperation, '0'))

    async def test_round_budget_stops_scheduling_and_lifespan_drains(self):
        await self.insert(['one.png', 'two.png'])
        self.recovery.settings.budget_seconds = 0.001

        recover = _round_budget_stops_scheduling_and_recover

        with patch.object(
            self.recovery,
            '_recover',
            new=AsyncMock(side_effect=recover),
        ) as recover_job:
            with self.assertRaises(TimeoutError):
                await self.recovery.sweep()
        self.assertLessEqual(recover_job.await_count, 1)
        async with self.recovery.lifespan():
            pass
        self.assertTrue(self.recovery.stop.is_set())
