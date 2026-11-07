from __future__ import annotations

import asyncio
import tempfile
from dataclasses import replace
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import patch
from uuid import uuid4

from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from starlette.datastructures import UploadFile

from examples.auth.models import EvidenceWriteOperation
from examples.auth.models import Site
from examples.auth.models import Violation
from examples.violation_records.evidence import EvidenceEventConflictError
from examples.violation_records.evidence import ViolationEvidence
from examples.violation_records.evidence_storage import (
    EvidenceStorageMaintenance,
)
from examples.violation_records.evidence_storage import EvidenceWrite
from examples.violation_records.evidence_storage import ViolationImageReadError
from examples.violation_records.violation_manager import ViolationManager
from tests.database_fixtures_test import OrmDatabase
from tests.image_fixtures_test import image_bytes


async def _unknown_commit_reply_keeps_the_lost_reply(commit, /):
    await commit()
    raise SQLAlchemyError('commit response lost')


class _DirectorySyncFailurePreventsCommittingEvidenceCallbacks:
    calls: Any
    original: Any
    testcase: TestViolationManager

    def _call_sync(self, path):
        self.calls.append(path)
        # Fail the storage-root sync after the final file was promoted.
        if path == self.testcase.root.resolve() and list(
            self.testcase.root.rglob('*.png'),
        ):
            raise OSError('directory sync failed')
        self.original(path)


class _CancelledReadCleansPartialUploadUpload:
    def __init__(self, *, _test_entered):
        self._test_entered = _test_entered

    reads = 0

    async def read(self, size):
        self.reads += 1
        if self.reads == 1:
            return b'partial'
        self._test_entered.set()
        await asyncio.Event().wait()


class TestViolationManager(OrmDatabase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manager = ViolationManager(self.root)
        self.recovery = EvidenceStorageMaintenance(self.engine, self.root)
        async with self.sessions() as db:
            db.add(Site(id=1, name='Site'))
            await db.commit()

    def evidence(self, **changes):
        values = dict(
            site='Site',
            site_id=1,
            stream_name='Camera',
            event_id=str(uuid4()),
            detection_time=datetime.now(timezone.utc),
            image_file=UploadFile(
                BytesIO(image_bytes()),
                filename='image.jpg',
            ),
        )
        return ViolationEvidence(**(values | changes))

    async def age_intents(self):
        async with self.sessions() as db:
            for record in (
                await db.scalars(select(EvidenceWriteOperation))
            ).all():
                record.next_attempt = datetime.now(timezone.utc) - timedelta(
                    seconds=1,
                )
                record.created_at = datetime.now(timezone.utc) - timedelta(
                    minutes=2,
                )
            await db.commit()

    async def test_upload_and_retry_keep_one_durable_image_and_fingerprint(
        self,
    ):
        evidence = self.evidence()
        async with self.sessions() as db:
            identifier = await self.manager.save_violation(db, evidence)
            retry = await self.manager.save_violation(db, evidence)
            self.assertEqual(identifier, retry)
            row = await db.get(Violation, identifier)
            self.assertTrue(row.event_fingerprint)
            self.assertTrue((self.root / row.image_path).is_file())
            self.assertEqual(
                await db.scalar(
                    select(func.count()).select_from(EvidenceWriteOperation),
                ),
                0,
            )
        self.assertEqual(len(list(self.root.rglob('*.png'))), 1)
        self.assertEqual(list(self.root.rglob('*.part')), [])

    async def test_empty_upload_leaves_no_file_or_record(self):
        async with self.sessions() as db:
            with self.assertRaises(ValueError):
                await self.manager.save_violation(
                    db,
                    self.evidence(image_file=UploadFile(BytesIO(b''))),
                )
            self.assertEqual(
                await db.scalar(select(func.count()).select_from(Violation)),
                0,
            )
        self.assertEqual(list(self.root.rglob('*.png')), [])
        self.assertEqual(list(self.root.rglob('*.part')), [])

    async def test_cancelled_read_cleans_partial_upload_and_recovers_intent(
        self,
    ):
        entered = asyncio.Event()

        Upload = partial(
            _CancelledReadCleansPartialUploadUpload,
            _test_entered=entered,
        )

        async with self.sessions() as db:
            task = asyncio.create_task(
                self.manager.save_violation(
                    db,
                    self.evidence(image_file=Upload()),
                ),
            )
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(list(self.root.rglob('*.part')), [])
        self.assertEqual(list(self.root.rglob('*.png')), [])
        await self.age_intents()
        await self.recovery.sweep()
        async with self.sessions() as db:
            self.assertEqual(
                await db.scalar(
                    select(func.count()).select_from(EvidenceWriteOperation),
                ),
                0,
            )

    async def test_unknown_commit_reply_keeps_the_committed_image(self):
        evidence = self.evidence()
        async with self.sessions() as db:
            commit = db.commit

            lost_reply = partial(
                _unknown_commit_reply_keeps_the_lost_reply,
                commit,
            )

            with patch.object(db, 'commit', lost_reply):
                with self.assertRaises(SQLAlchemyError):
                    await self.manager.save_violation(db, evidence)
        async with self.sessions() as db:
            row = await db.scalar(select(Violation))
            self.assertTrue((self.root / row.image_path).is_file())
            self.assertEqual(
                await self.manager.save_violation(db, evidence),
                row.id,
            )

    async def test_failure_before_commit_cleans_promoted_file(self):
        async with self.sessions() as db:
            with patch.object(
                db,
                'flush',
                new=AsyncMock(side_effect=SQLAlchemyError('offline')),
            ):
                with self.assertRaises(SQLAlchemyError):
                    await self.manager.save_violation(db, self.evidence())
        self.assertEqual(list(self.root.rglob('*.png')), [])
        self.assertEqual(list(self.root.rglob('*.part')), [])

    async def test_changed_image_metadata_or_time_conflicts_for_the_same_event(
        self,
    ):
        evidence = self.evidence()
        async with self.sessions() as db:
            identifier = await self.manager.save_violation(db, evidence)
        for changes in (
            {'model_version': 'v2'},
            {'detection_time': evidence.detection_time + timedelta(seconds=1)},
            {'image_file': UploadFile(BytesIO(image_bytes('blue')))},
        ):
            async with self.sessions() as db:
                with self.assertRaises(EvidenceEventConflictError):
                    await self.manager.save_violation(
                        db,
                        replace(evidence, **changes),
                    )
        async with self.sessions() as db:
            self.assertEqual(await db.scalar(select(Violation.id)), identifier)
        self.assertEqual(len(list(self.root.rglob('*.png'))), 1)

    async def test_legacy_record_is_verified_and_lazily_gets_a_fingerprint(
        self,
    ):
        evidence = self.evidence(detection_time=None)
        async with self.sessions() as db:
            identifier = await self.manager.save_violation(db, evidence)
            row = await db.get(Violation, identifier)
            row.event_fingerprint = None
            await db.commit()
            self.assertEqual(
                await self.manager.save_violation(db, evidence),
                identifier,
            )
            await db.refresh(row)
            self.assertTrue(row.event_fingerprint)

    async def test_recovery_skips_a_live_writer_then_removes_crash_orphans(
        self,
    ):
        async with self.sessions() as db:
            write = EvidenceWrite(self.root, self.root / 'orphan.png')
            await write.begin(db)
            await write.receive(UploadFile(BytesIO(image_bytes())))
            await write.promote()
            await self.age_intents()
            await self.recovery.sweep()
            self.assertTrue(write.final.exists())
            await db.rollback()
        await self.age_intents()
        await self.recovery.sweep()
        self.assertFalse(write.final.exists())

    async def test_storage_read_failure_removes_the_partial_file(self):
        write = EvidenceWrite(self.root, self.root / 'failed.png')
        upload = AsyncMock(
            read=AsyncMock(side_effect=OSError('storage offline')),
        )
        try:
            with self.assertRaises(ViolationImageReadError):
                await write.receive(upload)
        finally:
            await write.cleanup()
        self.assertFalse(write.part.exists())

    async def test_uncertain_uncommitted_upload_is_recovered(
        self,
    ):
        async with self.sessions() as db:
            with patch.object(
                db,
                'commit',
                new=AsyncMock(side_effect=SQLAlchemyError('lost reply')),
            ):
                with self.assertRaises(SQLAlchemyError):
                    await self.manager.save_violation(db, self.evidence())
        self.assertEqual(len(list(self.root.rglob('*.png'))), 1)
        await self.age_intents()
        await self.recovery.sweep()
        self.assertEqual(list(self.root.rglob('*.png')), [])

    async def test_recovery_never_removes_an_image_owned_by_a_committed_row(
        self,
    ):
        async with self.sessions() as db:
            identifier = await self.manager.save_violation(db, self.evidence())
            row = await db.get(Violation, identifier)
            db.add(
                EvidenceWriteOperation(
                    id=uuid4().hex,
                    path=row.image_path,
                    created_at=datetime.now(timezone.utc)
                    - timedelta(minutes=2),
                ),
            )
            await db.commit()
        await self.recovery.sweep()
        self.assertTrue((self.root / row.image_path).is_file())

    async def test_equivalent_json_reuses_the_same_fingerprint(
        self,
    ):
        evidence = self.evidence(detections_json='[[1,2,3,4,0.8,0,1]]')
        async with self.sessions() as db:
            identifier = await self.manager.save_violation(db, evidence)
            self.assertEqual(
                await self.manager.save_violation(
                    db,
                    replace(
                        evidence,
                        detections_json='[ [1.0,2.0,3.0,4.0,0.8,0,1.0] ]',
                    ),
                ),
                identifier,
            )

    async def test_directory_sync_failure_prevents_committing_evidence(
        self,
    ):
        _callbacks = _DirectorySyncFailurePreventsCommittingEvidenceCallbacks()
        _callbacks.testcase = self
        from examples.violation_records import evidence_storage as storage

        _callbacks.original = storage.sync_directory
        _callbacks.calls = []

        sync = _callbacks._call_sync

        async with self.sessions() as db:
            with patch.object(storage, 'sync_directory', side_effect=sync):
                with self.assertRaises(OSError):
                    await self.manager.save_violation(db, self.evidence())
            self.assertEqual(
                await db.scalar(select(func.count()).select_from(Violation)),
                0,
            )
        self.assertIn(self.root.resolve(), _callbacks.calls)
        await self.age_intents()
        await self.recovery.sweep()
        self.assertEqual(list(self.root.rglob('*.png')), [])
