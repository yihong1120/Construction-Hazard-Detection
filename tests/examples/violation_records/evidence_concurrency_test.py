from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime
from datetime import timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import uuid4

from sqlalchemy import func
from sqlalchemy import select
from starlette.datastructures import UploadFile

from examples.auth.models import Site
from examples.auth.models import Violation
from examples.violation_records.evidence import ViolationEvidence
from examples.violation_records.violation_manager import ViolationManager
from tests.database_fixtures_test import OrmDatabase
from tests.image_fixtures_test import image_bytes


class _ConcurrentRetriesStoreOneRowAndOneCallbacks:
    both_read: Any
    event_id: Any
    manager: Any
    original: Any
    reads: Any
    testcase: EvidenceConcurrencyTest
    timestamp: Any

    async def _call_existing(self, db, evidence, write):

        result = await self.original(db, evidence, write)
        self.reads += 1
        if self.reads == 2:
            self.both_read.set()
        if self.reads <= 2:
            await asyncio.wait_for(self.both_read.wait(), timeout=5)
        return result

    async def _call_upload(self):
        async with self.testcase.sessions() as db:
            return await self.manager.save_violation(
                db,
                ViolationEvidence(
                    site='Site',
                    site_id=1,
                    stream_name='Camera',
                    event_id=self.event_id,
                    detection_time=self.timestamp,
                    image_file=UploadFile(BytesIO(image_bytes())),
                ),
            )


class EvidenceConcurrencyTest(OrmDatabase):
    async def test_concurrent_retries_store_one_row_and_one_file(self) -> None:
        _callbacks = _ConcurrentRetriesStoreOneRowAndOneCallbacks()
        _callbacks.testcase = self
        async with self.sessions() as db:
            db.add(Site(id=1, name='Site'))
            await db.commit()
        with tempfile.TemporaryDirectory() as root:
            _callbacks.manager = ViolationManager(root)
            _callbacks.event_id = str(uuid4())
            _callbacks.both_read = asyncio.Event()
            _callbacks.original = _callbacks.manager._existing_event
            _callbacks.reads = 0

            existing = _callbacks._call_existing

            _callbacks.timestamp = datetime.now(timezone.utc)

            upload = _callbacks._call_upload

            with patch.object(_callbacks.manager, '_existing_event', existing):
                first, second = await asyncio.gather(upload(), upload())
            self.assertEqual(first, second)
            self.assertGreaterEqual(_callbacks.reads, 3)
            async with self.sessions() as db:
                self.assertEqual(
                    await db.scalar(
                        select(func.count()).select_from(Violation),
                    ),
                    1,
                )
            self.assertEqual(len(list(Path(root).rglob('*.png'))), 1)
