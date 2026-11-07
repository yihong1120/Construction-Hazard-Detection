from __future__ import annotations

import tempfile
from datetime import datetime
from datetime import timezone
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import patch
from uuid import uuid4

from sqlalchemy import insert
from sqlalchemy import select
from starlette.datastructures import UploadFile

from examples.auth.models import Site
from examples.auth.models import Violation
from examples.local_notification_server import (
    notification_delivery_service as delivery_service,
)
from examples.local_notification_server.schemas import SiteNotifyRequest
from examples.violation_records.evidence import EvidenceEventConflictError
from examples.violation_records.evidence import ViolationEvidence
from examples.violation_records.site_scope import violation_site_scope
from examples.violation_records.violation_manager import ViolationManager
from tests.database_fixtures_test import OrmDatabase
from tests.image_fixtures_test import image_bytes


class EvidenceIdentityTest(OrmDatabase):
    async def asyncSetUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        await super().asyncSetUp()
        async with self.sessions() as db:
            db.add_all(
                [Site(id=1, name='Visible'), Site(id=2, name='Private')],
            )
            await db.commit()
        self.manager = ViolationManager(self.directory.name)

    async def asyncTearDown(self) -> None:
        await super().asyncTearDown()

    def evidence(self, **changes) -> ViolationEvidence:
        values = dict(
            site='Visible',
            site_id=1,
            stream_name='Camera',
            detection_time=datetime.now(timezone.utc),
            image_file=UploadFile(
                BytesIO(image_bytes()), filename='image.jpg',
            ),
            event_id=str(uuid4()),
        )
        return ViolationEvidence(**(values | changes))

    async def test_idempotent_upload_stores_one_row_and_one_image(
        self,
    ) -> None:
        evidence = self.evidence()
        async with self.sessions() as db:
            first = await self.manager.save_violation(db, evidence)
            retry = await self.manager.save_violation(db, evidence)
            self.assertEqual(first, retry)
            row = await db.get(Violation, first)
            self.assertEqual(
                (row.site_id, row.site, row.event_id),
                (1, 'Visible', evidence.event_id),
            )
        self.assertEqual(
            len(list(Path(self.directory.name).rglob('*.png'))),
            1,
        )

    async def test_event_id_cannot_be_replayed_into_another_site(self) -> None:
        evidence = self.evidence()
        async with self.sessions() as db:
            await self.manager.save_violation(db, evidence)
            with self.assertRaises(EvidenceEventConflictError):
                await self.manager.save_violation(
                    db,
                    self.evidence(
                        site='Private',
                        site_id=2,
                        event_id=evidence.event_id,
                    ),
                )

    async def test_rename_keeps_snapshot_and_name_reuse_grants_no_access(
        self,
    ) -> None:
        async with self.sessions() as db:
            identifier = await self.manager.save_violation(db, self.evidence())
            site = await db.get(Site, 1)
            site.name = 'Renamed'
            await db.commit()
            db.add(Site(id=3, name='Visible'))
            await db.commit()
            self.assertEqual(
                list(
                    (
                        await db.scalars(
                            select(Violation.id).where(
                                violation_site_scope(['Visible']),
                            ),
                        )
                    ).all(),
                ),
                [],
            )
            self.assertEqual(
                list(
                    (
                        await db.scalars(
                            select(Violation.id).where(
                                violation_site_scope(['Renamed']),
                            ),
                        )
                    ).all(),
                ),
                [identifier],
            )
            row = await db.get(Violation, identifier)
            self.assertEqual(row.site, 'Visible')

    async def test_legacy_rows_use_name_scope_until_backfilled(self) -> None:
        async with self.sessions() as db:
            await db.execute(
                insert(Violation).values(
                    id=1,
                    site='Visible',
                    site_id=None,
                    stream_name='Camera',
                    image_path='legacy.png',
                    violation_type_codes=[],
                ),
            )
            await db.commit()
            self.assertEqual(
                list(
                    (
                        await db.scalars(
                            select(Violation.id).where(
                                violation_site_scope(['Visible']),
                            ),
                        )
                    ).all(),
                ),
                [1],
            )
            self.assertEqual(
                list(
                    (
                        await db.scalars(
                            select(Violation.id).where(
                                violation_site_scope(['Private']),
                            ),
                        )
                    ).all(),
                ),
                [],
            )

    async def test_site_delete_cascades_evidence_even_with_loaded_relationship(
        self,
    ) -> None:
        async with self.sessions() as db:
            identifier = await self.manager.save_violation(db, self.evidence())
            site = await db.get(Site, 1)
            await db.refresh(site, ['violations'])
            await db.delete(site)
            await db.commit()
        async with self.sessions() as db:
            self.assertIsNone(await db.get(Violation, identifier))

    async def test_pending_notification_uses_id_after_name_reuse(self) -> None:
        request = SiteNotifyRequest(
            site='Visible', site_id=1, stream_name='Camera', violation_id=42,
            body={'warning_no_hardhat': {'count': 1}},
            type='violation', title='Alert',
            deep_link='/violations', metadata={},
        )
        redis = AsyncMock()
        redis.set.return_value = True
        redis.eval.return_value = 1
        delivery = AsyncMock(return_value={'success': True})
        async with self.sessions() as db:
            site = await db.get(Site, 1)
            site.name = 'Renamed'
            await db.commit()
            db.add(Site(id=3, name='Visible'))
            await db.commit()
            with patch.object(
                delivery_service,
                '_deliver_site_notification',
                delivery,
            ):
                result = await delivery_service.send_site_notification(
                    request, db, redis,
                )
                self.assertTrue(result['success'])
                call = delivery.await_args
                assert call is not None
                self.assertEqual(call.args[0].site, 'Renamed')
                self.assertEqual(
                    redis.set.await_args.args[0],
                    'fcm_notification_dedupe:1:42',
                )
                result = await delivery_service.send_site_notification(
                    request.model_copy(update={'site_id': 999}), db, redis,
                )
                self.assertFalse(result['success'])
                self.assertEqual(delivery.await_count, 1)
