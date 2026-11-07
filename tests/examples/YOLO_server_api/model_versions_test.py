from __future__ import annotations

from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from examples.auth.models import DetectionModelCatalog
from examples.auth.models import DetectionModelVersion
from examples.YOLO_server_api import catalog_repository as repository
from examples.YOLO_server_api import model_registry as registry
from tests.database_fixtures_test import OrmDatabase
from tests.image_fixtures_test import image_bytes


class EvidenceModelVersionTest(OrmDatabase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.old = registry.ModelEntry(
            id='detector',
            display_name='Detector',
            version='v1',
            artifact='/missing/old.pt',
            sha256='a' * 64,
            capabilities=['image', 'stream'],
            tenant_ids=['t1'],
            site_ids=[20],
            classes=[{'id': 7, 'code': 'helmet', 'display_name': 'Helmet'}],
        )
        self.new = self.old.model_copy(
            update={
                'version': 'v2',
                'sha256': 'b' * 64,
                'classes': [
                    registry.ModelClass(
                        id=42,
                        code='person',
                        display_name='Person',
                    ),
                ],
            },
        )
        self.repo = repository.CatalogRepository(self.sessions)
        patched = patch.object(repository, 'catalog_repository', self.repo)
        patched.start()
        self.addCleanup(patched.stop)
        async with self.sessions() as db:
            db.add(
                DetectionModelCatalog(
                    model_key='detector',
                    definition=self.old.model_dump(exclude={'id'}),
                ),
            )
            await db.commit()

    async def update(self, entry):
        async with self.sessions() as db:
            current = await db.get(DetectionModelCatalog, entry.id)
            current.definition = entry.model_dump(exclude={'id'})
            await db.commit()

    async def current(self):
        return await registry.select_model_async(
            'detector',
            None,
            't1',
            'user',
            'stream',
            20,
        )

    async def test_old_evidence_uses_history_without_loading_weights(
        self,
    ):
        await self.update(self.new)
        with patch.object(
            registry,
            'verify_artifact',
            side_effect=AssertionError('weights not needed'),
        ):
            evidence = await registry.resolve_evidence_version(
                await self.current(),
                'v1',
            )
        self.assertEqual(evidence.version, 'v1')
        self.assertEqual(
            [(c.id, c.code) for c in evidence.classes],
            [(7, 'helmet')],
        )
        current = await self.current()
        self.assertEqual(current.version, 'v2')
        with self.assertRaises(HTTPException) as raised:
            await registry.resolve_evidence_version(current, 'unknown')
        self.assertEqual(raised.exception.status_code, 409)

    async def test_history_never_reinstates_revoked_current_permissions(self):
        await self.update(self.new.model_copy(update={'tenant_ids': []}))
        with self.assertRaises(HTTPException) as raised:
            await self.current()
        self.assertEqual(raised.exception.status_code, 403)
        async with self.sessions() as db:
            self.assertIsNotNone(
                await db.get(DetectionModelVersion, ('detector', 'v1')),
            )

    async def test_catalog_cannot_reuse_version_with_changed_hash_or_classes(
        self,
    ):
        for changes in ({'sha256': 'b' * 64}, {'classes': self.new.classes}):
            with self.assertRaises(IntegrityError):
                await self.update(self.old.model_copy(update=changes))
        self.assertEqual((await self.current()).sha256, 'a' * 64)

    async def test_rollback_keeps_published_history_and_history_is_immutable(
        self,
    ):
        await self.update(self.new)
        await self.update(self.old)
        evidence = await registry.resolve_evidence_version(
            await self.current(),
            'v2',
        )
        self.assertEqual(evidence.classes[0].id, 42)
        async with self.sessions() as db:
            row = await db.get(DetectionModelVersion, ('detector', 'v1'))
            row.sha256 = 'c' * 64
            with self.assertRaises(IntegrityError):
                await db.commit()
            await db.rollback()
        async with self.sessions() as db:
            await db.delete(await db.get(DetectionModelCatalog, 'detector'))
            await db.commit()
            self.assertEqual(
                len((await db.scalars(select(DetectionModelVersion))).all()),
                2,
            )
        with self.assertRaises(HTTPException):
            await self.current()

    async def test_delayed_upload_validates_against_the_historical_class_ids(
        self,
    ):
        from io import BytesIO
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        from fastapi import UploadFile
        from examples.violation_records import (
            violation_upload_service as uploads,
        )
        from examples.violation_records.evidence import ViolationEvidence

        await self.update(self.new)
        evidence = ViolationEvidence(
            site='Site',
            site_id=20,
            stream_name='Camera',
            model_id='detector',
            model_version='v1',
            detections_json='[[0,0,20,20,0.9,7,1]]',
            image_file=UploadFile(
                file=BytesIO(image_bytes()),
                filename='evidence.jpg',
            ),
        )
        db = SimpleNamespace(
            get=AsyncMock(
                return_value=SimpleNamespace(tenant_id='t1', role='user'),
            ),
        )
        credentials = SimpleNamespace(
            subject={'username': 'operator', 'user_id': 1, 'tenant_id': 't1'},
        )
        with patch.object(
            uploads,
            'require_site',
            new=AsyncMock(),
        ), patch.object(
            uploads.violation_manager,
            'save_violation',
            new=AsyncMock(return_value=17),
        ) as save:
            result = await uploads.upload_violation(evidence, db, credentials)
        self.assertEqual(result.violation_id, 17)
        stored = save.await_args.args[1]
        self.assertEqual(stored.source.model_version, 'v1')
        self.assertEqual(stored.source.model_classes[0]['id'], 7)

    async def test_history_outage_is_retryable_instead_of_unknown_version(
        self,
    ):
        from unittest.mock import AsyncMock

        await self.update(self.new)
        current = await self.current()
        with patch.object(
            self.repo,
            'version',
            new=AsyncMock(side_effect=RuntimeError('offline')),
        ):
            with self.assertRaises(HTTPException) as raised:
                await registry.resolve_evidence_version(current, 'v1')
        self.assertEqual(raised.exception.status_code, 503)
