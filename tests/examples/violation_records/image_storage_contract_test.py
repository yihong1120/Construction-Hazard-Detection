from __future__ import annotations

import tempfile
from dataclasses import replace
from functools import partial
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import func
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from starlette.datastructures import UploadFile

from examples.auth.models import EvidenceWriteOperation
from examples.auth.models import Site
from examples.auth.models import Violation
from examples.shared.evidence_metadata import validate_metadata
from examples.shared.image_validation import ImageValidationError
from examples.violation_records.evidence import ViolationEvidence
from examples.violation_records.evidence_storage import (
    EvidenceRecoverySettings,
)
from examples.violation_records.evidence_storage import (
    EvidenceStorageMaintenance,
)
from examples.violation_records.violation_manager import ViolationManager
from tests.database_fixtures_test import OrmDatabase
from tests.image_fixtures_test import image_bytes


async def _lost_commit_response_retains_the_lost_reply(commit, /):
    await commit()
    raise SQLAlchemyError('reply lost')


class ImageStorageContractTest(OrmDatabase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.manager = ViolationManager(self.root)
        async with self.sessions() as db:
            db.add(Site(id=1, name='site'))
            await db.commit()

    def evidence(self, data):
        return ViolationEvidence(
            site='site',
            site_id=1,
            stream_name='cam',
            image_file=UploadFile(BytesIO(data), filename='claimed.png'),
        )

    async def test_jpeg_keeps_original_bytes_and_uses_actual_extension(self):
        original = image_bytes(image_format='JPEG')
        async with self.sessions() as db:
            identifier = await self.manager.save_violation(
                db,
                self.evidence(original),
            )
            record = await db.get(Violation, identifier)
            path = self.root / record.image_path
            self.assertEqual(path.suffix, '.jpg')
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(
                await db.scalar(
                    select(func.count()).select_from(EvidenceWriteOperation),
                ),
                0,
            )

    async def test_invalid_images_and_limits_leave_no_published_file(
        self,
    ):
        cases = [
            (b'not an image', 400),
            (image_bytes()[:40], 400),
            (image_bytes(image_format='GIF'), 400),
            (image_bytes(), 413),
        ]
        for data, status in cases:
            with self.subTest(status=status):
                self.manager.image_limits.max_upload_bytes = (
                    1 if status == 413 else 1024
                )
                async with self.sessions() as db:
                    with self.assertRaises(ImageValidationError) as error:
                        await self.manager.save_violation(
                            db,
                            self.evidence(data),
                        )
                    self.assertEqual(error.exception.status_code, status)
                self.assertEqual(list(self.root.rglob('*.part')), [])
                self.assertEqual(list(self.root.rglob('*.png')), [])
        self.manager.image_limits.max_upload_bytes = 1024
        self.manager.image_limits.max_pixels = 1
        async with self.sessions() as db:
            with self.assertRaises(ImageValidationError) as error:
                await self.manager.save_violation(
                    db,
                    self.evidence(image_bytes()),
                )
            self.assertEqual(
                error.exception.code,
                'IMAGE_DIMENSIONS_EXCEEDED',
            )
            self.assertEqual(
                await db.scalar(
                    select(func.count()).select_from(Violation),
                ),
                0,
            )

    async def test_changed_extension_has_durable_recovery(self):
        with patch(
            'examples.violation_records.evidence_storage.os.replace',
            side_effect=OSError('publication interrupted'),
        ):
            async with self.sessions() as db:
                with self.assertRaises(OSError):
                    await self.manager.save_violation(
                        db,
                        self.evidence(image_bytes(image_format='JPEG')),
                    )
        async with self.sessions() as db:
            pending = await db.scalar(select(EvidenceWriteOperation))
            self.assertTrue(pending.path.endswith('.jpg'))
            identifier = pending.id
        recovery = EvidenceStorageMaintenance(
            self.engine,
            self.root,
            EvidenceRecoverySettings(
                min_age_seconds=0,
            ),
        )
        self.assertTrue(await recovery._recover(identifier))
        self.assertEqual(list(self.root.rglob('*.jpg')), [])
        self.assertEqual(list(self.root.rglob('*.part')), [])

    async def test_lost_commit_response_retains_the_jpeg_record_and_image(
        self,
    ):
        async with self.sessions() as db:
            commit = db.commit

            lost_reply = partial(
                _lost_commit_response_retains_the_lost_reply,
                commit,
            )

            with patch.object(db, 'commit', lost_reply):
                with self.assertRaises(SQLAlchemyError):
                    await self.manager.save_violation(
                        db,
                        self.evidence(image_bytes(image_format='JPEG')),
                    )
        async with self.sessions() as db:
            record = await db.scalar(select(Violation))
            self.assertTrue(record.image_path.endswith('.jpg'))
            self.assertTrue((self.root / record.image_path).is_file())
            self.assertEqual(
                await db.scalar(
                    select(func.count()).select_from(EvidenceWriteOperation),
                ),
                0,
            )

    async def test_metadata_reuse_preserves_legacy_null(
        self,
    ):
        evidence = replace(
            self.evidence(image_bytes()),
            warnings_json='{"warning_no_hardhat":{"count":1}}',
            cone_polygon_json='null',
        )
        parsed = validate_metadata(evidence)
        self.assertEqual(parsed.source.cone_polygon_json, 'null')
        self.assertIs(validate_metadata(parsed), parsed)
        with self.assertRaises(ValueError):
            parsed.with_source(warnings_json='{}')
        with patch(
            'examples.shared.evidence_metadata.'
            'WARNING_PAYLOAD_ADAPTER.validate_json',
            side_effect=AssertionError('metadata parsed twice'),
        ):
            async with self.sessions() as db:
                await self.manager.save_violation(db, parsed)
