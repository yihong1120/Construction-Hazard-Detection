from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from sqlalchemy.exc import SQLAlchemyError

from examples.auth.models import Violation
from examples.violation_records import violation_manager

ViolationManager = violation_manager.ViolationManager


class TestViolationManager(unittest.IsolatedAsyncioTestCase):
    """Verify upload-only violation persistence."""

    async def test_save_violation_streams_upload_and_returns_identifier(
        self,
    ) -> None:
        """The manager accepts only a bounded asynchronous upload source."""
        with tempfile.TemporaryDirectory() as directory:
            manager = ViolationManager(directory)
            upload = AsyncMock()
            upload.read = AsyncMock(side_effect=[b'image', b''])
            db = SimpleNamespace(
                add=MagicMock(),
                commit=AsyncMock(),
                refresh=AsyncMock(
                    side_effect=lambda row: setattr(row, 'id', 42),
                ),
                execute=AsyncMock(
                    return_value=SimpleNamespace(
                        scalar_one_or_none=lambda: None,
                    ),
                ),
                rollback=AsyncMock(),
            )

            violation_id = await manager.save_violation(
                db=db,
                site='Site A',
                stream_name='Camera A',
                detection_time=datetime(2026, 8, 23),
                image_file=upload,
            )

            self.assertEqual(violation_id, 42)
            violation = db.add.call_args.args[0]
            self.assertIsInstance(violation, Violation)
            self.assertTrue((Path(directory) / violation.image_path).exists())

    async def test_empty_upload_raises_a_domain_error(self) -> None:
        """An empty upload cannot create a database record or return None."""
        with tempfile.TemporaryDirectory() as directory:
            manager = ViolationManager(directory)
            upload = AsyncMock()
            upload.read = AsyncMock(return_value=b'')
            db = SimpleNamespace(
                add=MagicMock(),
                commit=AsyncMock(),
                execute=AsyncMock(),
                rollback=AsyncMock(),
            )

            with self.assertRaises(ValueError):
                await manager.save_violation(
                    db=db,
                    site='Site A',
                    stream_name='Camera A',
                    detection_time=datetime(2026, 8, 23),
                    image_file=upload,
                )

            db.add.assert_not_called()

    async def test_save_validates_metadata_and_cleans_up_failed_inserts(
        self,
    ) -> None:
        """Validated detector data is persisted and failures remove media."""
        with tempfile.TemporaryDirectory() as directory:
            manager = ViolationManager(directory)
            upload = AsyncMock()
            upload.read = AsyncMock(side_effect=[b'image', b''])
            db = SimpleNamespace(rollback=AsyncMock())

            with patch.object(
                manager,
                '_insert_violation_record',
                AsyncMock(side_effect=SQLAlchemyError('offline')),
            ):
                with self.assertRaises(SQLAlchemyError):
                    await manager.save_violation(
                        db=db,
                        site='Site A',
                        stream_name='Camera A',
                        detection_time=datetime(2026, 8, 23),
                        image_file=upload,
                        warnings_json='{"warning_no_hardhat":{"count":1}}',
                        detections_json='[[0,0,0,0,0,0,0]]',
                    )

            self.assertEqual(list(Path(directory).rglob('*.png')), [])
            db.rollback.assert_awaited_once()

    async def test_write_upload_translates_storage_read_errors(
        self,
    ) -> None:
        """Storage read failures leave no partial evidence file behind."""
        with tempfile.TemporaryDirectory() as directory:
            manager = ViolationManager(directory)
            upload = AsyncMock()
            upload.read = AsyncMock(side_effect=OSError('storage offline'))
            image_path = Path(directory) / 'evidence.png'

            with self.assertRaises(violation_manager.ViolationImageReadError):
                await manager._write_upload_file(upload, image_path, 32)

            self.assertFalse(image_path.exists())
