from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from examples.db_management.services import site_media_cleanup


class TestSiteMediaCleanup(unittest.IsolatedAsyncioTestCase):
    """Verify durable media cleanup claims do not hold DB locks for I/O."""

    def _db_with_jobs(self, jobs: list[MagicMock]) -> MagicMock:
        """Perform db with jobs.

        Args:
            jobs: Value used by this callable.

        Returns:
            The callable result.
        """
        db = MagicMock()
        db.scalars = AsyncMock(
            return_value=MagicMock(all=MagicMock(return_value=jobs)),
        )
        db.commit = AsyncMock()
        db.execute = AsyncMock()
        return db

    async def test_drain_claims_then_completes_jobs(self) -> None:
        """Test drain claims then completes jobs."""
        job = MagicMock(
            id=3,
            path='/tmp/evidence.jpg',
            attempt_count=0,
            completed_at=None,
            lease_token=None,
            lease_expires_at=None,
            last_error=None,
        )
        db = self._db_with_jobs([job])

        with patch.object(site_media_cleanup, '_delete_file') as delete_file:
            completed = await site_media_cleanup.drain_site_media_cleanup_jobs(
                db,
            )

        self.assertEqual(completed, 1)
        delete_file.assert_called_once_with('/tmp/evidence.jpg')
        self.assertEqual(job.attempt_count, 1)
        self.assertIsInstance(job.completed_at, datetime)
        self.assertIsNone(job.lease_token)
        self.assertIsNone(job.lease_expires_at)
        self.assertEqual(db.commit.await_count, 2)

    async def test_drain_releases_failed_job_after_backoff(self) -> None:
        """Test drain releases failed job after backoff."""
        job = MagicMock(
            id=4,
            path='/tmp/evidence.jpg',
            attempt_count=0,
            completed_at=None,
            lease_token=None,
            lease_expires_at=None,
            last_error=None,
        )
        db = self._db_with_jobs([job])

        with patch.object(
            site_media_cleanup,
            '_delete_file',
            side_effect=OSError('disk unavailable'),
        ):
            completed = await site_media_cleanup.drain_site_media_cleanup_jobs(
                db,
            )

        self.assertEqual(completed, 0)
        self.assertEqual(job.attempt_count, 1)
        self.assertIsNone(job.completed_at)
        self.assertIsNone(job.lease_token)
        self.assertIsInstance(job.lease_expires_at, datetime)
        self.assertEqual(job.last_error, 'disk unavailable')
        self.assertEqual(db.commit.await_count, 2)

    async def test_enqueue_for_site_uses_insert_select(self) -> None:
        """Test enqueue for site uses insert select."""
        db = self._db_with_jobs([])

        await site_media_cleanup.enqueue_site_media_cleanup_for_site(
            'Roadwork',
            db,
        )

        statement = str(db.execute.await_args.args[0])
        self.assertIn('INSERT INTO site_media_cleanup_jobs', statement)
        self.assertIn('SELECT DISTINCT violations.image_path', statement)

    async def test_enqueue_deduplicates_paths_and_empty_drains_do_no_io(
        self,
    ) -> None:
        """Enqueue is deduplicated and an empty queue commits no lease."""
        db = self._db_with_jobs([])

        await site_media_cleanup.enqueue_site_media_cleanup_jobs(
            ['first.png', '', 'first.png', 'second.png'],
            db,
        )
        statement = db.execute.await_args.args[0]
        parameters = statement.compile().params.values()
        self.assertIn('first.png', parameters)
        self.assertIn('second.png', parameters)

        completed = await site_media_cleanup.drain_site_media_cleanup_jobs(db)
        self.assertEqual(completed, 0)
        db.commit.assert_not_awaited()

    def test_delete_file_ignores_non_file_paths(self) -> None:
        """Deletion only accepts concrete regular files."""
        with patch.object(
            site_media_cleanup.Path,
            'is_file',
            return_value=False,
        ):
            site_media_cleanup._delete_file('/tmp/missing.png')

        with TemporaryDirectory() as directory:
            evidence = Path(directory) / 'evidence.png'
            evidence.write_bytes(b'image')
            site_media_cleanup._delete_file(str(evidence))
            self.assertFalse(evidence.exists())
