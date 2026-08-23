from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from typing import Final
from uuid import uuid4

from sqlalchemy import or_
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.models import SiteMediaCleanupJob
from examples.auth.models import Violation

logger = logging.getLogger(__name__)
_cleanup_enqueue_chunk_size: Final[int] = 250
_cleanup_error_max_length: Final[int] = 500
_cleanup_lease_seconds: Final[int] = 300
_cleanup_file_concurrency: Final[int] = 8
_cleanup_retry_max_seconds: Final[int] = 3600


def _chunks(values: list[str], size: int) -> Iterator[list[str]]:
    """Yield bounded chunks without allocating a second list of chunks."""
    for start in range(0, len(values), size):
        yield values[start: start + size]


async def enqueue_site_media_cleanup_jobs(
    image_paths: list[str],
    db: AsyncSession,
) -> None:
    """Persist unique image deletions before the owning site is removed."""
    paths = list(dict.fromkeys(path for path in image_paths if path))
    for path_chunk in _chunks(paths, _cleanup_enqueue_chunk_size):
        await db.execute(
            pg_insert(SiteMediaCleanupJob)
            .values([{'path': path} for path in path_chunk])
            .on_conflict_do_update(
                index_elements=['path'],
                set_={
                    'attempt_count': 0,
                    'last_error': None,
                    'completed_at': None,
                    'lease_token': None,
                    'lease_expires_at': None,
                },
            ),
        )


async def enqueue_site_media_cleanup_for_site(
    site_name: str,
    db: AsyncSession,
) -> None:
    """Queue a site's evidence paths inside PostgreSQL without loading them."""
    source_paths = (
        select(Violation.image_path)
        .where(Violation.site == site_name, Violation.image_path.is_not(None))
        .distinct()
    )
    await db.execute(
        pg_insert(SiteMediaCleanupJob)
        .from_select(['path'], source_paths)
        .on_conflict_do_update(
            index_elements=['path'],
            set_={
                'attempt_count': 0,
                'last_error': None,
                'completed_at': None,
                'lease_token': None,
                'lease_expires_at': None,
            },
        ),
    )


def _delete_file(path: str) -> None:
    """Delete one regular file without following a directory cleanup path."""
    candidate = Path(path)
    if candidate.is_file():
        candidate.unlink(missing_ok=True)


async def drain_site_media_cleanup_jobs(
    db: AsyncSession,
    *,
    limit: int = 100,
) -> int:
    """Claim then delete a bounded set of durable media-cleanup jobs.

    A short transaction leases rows with ``SKIP LOCKED`` before filesystem I/O
    begins.  A crashed worker's lease expires, so another worker can retry
    without holding database row locks while deleting files.
    """
    now = datetime.now(timezone.utc)
    lease_token = str(uuid4())
    lease_expires_at = now + timedelta(seconds=_cleanup_lease_seconds)
    jobs = list(
        (
            await db.scalars(
                select(SiteMediaCleanupJob)
                .where(
                    SiteMediaCleanupJob.completed_at.is_(None),
                    or_(
                        SiteMediaCleanupJob.lease_expires_at.is_(None),
                        SiteMediaCleanupJob.lease_expires_at <= now,
                    ),
                )
                .order_by(SiteMediaCleanupJob.id)
                .limit(limit)
                .with_for_update(skip_locked=True),
            )
        ).all(),
    )
    if not jobs:
        return 0

    for job in jobs:
        job.attempt_count += 1
        job.lease_token = lease_token
        job.lease_expires_at = lease_expires_at
    await db.commit()

    semaphore = asyncio.Semaphore(_cleanup_file_concurrency)

    async def delete_claimed(job: SiteMediaCleanupJob) -> OSError | None:
        """Perform delete claimed.

        Args:
            job: Value used by this callable.

        Returns:
            The callable result.
        """
        try:
            async with semaphore:
                await asyncio.to_thread(_delete_file, job.path)
        except OSError as exc:
            return exc
        return None

    errors = await asyncio.gather(*(delete_claimed(job) for job in jobs))
    completed = 0
    finished_at = datetime.now(timezone.utc)
    for job, error in zip(jobs, errors, strict=True):
        if error is not None:
            job.last_error = str(error)[:_cleanup_error_max_length]
            job.lease_token = None
            job.lease_expires_at = finished_at + timedelta(
                seconds=min(
                    _cleanup_retry_max_seconds,
                    2 ** min(job.attempt_count, 12),
                ),
            )
            logger.warning(
                'Site media cleanup failed job_id=%s error_type=%s',
                job.id,
                type(error).__name__,
            )
        else:
            job.completed_at = finished_at
            job.last_error = None
            job.lease_token = None
            job.lease_expires_at = None
            completed += 1
    await db.commit()
    return completed
