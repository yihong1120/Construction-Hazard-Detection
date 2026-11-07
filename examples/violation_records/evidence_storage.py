from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from functools import partial
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from pydantic import Field
from sqlalchemy import select
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from examples.auth.database import engine
from examples.auth.models import EvidenceWriteOperation
from examples.auth.models import Violation
from examples.auth.operation_database import operation_database
from examples.shared.image_validation import EvidenceImageSettings
from examples.shared.image_validation import ImageValidationError
from examples.shared.image_validation import VerifiedImage
from examples.shared.image_validation import verify_image
from examples.shared.image_work import run_image_work
from examples.violation_records.settings import STATIC_DIR
from src.async_tasks import finish_on_cancel
from src.settings.base import EnvironmentSettings

logger = logging.getLogger(__name__)


class EmptyViolationImageError(ValueError):
    pass


class ViolationImageReadError(OSError):
    pass


def sessions_for(db):
    bind = db.bind
    if bind is None:
        raise RuntimeError('Evidence storage requires a bound database')
    return operation_database(bind).sessions


def sync_directory(path: Path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class EvidenceWrite:
    """Own the staged file until a committed row or recovery owns it."""

    def __init__(self, root: Path, final: Path, limits=None):
        self.id = uuid4().hex
        self.root = root.resolve()
        self.final = final.resolve()
        self.part = self.root / '.staging' / (self.id + '.part')
        self.committing = False
        self.image_sha256 = ''
        self.limits = limits or EvidenceImageSettings()
        self.image: VerifiedImage | None = None
        self._sessions = None
        self._journal_path = self.final

    async def begin(self, db):
        self._sessions = sessions_for(db)
        await db.execute(
            text('SELECT pg_advisory_xact_lock(hashtextextended(:id,4103))'),
            {'id': self.id},
        )

        await finish_on_cancel(self._record_journal())

    async def receive(self, upload, chunk_size=1024 * 1024):
        self.part.parent.mkdir(parents=True, exist_ok=True)
        await finish_on_cancel(asyncio.to_thread(sync_directory, self.root))
        digest = hashlib.sha256()
        wrote = False
        total = 0
        try:
            if hasattr(upload, 'seek'):
                await upload.seek(0)
            with self.part.open('xb') as output:
                while True:
                    chunk = await upload.read(chunk_size)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > self.limits.max_upload_bytes:
                        raise ImageValidationError(
                            'IMAGE_UPLOAD_LIMIT_EXCEEDED',
                            413,
                        )
                    wrote = True
                    digest.update(chunk)
                    await finish_on_cancel(
                        asyncio.to_thread(output.write, chunk),
                    )

                await finish_on_cancel(
                    asyncio.to_thread(_flush_evidence_file, output),
                )
        except OSError as error:
            raise ViolationImageReadError(
                'Evidence storage write failed',
            ) from error
        if not wrote:
            raise EmptyViolationImageError('Empty image file')
        self.image_sha256 = digest.hexdigest()
        self.image = await run_image_work(
            partial(verify_image, self.part, self.limits),
        )
        self.final = self.final.with_suffix(self.image.suffix)

    async def promote(self):
        if self.final != self._journal_path and self._sessions is not None:

            # Journal the new path before publishing a different format.
            await finish_on_cancel(self._repath_journal())
            self._journal_path = self.final

        await finish_on_cancel(asyncio.to_thread(self._promote_sync))

    async def cleanup(self):

        await finish_on_cancel(asyncio.to_thread(self._cleanup_sync))

    async def _record_journal(self):
        async with self._sessions() as journal:
            journal.add(
                EvidenceWriteOperation(
                    id=self.id,
                    path=self.final.relative_to(self.root).as_posix(),
                ),
            )
            await journal.commit()

    async def _repath_journal(self):
        async with self._sessions() as journal:
            record = await journal.get(EvidenceWriteOperation, self.id)
            if record is None:
                raise RuntimeError(
                    'Evidence write operation is missing',
                )
            record.path = self.final.relative_to(self.root).as_posix()
            await journal.commit()

    def _promote_sync(self):
        self.final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.part, self.final)
        sync_directory(self.part.parent)
        sync_directory(self.final.parent)
        sync_directory(self.root)

    def _cleanup_sync(self):
        paths = [self.part] if self.committing else [self.part, self.final]
        for path in paths:
            path.unlink(missing_ok=True)
            if path.parent.exists():
                sync_directory(path.parent)


class EvidenceRecoverySettings(EnvironmentSettings):
    model_config = {'env_prefix': 'EVIDENCE_RECOVERY_'}
    interval_seconds: float = Field(30, gt=0, allow_inf_nan=False)
    min_age_seconds: float = Field(60, ge=0, allow_inf_nan=False)
    batch_size: int = Field(128, ge=1, le=4096)
    budget_seconds: float = Field(2, gt=0, le=30, allow_inf_nan=False)
    retry_seconds: float = Field(5, gt=0, allow_inf_nan=False)
    max_retry_seconds: float = Field(300, ge=1, allow_inf_nan=False)


class EvidenceStorageMaintenance:

    def __init__(self, database=engine, root=STATIC_DIR, settings=None):
        self.database = operation_database(database)
        self.engine = self.database.engine
        self.sessions = self.database.sessions
        self.root = Path(root).resolve()
        self.settings = settings or EvidenceRecoverySettings()
        self.stop = asyncio.Event()

    def paths(self, record):
        final = (self.root / record.path).resolve()
        if not final.is_relative_to(self.root):
            raise ValueError('Evidence recovery path is outside storage')
        return self.root / '.staging' / (record.id + '.part'), final

    async def _defer(self, operation_id, error, failed=True):
        async with self.sessions() as db:
            await db.execute(text("SET LOCAL statement_timeout='1000ms'"))
            record = await db.get(EvidenceWriteOperation, operation_id)
            if record is None:
                return
            if failed:
                record.attempts += 1
            record.last_error = error
            delay = min(
                self.settings.max_retry_seconds,
                self.settings.retry_seconds * 2 ** min(record.attempts, 8),
            )
            record.next_attempt = datetime.now(timezone.utc) + timedelta(
                seconds=delay,
            )
            await finish_on_cancel(db.commit())

    async def _recover(self, operation_id):
        async with self.database.locks.begin() as lock:
            await lock.execute(text("SET LOCAL statement_timeout='1000ms'"))
            if not await lock.scalar(
                text(
                    'SELECT '
                    'pg_try_advisory_xact_lock(hashtextextended(:id,'
                    '4103))',
                ),
                {'id': operation_id},
            ):
                return False
            async with self.sessions() as db:
                await db.execute(text("SET LOCAL statement_timeout='1000ms'"))
                record = await db.get(EvidenceWriteOperation, operation_id)
                if record is None:
                    return True
                exists = await db.scalar(
                    select(Violation.id)
                    .where(Violation.image_path == record.path)
                    .limit(1),
                )
                part, final = self.paths(record)

                await finish_on_cancel(
                    asyncio.to_thread(
                        _remove_recovery_files,
                        part,
                        final,
                        bool(exists),
                    ),
                )
                await db.delete(record)
                await finish_on_cancel(db.commit())
                return True

    async def sweep(self):
        async with asyncio.timeout(self.settings.budget_seconds):
            await self._sweep()

    async def _sweep(self):
        deadline = time.monotonic() + self.settings.budget_seconds
        async with self.sessions() as db:
            await db.execute(text("SET LOCAL statement_timeout='1000ms'"))
            stamp = datetime.now(timezone.utc)
            ids = list(
                await db.scalars(
                    select(EvidenceWriteOperation.id)
                    .where(
                        EvidenceWriteOperation.created_at
                        <= stamp
                        - timedelta(seconds=self.settings.min_age_seconds),
                        EvidenceWriteOperation.next_attempt <= stamp,
                    )
                    .order_by(
                        EvidenceWriteOperation.next_attempt,
                        EvidenceWriteOperation.created_at,
                    )
                    .limit(self.settings.batch_size),
                ),
            )
        for operation_id in ids:
            if self.stop.is_set() or time.monotonic() >= deadline:
                return
            try:
                if not await self._recover(operation_id):
                    await self._defer(
                        operation_id,
                        'writer_active',
                        failed=False,
                    )
            except (OSError, ValueError, SQLAlchemyError) as error:
                logger.warning(
                    'Evidence recovery deferred operation=%s error=%s',
                    operation_id,
                    type(error).__name__,
                )
                await self._defer(operation_id, type(error).__name__)

    @asynccontextmanager
    async def lifespan(self):
        task = asyncio.create_task(self._run())
        try:
            yield self
        finally:
            self.stop.set()
            await finish_on_cancel(task)

    async def _run(self):
        while not self.stop.is_set():
            try:
                await asyncio.wait_for(
                    self.stop.wait(),
                    self.settings.interval_seconds,
                )
            except TimeoutError:
                try:
                    await self.sweep()
                except Exception as error:
                    logger.warning(
                        'Evidence maintenance deferred error=%s',
                        type(error).__name__,
                    )


def _remove_recovery_files(part: Path, final: Path, exists: bool) -> None:
    part.unlink(missing_ok=True)
    if not exists:
        final.unlink(missing_ok=True)
    for directory in {part.parent, final.parent}:
        if directory.exists():
            sync_directory(directory)


def _flush_evidence_file(output: BinaryIO) -> None:
    output.flush()
    os.fsync(output.fileno())
