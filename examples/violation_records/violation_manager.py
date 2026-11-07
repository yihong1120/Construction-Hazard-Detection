from __future__ import annotations

import asyncio
import hashlib
import uuid
from dataclasses import replace
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Final

from sqlalchemy import delete
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.models import EvidenceWriteOperation
from examples.auth.models import Site
from examples.auth.models import StreamConfig
from examples.auth.models import Violation
from examples.shared.evidence_metadata import validate_metadata
from examples.shared.evidence_metadata import ValidatedMetadata
from examples.shared.filename_utils import sanitize_filename
from examples.shared.image_validation import EvidenceImageSettings
from examples.violation_records.evidence import EvidenceEventConflictError
from examples.violation_records.evidence import ViolationEvidence
from examples.violation_records.evidence_fingerprint import (
    evidence_fingerprint,
)
from examples.violation_records.evidence_fingerprint import normalized_evidence
from examples.violation_records.evidence_storage import (
    EmptyViolationImageError,
)
from examples.violation_records.evidence_storage import EvidenceWrite
from examples.violation_records.evidence_storage import ViolationImageReadError
from examples.violation_records.settings import STATIC_DIR
from examples.violation_records.violation_types import (
    violation_type_codes,
)
from src.async_tasks import finish_on_cancel

__all__ = [
    'ViolationManager',
    'EmptyViolationImageError',
    'ViolationImageReadError',
]

_upload_chunk_size: Final[int] = 1024 * 1024


class ViolationManager:
    """Store violation evidence images and their database records.

    Images are persisted before their matching database row so a committed row
    never points to a file that was not successfully written.
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        """Initialise a manager with a base directory for evidence images.

        Args:
            base_dir: Optional base directory; defaults to configured static
                media storage.
        """
        self.base_dir: Path = Path(base_dir or STATIC_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.image_limits = EvidenceImageSettings()

    async def save_violation(
        self,
        db: AsyncSession,
        evidence: ViolationEvidence | ValidatedMetadata[ViolationEvidence],
        *,
        chunk_size: int = _upload_chunk_size,
    ) -> int:
        parsed = validate_metadata(evidence)
        evidence = parsed.source
        if evidence.site_id is None:
            site_id = await db.scalar(
                select(Site.id).where(Site.name == evidence.site),
            )
            evidence = replace(evidence, site_id=site_id)
        detection_time = (
            evidence.detection_time or datetime.now(timezone.utc)
        ).astimezone(timezone.utc)
        write = EvidenceWrite(
            self.base_dir,
            self._build_image_path(detection_time),
            self.image_limits,
        )
        fingerprint = None
        try:
            await write.begin(db)
            await write.receive(evidence.image_file, chunk_size)
            fingerprint = evidence_fingerprint(evidence, write.image_sha256)
            if evidence.event_id is not None:
                existing = await self._existing_event(db, evidence, write)
                if existing is not None:
                    await write.cleanup()
                    await db.execute(
                        delete(EvidenceWriteOperation).where(
                            EvidenceWriteOperation.id == write.id,
                        ),
                    )
                    await finish_on_cancel(db.commit())
                    return existing
            await write.promote()
            stream_config_id = await self._find_stream_config_id(
                db,
                evidence.site,
                evidence.stream_name,
                evidence.site_id,
            )
            record = Violation(
                site=evidence.site,
                site_id=evidence.site_id,
                event_id=evidence.event_id,
                event_fingerprint=fingerprint,
                stream_name=evidence.stream_name,
                stream_config_id=stream_config_id,
                detection_time=detection_time,
                image_path=write.final.relative_to(
                    self.base_dir.resolve(),
                ).as_posix(),
                warnings_json=evidence.warnings_json,
                detections_json=evidence.detections_json,
                violation_type_codes=(
                    violation_type_codes(parsed.warnings)
                    if parsed.warnings is not None
                    else []
                ),
                cone_polygon_json=evidence.cone_polygon_json,
                pole_polygon_json=evidence.pole_polygon_json,
                model_id=evidence.model_id,
                model_version=evidence.model_version,
                model_classes=evidence.model_classes,
            )
            db.add(record)
            await db.flush()
            identifier = record.id
            await db.execute(
                delete(EvidenceWriteOperation).where(
                    EvidenceWriteOperation.id == write.id,
                ),
            )
            write.committing = True
            await finish_on_cancel(db.commit())
            return identifier
        except IntegrityError:
            await finish_on_cancel(db.rollback())
            write.committing = False
            if evidence.event_id is not None and fingerprint is not None:
                existing = await self._existing_event(db, evidence, write)
                if existing is not None:
                    await write.cleanup()
                    await db.execute(
                        delete(EvidenceWriteOperation).where(
                            EvidenceWriteOperation.id == write.id,
                        ),
                    )
                    await finish_on_cancel(db.commit())
                    return existing
            raise
        except BaseException:
            await finish_on_cancel(db.rollback())
            raise
        finally:
            await write.cleanup()

    def _build_image_path(self, detection_time: datetime) -> Path:
        """Create a date directory and return a unique evidence image path.

        Args:
            detection_time: Detection time used to partition image storage.

        Returns:
            Provisional image path; verification selects the extension.
        """
        date_folder: str = detection_time.strftime('%Y-%m-%d')
        day_dir: Path = self.base_dir / date_folder
        day_dir.mkdir(parents=True, exist_ok=True)

        filename: str = sanitize_filename(f'{uuid.uuid4()}.png')
        return day_dir / filename

    async def _find_stream_config_id(
        self,
        db: AsyncSession,
        site: str,
        stream_name: str,
        site_id: int | None = None,
    ) -> int | None:
        """Resolve the current camera configuration identifier.

        Args:
            db: Database session used to find the configured camera.
            site: Site name associated with the upload.
            stream_name: Camera name associated with the upload.

        Returns:
            Stable stream configuration identifier when the camera exists.
        """
        statement = (
            select(StreamConfig.id)
            .join(Site, StreamConfig.site_id == Site.id)
            .where(
                (
                    Site.id == site_id
                    if site_id is not None
                    else Site.name == site
                ),
                StreamConfig.stream_name == stream_name,
            )
        )
        return (await db.execute(statement)).scalar_one_or_none()

    async def _existing_event(
        self,
        db: AsyncSession,
        evidence: ViolationEvidence,
        write: EvidenceWrite,
    ) -> int | None:
        row = await db.scalar(
            select(Violation)
            .where(Violation.event_id == evidence.event_id)
            .with_for_update(),
        )
        if row is None:
            return None
        if (
            row.site_id != evidence.site_id
            or row.stream_name != evidence.stream_name
        ):
            raise EvidenceEventConflictError(
                'Event identifier belongs to another resource',
            )
        actual = evidence_fingerprint(evidence, write.image_sha256)
        if row.event_fingerprint is None:
            path = (self.base_dir / row.image_path).resolve()
            if (
                not path.is_relative_to(self.base_dir.resolve())
                or not path.is_file()
            ):
                raise EvidenceEventConflictError(
                    'Legacy evidence cannot be verified',
                )

            try:
                old = normalized_evidence(
                    replace(
                        evidence,
                        detection_time=(
                            row.detection_time
                            if evidence.detection_time is not None
                            else None
                        ),
                        model_id=row.model_id,
                        model_version=row.model_version,
                        warnings_json=row.warnings_json,
                        detections_json=row.detections_json,
                        cone_polygon_json=row.cone_polygon_json,
                        pole_polygon_json=row.pole_polygon_json,
                    ),
                )
            except (ValueError, TypeError) as error:
                raise EvidenceEventConflictError(
                    'Legacy evidence cannot be verified',
                ) from error
            expected = evidence_fingerprint(
                old,
                await asyncio.to_thread(_file_digest, path),
            )
        else:
            expected = row.event_fingerprint
        if expected != actual:
            raise EvidenceEventConflictError(
                'Event identifier has different evidence content',
            )
        row.event_fingerprint = actual
        return int(row.id)


def _file_digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
