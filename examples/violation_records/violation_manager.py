from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Final

import aiofiles  # type: ignore[import-untyped]
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.models import Site
from examples.auth.models import StreamConfig
from examples.auth.models import Violation
from examples.shared.filename_utils import sanitize_filename
from examples.violation_records.schemas import ViolationDetectionRows
from examples.violation_records.schemas import ViolationWarningPayload
from examples.violation_records.settings import STATIC_DIR
from examples.violation_records.violation_types import (
    violation_type_codes_from_warnings,
)

_upload_chunk_size: Final[int] = 1024 * 1024


class EmptyViolationImageError(ValueError):
    """Raise when an uploaded violation image contains no bytes.

    This error lets the HTTP layer distinguish an empty upload from a storage
    or database failure.
    """


class ViolationImageReadError(OSError):
    """Raise when streaming an uploaded violation image fails.

    The partially written file is removed before this error leaves the manager.
    """


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

    async def save_violation(
        self,
        db: AsyncSession,
        site: str,
        stream_name: str,
        detection_time: datetime,
        image_bytes: bytes | None = None,
        warnings_json: str | None = None,
        detections_json: str | None = None,
        cone_polygon_json: str | None = None,
        pole_polygon_json: str | None = None,
        image_file: Any | None = None,
        chunk_size: int = _upload_chunk_size,
    ) -> int | None:
        """
        Save a violation record to the database and store the associated image.

        Args:
            db (AsyncSession): The SQLAlchemy session for database operations.
            site (str): The name of the associated site.
            stream_name (str): The name of the video stream or camera.
            detection_time (datetime): The timestamp of the detection.
            warnings_json (str | None, optional): A JSON string describing
                any warnings associated with this violation. Defaults to None.
            detections_json (str | None, optional): A JSON string describing
                detected items. Defaults to None.
            cone_polygon_json (str | None, optional): A JSON string describing
                safety cone polygons in the image. Defaults to None.
            pole_polygon_json (str | None, optional): A JSON string describing
                safety pole polygons in the image. Defaults to None.

        Returns:
            int | None: The ID of the newly created Violation record, or None
                if an error occurred during the process.
        """
        image_path: Path | None = None
        try:
            if warnings_json is not None:
                warnings_json = ViolationWarningPayload.model_validate_json(
                    warnings_json,
                ).model_dump_json()
            if detections_json is not None:
                detections_json = ViolationDetectionRows.model_validate_json(
                    detections_json,
                ).model_dump_json()
            detection_time = detection_time.astimezone()
            image_path = self._build_image_path(detection_time)
            if image_file is not None:
                await self._write_upload_file(
                    image_file,
                    image_path,
                    chunk_size=chunk_size,
                )
            else:
                await self._write_image_bytes(image_bytes, image_path)

            new_violation = await self._insert_violation_record(
                db=db,
                site=site,
                stream_name=stream_name,
                detection_time=detection_time,
                image_path=image_path,
                warnings_json=warnings_json,
                detections_json=detections_json,
                cone_polygon_json=cone_polygon_json,
                pole_polygon_json=pole_polygon_json,
            )

            logging.info(
                f"Violation saved successfully: ID={new_violation.id}",
            )
            return new_violation.id
        except (EmptyViolationImageError, ViolationImageReadError):
            raise

        except Exception as exc:
            if image_path is not None:
                image_path.unlink(missing_ok=True)
            try:
                await db.rollback()
            except Exception:
                pass
            logging.error(f"[ViolationManager] save_violation failed: {exc}")
            print(f"[ViolationManager] save_violation failed: {exc}")
            return None

    def _build_image_path(self, detection_time: datetime) -> Path:
        """Create a date directory and return a unique evidence image path.

        Args:
            detection_time: Detection time used to partition image storage.

        Returns:
            Writable absolute PNG path below the manager base directory.
        """
        date_folder: str = detection_time.strftime('%Y-%m-%d')
        day_dir: Path = self.base_dir / date_folder
        day_dir.mkdir(parents=True, exist_ok=True)

        filename: str = sanitize_filename(f"{uuid.uuid4()}.png")
        return day_dir / filename

    async def _write_image_bytes(
        self,
        image_bytes: bytes | None,
        image_path: Path,
    ) -> None:
        """Write an in-memory evidence image to disk.

        Args:
            image_bytes: Image bytes supplied by an internal caller.
            image_path: Destination path for the evidence image.

        Raises:
            EmptyViolationImageError: If no image bytes were supplied.
        """
        if not image_bytes:
            raise EmptyViolationImageError('Empty image file')
        async with aiofiles.open(image_path, mode='wb') as f:
            await f.write(image_bytes)

    async def _write_upload_file(
        self,
        image_file: Any,
        image_path: Path,
        chunk_size: int,
    ) -> None:
        """Stream an uploaded evidence image to disk in bounded chunks.

        Args:
            image_file: UploadFile-compatible asynchronous image source.
            image_path: Destination path for the evidence image.
            chunk_size: Maximum bytes read and written per iteration.

        Raises:
            EmptyViolationImageError: If the upload contains no bytes.
            ViolationImageReadError: If the upload cannot be read or written.
        """
        wrote_any = False
        try:
            async with aiofiles.open(image_path, mode='wb') as f:
                while True:
                    chunk = await image_file.read(chunk_size)
                    if not chunk:
                        break
                    wrote_any = True
                    await f.write(chunk)
        except Exception as exc:
            image_path.unlink(missing_ok=True)
            raise ViolationImageReadError(
                'Failed to read image file',
            ) from exc

        if not wrote_any:
            image_path.unlink(missing_ok=True)
            raise EmptyViolationImageError('Empty image file')

    async def _insert_violation_record(
        self,
        db: AsyncSession,
        site: str,
        stream_name: str,
        detection_time: datetime,
        image_path: Path,
        warnings_json: str | None,
        detections_json: str | None,
        cone_polygon_json: str | None,
        pole_polygon_json: str | None,
    ) -> Violation:
        """Insert the database row after its evidence image is persisted.

        Args:
            db: Database session used to persist the violation.
            site: Site name associated with the violation.
            stream_name: Camera name associated with the violation.
            detection_time: Time at which the violation was detected.
            image_path: Persisted absolute evidence-image path.
            warnings_json: Optional validated warning JSON.
            detections_json: Optional validated detection JSON.
            cone_polygon_json: Optional safety-cone polygon JSON.
            pole_polygon_json: Optional utility-pole polygon JSON.

        Returns:
            Persisted violation ORM record.
        """
        stream_config_id = await self._find_stream_config_id(
            db,
            site,
            stream_name,
        )
        new_violation = Violation(
            site=site,
            stream_name=stream_name,
            stream_config_id=stream_config_id,
            detection_time=detection_time,
            image_path=image_path.relative_to(self.base_dir).as_posix(),
            warnings_json=warnings_json,
            violation_type_codes=violation_type_codes_from_warnings(
                warnings_json,
            ),
            detections_json=detections_json,
            cone_polygon_json=cone_polygon_json,
            pole_polygon_json=pole_polygon_json,
        )
        db.add(new_violation)
        await db.commit()
        await db.refresh(new_violation)
        return new_violation

    async def _find_stream_config_id(
        self,
        db: AsyncSession,
        site: str,
        stream_name: str,
    ) -> int | None:
        """Resolve a stable camera identifier while retaining legacy uploads.

        Args:
            db: Database session used to find the configured camera.
            site: Site name associated with the upload.
            stream_name: Camera name associated with the upload.

        Returns:
            Stable stream configuration identifier, or ``None`` for a legacy
            upload with no configured camera.
        """
        statement = (
            select(StreamConfig.id)
            .join(Site, StreamConfig.site_id == Site.id)
            .where(
                Site.name == site,
                StreamConfig.stream_name == stream_name,
            )
        )
        return (await db.execute(statement)).scalar_one_or_none()
