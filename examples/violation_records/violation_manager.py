from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Final

import aiofiles  # type: ignore[import-untyped]
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.models import Violation
from examples.shared.filename_utils import sanitize_filename
from examples.violation_records.settings import STATIC_DIR

_upload_chunk_size: Final[int] = 1024 * 1024


class EmptyViolationImageError(ValueError):
    """Raised when an uploaded violation image has no bytes."""


class ViolationImageReadError(OSError):
    """Raised when streaming an uploaded violation image fails."""


class ViolationManager:
    """
    A manager class responsible for storing violation records in both the local
    file system (for images) and the database via SQLAlchemy ORM.
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        """
        Initialise the manager with a base directory for storing images.

        This creates the specified base directory if it does not already exist.

        Args:
            base_dir (str | Path | None, optional): The base directory path
                where images will be stored. If None, defaults to STATIC_DIR
                from settings. Defaults to None.
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
        """Create the day directory and return a unique image path."""
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
        """Write an in-memory image payload to disk."""
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
        """Stream an UploadFile-like object to disk in bounded chunks."""
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
        """Insert the database row after the image has been persisted."""
        new_violation = Violation(
            site=site,
            stream_name=stream_name,
            detection_time=detection_time,
            image_path=str(image_path),
            warnings_json=warnings_json,
            detections_json=detections_json,
            cone_polygon_json=cone_polygon_json,
            pole_polygon_json=pole_polygon_json,
        )
        db.add(new_violation)
        await db.commit()
        await db.refresh(new_violation)
        return new_violation
