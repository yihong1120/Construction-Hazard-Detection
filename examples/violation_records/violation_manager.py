from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path

import aiofiles
from sqlalchemy.ext.asyncio import AsyncSession
from werkzeug.utils import secure_filename

from examples.auth.models import Violation
from examples.violation_records.settings import STATIC_DIR


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
        image_bytes: bytes,
        warnings_json: str | None = None,
        detections_json: str | None = None,
        cone_polygon_json: str | None = None,
        pole_polygon_json: str | None = None,
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
        try:
            # 1) Create date-based folder: e.g. "2025-04-09"
            date_folder: str = detection_time.strftime('%Y-%m-%d')
            day_dir: Path = self.base_dir / date_folder
            day_dir.mkdir(parents=True, exist_ok=True)

            # 2) Generate a unique filename using UUID (suffix: .png)
            filename: str = secure_filename(f"{uuid.uuid4()}.png")
            image_path: Path = day_dir / filename

            # 3) Asynchronously write the file to disk
            async with aiofiles.open(image_path, mode='wb') as f:
                await f.write(image_bytes)

            # 4) Insert a new violation record into the database
            new_violation = Violation(
                site=site,
                stream_name=stream_name,
                detection_time=detection_time,
                image_path=str(image_path),  # Convert Path to string
                warnings_json=warnings_json,
                detections_json=detections_json,
                cone_polygon_json=cone_polygon_json,
                pole_polygon_json=pole_polygon_json,
            )
            db.add(new_violation)
            await db.commit()
            await db.refresh(new_violation)

            logging.info(
                f"Violation saved successfully: ID={new_violation.id}",
            )
            return new_violation.id

        except Exception as exc:
            logging.error(f"[ViolationManager] save_violation failed: {exc}")
            print(f"[ViolationManager] save_violation failed: {exc}")
            return None
