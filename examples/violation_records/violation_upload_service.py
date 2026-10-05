from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from fastapi import HTTPException
from fastapi import UploadFile
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth import user_service
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.models import Site
from examples.auth.models import User
from examples.db_management.services.site_access import require_site
from examples.violation_records.overlay_geometry import decode_polygons
from examples.violation_records.schemas import UploadViolationResponse
from examples.violation_records.schemas import ViolationDetectionRows
from examples.violation_records.settings import STATIC_DIR
from examples.violation_records.violation_manager import (
    EmptyViolationImageError,
)
from examples.violation_records.violation_manager import (
    ViolationImageReadError,
)
from examples.violation_records.violation_manager import ViolationManager
from examples.YOLO_server_api.model_registry import require_model

violation_manager = ViolationManager(STATIC_DIR)
logger = logging.getLogger(__name__)


async def upload_violation(
    site: str,
    stream_name: str,
    detection_time: datetime | None,
    warnings_json: str | None,
    detections_json: str | None,
    cone_polygon_json: str | None,
    pole_polygon_json: str | None,
    image: UploadFile,
    db: AsyncSession,
    credentials: JwtAuthorizationCredentials,
    model_id: str | None = None,
    model_version: str | None = None,
) -> UploadViolationResponse:
    """Store an authorised evidence image and its detector metadata."""
    username = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    site_names = await user_service.get_effective_site_names(
        username,
        db,
    )
    if site not in site_names:
        logger.info('Rejected violation upload for unauthorised site %s', site)
        raise HTTPException(status_code=403, detail='No access to this site')

    for raw in (cone_polygon_json, pole_polygon_json):
        try:
            decode_polygons(raw)
        except (ValueError, TypeError, OverflowError) as exc:
            raise HTTPException(
                422, detail='Invalid cluster polygons',
            ) from exc

    if not model_id or not model_version:
        raise HTTPException(
            422, detail='model_id and model_version are required',
        )
    user = await db.get(User, credentials.subject['user_id'])
    if user is None or str(user.tenant_id) != credentials.subject.get(
        'tenant_id',
    ):
        raise HTTPException(403, detail='Invalid account scope')
    site_id = await db.scalar(select(Site.id).where(Site.name == site))
    await require_site(db, user, site_id)
    entry = await asyncio.to_thread(
        require_model,
        model_id,
        model_version,
        str(user.tenant_id),
        user.role,
        'stream',
        site_id,
    )
    if detections_json is not None:
        try:
            rows = ViolationDetectionRows.model_validate_json(
                detections_json,
            ).root
        except ValidationError as exc:
            raise HTTPException(
                422, detail='Invalid detection rows',
            ) from exc
        class_ids = {c.id for c in entry.classes}
        if any(row[5] not in class_ids for row in rows):
            raise HTTPException(
                422, detail='Detection class is not in model version',
            )
    recorded_at = (
        detection_time.astimezone()
        if detection_time is not None
        else datetime.now().astimezone()
    )
    try:
        violation_id = await violation_manager.save_violation(
            db=db,
            site=site,
            stream_name=stream_name,
            detection_time=recorded_at,
            image_file=image,
            warnings_json=warnings_json,
            detections_json=detections_json,
            cone_polygon_json=cone_polygon_json,
            pole_polygon_json=pole_polygon_json,
            model_id=entry.id,
            model_version=entry.version,
            model_classes=[
                c.model_dump(exclude_none=True) for c in entry.classes
            ],
        )
    except (EmptyViolationImageError, ViolationImageReadError) as exc:
        raise HTTPException(
            status_code=400,
            detail='Failed to read image file',
        ) from exc

    return UploadViolationResponse(
        message='Violation uploaded successfully.',
        violation_id=violation_id,
    )
