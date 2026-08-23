from __future__ import annotations

import logging
from datetime import datetime

from fastapi import HTTPException
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth import user_service
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.violation_records.schemas import UploadViolationResponse
from examples.violation_records.settings import STATIC_DIR
from examples.violation_records.violation_manager import (
    EmptyViolationImageError,
)
from examples.violation_records.violation_manager import (
    ViolationImageReadError,
)
from examples.violation_records.violation_manager import ViolationManager

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
) -> UploadViolationResponse:
    """Store an authorised evidence image and its detector metadata."""
    username = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    site_names = await user_service.get_cached_effective_site_names(
        username,
        db,
    )
    if site not in site_names:
        logger.info('Rejected violation upload for unauthorised site %s', site)
        raise HTTPException(status_code=403, detail='No access to this site')

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
