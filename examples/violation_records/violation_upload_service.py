from __future__ import annotations

import logging

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth import user_service
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.models import Site
from examples.auth.models import User
from examples.db_management.services.site_access import require_site
from examples.shared.evidence_metadata import validate_metadata
from examples.shared.image_validation import ImageValidationError
from examples.violation_records.evidence import EvidenceEventConflictError
from examples.violation_records.evidence import ViolationEvidence
from examples.violation_records.schemas import UploadViolationResponse
from examples.violation_records.settings import STATIC_DIR
from examples.violation_records.violation_manager import (
    EmptyViolationImageError,
)
from examples.violation_records.violation_manager import (
    ViolationImageReadError,
)
from examples.violation_records.violation_manager import ViolationManager
from examples.YOLO_server_api.model_registry import resolve_evidence_version
from examples.YOLO_server_api.model_registry import select_model_async

violation_manager = ViolationManager(STATIC_DIR)
logger = logging.getLogger(__name__)


async def upload_violation(
    evidence: ViolationEvidence,
    db: AsyncSession,
    credentials: JwtAuthorizationCredentials,
) -> UploadViolationResponse:
    """Store an authorised evidence image and its detector metadata."""
    site_id = evidence.site_id
    username = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')
    if site_id is None:
        site_names = await user_service.get_effective_site_names(username, db)
        if evidence.site not in site_names:
            logger.info(
                'Rejected violation upload for unauthorised site %s',
                evidence.site,
            )
            raise HTTPException(
                status_code=403,
                detail='No access to this site',
            )
    try:
        parsed = validate_metadata(evidence)
    except (ValidationError, ValueError, TypeError, OverflowError) as error:
        raise HTTPException(422, detail='Invalid evidence metadata') from error
    evidence = parsed.source
    if not evidence.model_id or not evidence.model_version:
        raise HTTPException(
            422,
            detail='model_id and model_version are required',
        )
    user = await db.get(User, credentials.subject['user_id'])
    if user is None or str(user.tenant_id) != credentials.subject.get(
        'tenant_id',
    ):
        raise HTTPException(403, detail='Invalid account scope')
    if site_id is None:
        site_id = await db.scalar(
            select(Site.id).where(Site.name == evidence.site),
        )
    await require_site(db, user, site_id)
    current = await select_model_async(
        evidence.model_id,
        None,
        str(user.tenant_id),
        user.role,
        'stream',
        site_id,
    )
    entry = await resolve_evidence_version(current, evidence.model_version)
    if parsed.detections is not None:
        rows = parsed.detections
        class_ids = {c.id for c in entry.classes}
        if any(row[5] not in class_ids for row in rows):
            raise HTTPException(
                422,
                detail='Detection class is not in model version',
            )
    try:
        violation_id = await violation_manager.save_violation(
            db,
            parsed.with_source(
                site_id=site_id,
                model_id=entry.id,
                model_version=entry.version,
                model_classes=[
                    c.model_dump(exclude_none=True) for c in entry.classes
                ],
            ),
        )
    except ImageValidationError as exc:
        raise HTTPException(
            exc.status_code, detail={'code': exc.code},
        ) from exc
    except EvidenceEventConflictError as exc:
        raise HTTPException(409, 'event_id_conflict') from exc
    except (EmptyViolationImageError, ViolationImageReadError) as exc:
        raise HTTPException(
            status_code=400,
            detail='Failed to read image file',
        ) from exc
    return UploadViolationResponse(
        message='Violation uploaded successfully.',
        violation_id=violation_id,
    )
