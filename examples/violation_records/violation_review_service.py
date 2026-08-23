from __future__ import annotations

from datetime import datetime
from datetime import timezone

from fastapi import HTTPException
from fastapi import Request
from sqlalchemy import and_
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth import user_service
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.models import Site
from examples.auth.models import User
from examples.auth.models import Violation
from examples.auth.models import ViolationFeedback
from examples.auth.models import ViolationReviewAuditLog
from examples.violation_records.media_service import image_size_for_violation
from examples.violation_records.schemas import ViolationFeedbackCreate
from examples.violation_records.schemas import ViolationFeedbackResponse
from examples.violation_records.schemas import ViolationItem
from examples.violation_records.schemas import ViolationReviewAuditItem
from examples.violation_records.schemas import ViolationReviewStatus
from examples.violation_records.schemas import ViolationReviewUpdate
from examples.violation_records.settings import STATIC_DIR
from examples.violation_records.violation_services import (
    _feedback_detection_ids_from_json,
)
from examples.violation_records.violation_services import (
    _feedback_to_response,
)
from examples.violation_records.violation_services import (
    _load_review_audit_logs,
)
from examples.violation_records.violation_services import _load_review_scope
from examples.violation_records.violation_services import (
    _load_violation_feedbacks,
)
from examples.violation_records.violation_services import (
    _overlay_objects_from_feedback,
)
from examples.violation_records.violation_services import (
    _violation_to_detail_item,
)


async def get_next_review_violation(
    request: Request,
    db: AsyncSession,
    credentials: JwtAuthorizationCredentials,
    review_status: ViolationReviewStatus = 'pending',
    site_id: int | None = None,
    current_id: int | None = None,
) -> ViolationItem | None:
    """Return the oldest reviewable flagged violation in the user's scope."""
    _, site_names = await _load_review_scope(
        credentials.subject['username'],
        db,
    )
    if not site_names:
        return None

    conditions = [
        Violation.site.in_(site_names),
        Violation.is_flagged.is_(True),
        Violation.review_status == review_status,
    ]
    if current_id is not None:
        conditions.append(Violation.id != current_id)
    if site_id is not None:
        site_name = await db.scalar(
            select(Site.name).where(Site.id == site_id),
        )
        if site_name not in set(site_names):
            raise HTTPException(status_code=403, detail='No access to site_id')
        conditions.append(Violation.site == site_name)

    violation = await db.scalar(
        select(Violation)
        .where(and_(*conditions))
        .order_by(
            Violation.flagged_at.asc().nullslast(),
            Violation.detection_time.asc(),
            Violation.id.asc(),
        )
        .limit(1),
    )
    if violation is None:
        return None
    return await _detail_response(violation, request, db, include_audit=True)


async def get_violation_review_audit_log(
    violation_id: int,
    db: AsyncSession,
    credentials: JwtAuthorizationCredentials,
) -> list[ViolationReviewAuditItem]:
    """Return the review audit timeline for an authorised flagged record."""
    _, site_names = await _load_review_scope(
        credentials.subject['username'],
        db,
    )
    record_id = await db.scalar(
        select(Violation.id).where(
            Violation.id == violation_id,
            Violation.site.in_(site_names),
            Violation.is_flagged.is_(True),
        ),
    )
    if record_id is None:
        raise HTTPException(
            status_code=403, detail='No access to this violation',
        )
    return await _load_review_audit_logs(db, violation_id)


async def _detail_response(
    violation: Violation,
    request: Request,
    db: AsyncSession,
    include_audit: bool,
) -> ViolationItem:
    """Build a detail response after the record has been authorised."""
    feedbacks = await _load_violation_feedbacks(db, violation.id)
    feedback_note = next(
        (feedback.note for feedback in feedbacks if feedback.note),
        None,
    )
    item = _violation_to_detail_item(violation, request, feedback_note)
    item.feedbacks = feedbacks
    item.overlay_objects = _overlay_objects_from_feedback(
        item.detections,
        feedbacks,
        await image_size_for_violation(violation.image_path, STATIC_DIR),
    )
    if include_audit and violation.is_flagged:
        item.review_audit_logs = await _load_review_audit_logs(
            db,
            violation.id,
        )
    return item


async def get_single_violation(
    violation_id: int,
    request: Request,
    db: AsyncSession,
    credentials: JwtAuthorizationCredentials,
) -> ViolationItem:
    """Return full evidence and feedback for one authorised violation."""
    site_names = await user_service.get_cached_effective_site_names(
        credentials.subject['username'],
        db,
    )
    violation = await db.scalar(
        select(Violation).where(
            Violation.id == violation_id,
            Violation.site.in_(site_names),
        ),
    )
    if violation is None:
        raise HTTPException(
            status_code=403, detail='No access to this violation',
        )
    return await _detail_response(violation, request, db, include_audit=True)


async def submit_violation_feedback(
    violation_id: int,
    payload: ViolationFeedbackCreate,
    db: AsyncSession,
    credentials: JwtAuthorizationCredentials,
) -> ViolationFeedbackResponse:
    """Create pending feedback and mark the matching violation as flagged."""
    username = credentials.subject['username']
    site_names = await user_service.get_cached_effective_site_names(
        username,
        db,
    )
    violation = await db.scalar(
        select(Violation).where(
            Violation.id == violation_id,
            Violation.site.in_(site_names),
        ),
    )
    if violation is None:
        raise HTTPException(
            status_code=403, detail='No access to this violation',
        )

    detection_ids = _feedback_detection_ids_from_json(
        violation.detections_json,
    )
    if (
        detection_ids is not None
        and payload.target_detection_id
        and payload.target_detection_id not in detection_ids
    ):
        raise HTTPException(
            status_code=422,
            detail='target_detection_id does not belong to this violation',
        )
    user_id = await db.scalar(select(User.id).where(User.username == username))
    if user_id is None:
        raise HTTPException(status_code=404, detail='User not found')

    created_at = datetime.now(timezone.utc)
    violation.is_flagged = True
    violation.flag_reason = payload.type
    violation.flagged_by = user_id
    violation.flagged_at = created_at
    violation.review_status = 'pending'
    feedback = ViolationFeedback(
        violation_id=violation.id,
        user_id=user_id,
        anonymous_id=payload.anonymous_id,
        target_detection_id=payload.target_detection_id,
        feedback_type=payload.type,
        original_label=payload.original_label,
        corrected_label=payload.corrected_label,
        original_bbox=payload.original_bbox,
        corrected_bbox=payload.corrected_bbox,
        model_version=payload.model_version,
        confidence=payload.confidence,
        note=payload.note,
        status='pending',
        created_at=created_at,
    )
    db.add(feedback)
    try:
        await db.commit()
    except SQLAlchemyError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail='Failed to create violation feedback',
        ) from exc
    await db.refresh(feedback)
    return _feedback_to_response(feedback)


async def review_violation(
    violation_id: int,
    payload: ViolationReviewUpdate,
    request: Request,
    db: AsyncSession,
    credentials: JwtAuthorizationCredentials,
) -> ViolationItem:
    """Record an administrator's review decision and immutable audit event."""
    reviewer, site_names = await _load_review_scope(
        credentials.subject['username'],
        db,
    )
    violation = await db.scalar(
        select(Violation).where(
            Violation.id == violation_id,
            Violation.site.in_(site_names),
        ),
    )
    if violation is None:
        raise HTTPException(
            status_code=403, detail='No access to this violation',
        )
    if not violation.is_flagged:
        raise HTTPException(
            status_code=404, detail='Flagged violation not found',
        )

    reviewed_at = datetime.now(timezone.utc)
    old_status = violation.review_status
    violation.review_status = payload.review_status
    violation.review_note = payload.review_note
    violation.reviewed_by = reviewer.id
    violation.reviewed_at = reviewed_at
    db.add(
        ViolationReviewAuditLog(
            violation_id=violation.id,
            action='review_status_changed',
            old_status=old_status,
            new_status=payload.review_status,
            review_note=payload.review_note,
            flagged_reason=violation.flag_reason,
            reviewed_by=reviewer.id,
            reviewed_at=reviewed_at,
        ),
    )
    try:
        await db.commit()
    except SQLAlchemyError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail='Failed to update violation review',
        ) from exc
    await db.refresh(violation)
    return _violation_to_detail_item(violation, request)
