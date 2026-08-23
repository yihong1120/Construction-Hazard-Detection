from __future__ import annotations

import base64
import binascii
import json
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from functools import lru_cache
from typing import cast as type_cast
from typing import Literal
from typing import Protocol
from urllib.parse import urlencode

from fastapi import HTTPException
from fastapi import Request
from sqlalchemy import and_
from sqlalchemy import case
from sqlalchemy import cast
from sqlalchemy import func
from sqlalchemy import Integer
from sqlalchemy import literal
from sqlalchemy import or_
from sqlalchemy import select
from sqlalchemy import String
from sqlalchemy import union_all
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from examples.auth import user_service as _user_service
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.models import Site
from examples.auth.models import StreamConfig
from examples.auth.models import User
from examples.auth.models import Violation
from examples.auth.models import ViolationFeedback
from examples.auth.models import ViolationReviewAuditLog
from examples.violation_records.schemas import FeedbackDetectionItem
from examples.violation_records.schemas import FeedbackStatus
from examples.violation_records.schemas import FeedbackType
from examples.violation_records.schemas import NormalizedBBox
from examples.violation_records.schemas import ViolationAnalyticsHourItem
from examples.violation_records.schemas import ViolationAnalyticsResponse
from examples.violation_records.schemas import ViolationAnalyticsSiteItem
from examples.violation_records.schemas import ViolationAnalyticsSummary
from examples.violation_records.schemas import ViolationAnalyticsTopSite
from examples.violation_records.schemas import ViolationAnalyticsTopType
from examples.violation_records.schemas import ViolationAnalyticsTrendItem
from examples.violation_records.schemas import ViolationAnalyticsTypeItem
from examples.violation_records.schemas import ViolationDetectionRows
from examples.violation_records.schemas import ViolationFeedbackItem
from examples.violation_records.schemas import ViolationFeedbackResponse
from examples.violation_records.schemas import ViolationItem
from examples.violation_records.schemas import ViolationListItem
from examples.violation_records.schemas import ViolationOverlayObject
from examples.violation_records.schemas import ViolationReviewAuditItem
from examples.violation_records.schemas import ViolationReviewStatus
from examples.violation_records.search_utils import SearchUtils
from examples.violation_records.violation_types import normalise_violation_type
from examples.violation_records.violation_types import (
    VIOLATION_TYPE_BY_CODE,
)
from examples.violation_records.violation_types import (
    VIOLATION_TYPE_DEFINITIONS,
)
from examples.violation_records.violation_types import (
    WARNING_PAYLOAD_ADAPTER,
)

_latest_feedback_note = (
    select(ViolationFeedback.note)
    .where(
        ViolationFeedback.violation_id == Violation.id,
        ViolationFeedback.note.is_not(None),
    )
    .order_by(
        ViolationFeedback.created_at.desc(),
        ViolationFeedback.id.desc(),
    )
    .limit(1)
    .correlate(Violation)
    .scalar_subquery()
    .label('feedback_note')
)

_violation_list_columns = (
    Violation.id.label('id'),
    Violation.site.label('site'),
    Violation.stream_name.label('stream_name'),
    Violation.detection_time.label('detection_time'),
    Violation.image_path.label('image_path'),
    Violation.warnings_json.label('warnings_json'),
    Violation.is_flagged.label('is_flagged'),
    Violation.review_status.label('review_status'),
    _latest_feedback_note,
)

ViolationListRow = tuple[
    int,
    str,
    str,
    datetime,
    str,
    str | None,
    bool,
    ViolationReviewStatus | None,
    str | None,
]


class _StreamScopeUser(Protocol):
    """Define identity fields used to authorise a selected camera stream.

    Attributes:
        role: Role determining cross-group stream access.
        group_id: Optional group scope for an administrator.
    """

    role: str
    group_id: int | None


@lru_cache(maxsize=1)
def _search_util() -> SearchUtils:
    """Load the keyword segmenter only when a search requires it.

    Returns:
        Cached CPU-backed multilingual search utility.
    """
    return SearchUtils(device=-1)


def _decode_detection_items(value: str | None) -> list[list[float]] | None:
    """Decode canonical tracked YOLO rows from a stored violation.

    Args:
        value: Optional persisted detection JSON.

    Returns:
        Validated detection rows, or ``None`` when absent.
    """
    if value is None:
        return None
    return ViolationDetectionRows.model_validate_json(value).root


def _warning_text_from_json(value: str | None) -> str | None:
    """Build a short warning summary for lightweight list rows.

    Args:
        value: Optional persisted detector-warning JSON.

    Returns:
        Bounded active-warning summary, or ``None`` when no warning is active.
    """
    if value is None:
        return None
    payload = WARNING_PAYLOAD_ADAPTER.validate_json(value)
    parts = [
        f"{key}: {warning.count}"
        for key, warning in payload.items()
        if warning.count > 0
    ]
    return ', '.join(parts)[:200] if parts else None


def _media_endpoint_url(
    endpoint_name: str,
    image_path: str,
    request: Request,
) -> str:
    """Build a protected media endpoint URL for an image path.

    Args:
        endpoint_name: Named image or thumbnail endpoint.
        image_path: Stored relative image path.
        request: Request used to create an absolute URL.

    Returns:
        Protected media URL containing the encoded image path.
    """
    query = urlencode({'image_path': image_path})
    return f"{request.url_for(endpoint_name)}?{query}"


def _feedback_detection_from_item(
    item: list[float],
    index: int,
) -> FeedbackDetectionItem:
    """Build a compact frontend-selectable detection item.

    Args:
        item: Validated tracked YOLO row.
        index: Zero-based detection position in the persisted row list.

    Returns:
        Public feedback detection item.
    """
    return FeedbackDetectionItem(
        id=f"det_{index}",
        label=f"class-{int(item[5])}",
        confidence=item[4],
        bbox=item[:4],
    )


def _feedback_detection_ids_from_json(value: str | None) -> set[str] | None:
    """Return target-detection IDs accepted for a stored record.

    Args:
        value: Optional persisted detection JSON.

    Returns:
        Accepted target identifiers, or ``None`` when detections are absent.
    """
    items = _decode_detection_items(value)
    if items is None:
        return None
    ids = {f"det_{index}" for index in range(len(items))}
    ids.update(
        str(int(item[6]))
        for item in items
        if int(item[6]) != -1
    )
    return ids


def _clamp_ratio(value: float) -> float:
    """Clamp a coordinate ratio to the inclusive 0..1 range.

    Args:
        value: Candidate ratio.

    Returns:
        Ratio constrained to the valid image-relative range.
    """
    return min(max(value, 0.0), 1.0)


def _bbox_to_normalized(
    bbox: list[float] | None,
    image_size: tuple[int, int] | None,
) -> NormalizedBBox | None:
    """Convert pixel or ratio bounds to a normalised bounding box.

    Args:
        bbox: Optional ``[x1, y1, x2, y2]`` bounds.
        image_size: Optional image width and height for pixel conversion.

    Returns:
        Image-relative bounding box, or ``None`` when invalid or incomplete.
    """
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox

    if x2 < x1 or y2 < y1:
        return None

    if all(0 <= value <= 1 for value in (x1, y1, x2, y2)):
        return NormalizedBBox(
            x=_clamp_ratio(x1),
            y=_clamp_ratio(y1),
            w=_clamp_ratio(x2 - x1),
            h=_clamp_ratio(y2 - y1),
        )

    if image_size is None:
        return None

    width, height = image_size
    if width <= 0 or height <= 0:
        return None

    return NormalizedBBox(
        x=_clamp_ratio(x1 / width),
        y=_clamp_ratio(y1 / height),
        w=_clamp_ratio((x2 - x1) / width),
        h=_clamp_ratio((y2 - y1) / height),
    )


def _bbox_nearly_equal(
    left: list[float] | None,
    right: list[float] | None,
) -> bool:
    """Compare bounding boxes with tolerance for serialised floats.

    Args:
        left: Optional first bounding box.
        right: Optional second bounding box.

    Returns:
        ``True`` when both boxes have equal co-ordinates within tolerance.
    """
    if left is None or right is None:
        return False
    return all(abs(a - b) <= 1e-6 for a, b in zip(left, right))


def _overlay_objects_from_feedback(
    detections: list[FeedbackDetectionItem] | None,
    feedbacks: list[ViolationFeedbackItem],
    image_size: tuple[int, int] | None,
) -> list[ViolationOverlayObject]:
    """Build structured overlay rows for the frontend painter.

    Args:
        detections: Optional detector selections for the violation.
        feedbacks: Feedback items used to flag or add overlays.
        image_size: Optional image dimensions for coordinate normalisation.

    Returns:
        Image-relative overlay objects for detections and false negatives.
    """
    overlay_objects: list[ViolationOverlayObject] = []
    feedback_by_detection_id: dict[str, ViolationFeedbackItem] = {}
    feedbacks_without_target: list[ViolationFeedbackItem] = []
    for feedback in feedbacks:
        if feedback.target_detection_id is None:
            feedbacks_without_target.append(feedback)
        else:
            feedback_by_detection_id.setdefault(
                feedback.target_detection_id,
                feedback,
            )

    for detection in detections or []:
        bbox = _bbox_to_normalized(detection.bbox, image_size)
        if bbox is None:
            continue
        matching_feedback = feedback_by_detection_id.get(detection.id)
        if matching_feedback is None:
            matching_feedback = next(
                (
                    candidate
                    for candidate in feedbacks_without_target
                    if _bbox_nearly_equal(
                        candidate.original_bbox,
                        detection.bbox,
                    )
                ),
                None,
            )
        overlay_objects.append(
            ViolationOverlayObject(
                object_id=detection.id,
                label=detection.label,
                confidence=detection.confidence,
                bbox=bbox,
                is_flagged=matching_feedback is not None,
                flag_reason=(
                    matching_feedback.type if matching_feedback else None
                ),
                flag_note=(
                    matching_feedback.note if matching_feedback else None
                ),
            ),
        )

    for feedback in feedbacks:
        if feedback.type != 'false_negative':
            continue
        bbox = _bbox_to_normalized(feedback.corrected_bbox, image_size)
        if bbox is None:
            continue
        overlay_objects.append(
            ViolationOverlayObject(
                object_id=f"feedback_{feedback.id}",
                label=feedback.corrected_label,
                confidence=None,
                bbox=bbox,
                is_flagged=True,
                flag_reason=feedback.type,
                flag_note=feedback.note,
            ),
        )

    return overlay_objects


def _violation_to_detail_item(
    violation: Violation,
    request: Request,
    feedback_note: str | None = None,
) -> ViolationItem:
    """Convert a violation ORM entity into a detailed response item.

    Args:
        violation: Persisted violation entity.
        request: Request used to build protected media URLs.
        feedback_note: Latest non-empty feedback note when already loaded.

    Returns:
        Public detailed violation response item.
    """
    rows = _decode_detection_items(violation.detections_json)
    detections = (
        [
            _feedback_detection_from_item(item, index)
            for index, item in enumerate(rows)
        ]
        if rows is not None
        else None
    )
    return ViolationItem(
        id=violation.id,
        site_name=violation.site,
        stream_name=violation.stream_name,
        detection_time=violation.detection_time.astimezone(),
        image_path=violation.image_path,
        image_url=_media_endpoint_url(
            'get_violation_image',
            violation.image_path,
            request,
        ),
        thumbnail_url=_media_endpoint_url(
            'get_violation_thumbnail',
            violation.image_path,
            request,
        ),
        created_at=violation.created_at.astimezone(),
        detection_items=violation.detections_json,
        warnings=violation.warnings_json,
        warning_text=_warning_text_from_json(violation.warnings_json),
        cone_polygons=violation.cone_polygon_json,
        pole_polygons=violation.pole_polygon_json,
        detections=detections,
        is_flagged=violation.is_flagged,
        flag_reason=violation.flag_reason,
        flagged_by=violation.flagged_by,
        flagged_at=violation.flagged_at,
        review_status=(
            violation.review_status if violation.is_flagged else None
        ),
        review_note=violation.review_note,
        reviewed_by=violation.reviewed_by,
        reviewed_at=violation.reviewed_at,
        feedback_note=feedback_note,
    )


def _violation_to_list_item(
    row: tuple[
        int,
        str,
        str,
        datetime,
        str,
        str | None,
        bool,
        ViolationReviewStatus | None,
        str | None,
    ],
    request: Request,
) -> ViolationListItem:
    """Convert a compact list projection into a list response item."""
    (
        violation_id,
        site,
        stream_name,
        detection_time,
        image_path,
        warnings_json,
        is_flagged,
        review_status,
        feedback_note,
    ) = row
    return ViolationListItem(
        id=violation_id,
        site_name=site,
        stream_name=stream_name,
        detection_time=detection_time.astimezone(),
        thumbnail_url=_media_endpoint_url(
            'get_violation_thumbnail',
            image_path,
            request,
        ),
        warning_text=_warning_text_from_json(warnings_json),
        is_flagged=is_flagged,
        review_status=review_status if is_flagged else None,
        feedback_note=feedback_note,
    )


def _feedback_to_item(feedback: ViolationFeedback) -> ViolationFeedbackItem:
    """Convert a feedback ORM row to a detail response item.

    Args:
        feedback: Persisted feedback ORM record.

    Returns:
        Public feedback item for a violation detail response.
    """
    return ViolationFeedbackItem(
        id=feedback.id,
        type=type_cast(FeedbackType, feedback.feedback_type),
        note=feedback.note,
        target_detection_id=feedback.target_detection_id,
        original_label=feedback.original_label,
        corrected_label=feedback.corrected_label,
        original_bbox=feedback.original_bbox,
        corrected_bbox=feedback.corrected_bbox,
        model_version=feedback.model_version,
        confidence=feedback.confidence,
        status=type_cast(FeedbackStatus, feedback.status),
        submitted_by=feedback.user_id,
        submitted_at=feedback.created_at.astimezone(),
    )


def _feedback_to_response(
    feedback: ViolationFeedback,
) -> ViolationFeedbackResponse:
    """Convert a feedback ORM row to its public response model.

    Args:
        feedback: Persisted feedback ORM record.

    Returns:
        Public response for a newly created feedback record.
    """
    return ViolationFeedbackResponse(
        id=feedback.id,
        violation_id=feedback.violation_id,
        type=type_cast(FeedbackType, feedback.feedback_type),
        target_detection_id=feedback.target_detection_id,
        original_label=feedback.original_label,
        corrected_label=feedback.corrected_label,
        original_bbox=feedback.original_bbox,
        corrected_bbox=feedback.corrected_bbox,
        model_version=feedback.model_version,
        confidence=feedback.confidence,
        note=feedback.note,
        status=type_cast(FeedbackStatus, feedback.status),
        created_at=feedback.created_at.astimezone(),
    )


def _review_audit_to_item(
    audit_log: ViolationReviewAuditLog,
) -> ViolationReviewAuditItem:
    """Convert a review audit row into the public timeline shape.

    Args:
        audit_log: Persisted immutable review audit record.

    Returns:
        Public review-history item.
    """
    return ViolationReviewAuditItem(
        id=audit_log.id,
        violation_id=audit_log.violation_id,
        actor_user_id=audit_log.reviewed_by,
        action=audit_log.action,
        old_status=type_cast(
            ViolationReviewStatus |
            None, audit_log.old_status,
        ),
        new_status=type_cast(ViolationReviewStatus, audit_log.new_status),
        note=audit_log.review_note,
        flagged_reason=audit_log.flagged_reason,
        created_at=audit_log.reviewed_at.astimezone(),
    )


async def _load_review_audit_logs(
    db: AsyncSession,
    violation_id: int,
) -> list[ViolationReviewAuditItem]:
    """Return review audit rows in newest-first timeline order.

    Args:
        db: Database session used to load audit records.
        violation_id: Identifier of the violation.

    Returns:
        Public review audit items ordered newest first.
    """
    result = await db.execute(
        select(ViolationReviewAuditLog)
        .where(ViolationReviewAuditLog.violation_id == violation_id)
        .order_by(
            ViolationReviewAuditLog.reviewed_at.desc(),
            ViolationReviewAuditLog.id.desc(),
        ),
    )
    return [
        _review_audit_to_item(audit_log)
        for audit_log in result.scalars().all()
    ]


async def _load_violation_feedbacks(
    db: AsyncSession,
    violation_id: int,
) -> list[ViolationFeedbackItem]:
    """Return feedback rows for a violation in newest-first order.

    Args:
        db: Database session used to load feedback.
        violation_id: Identifier of the violation.

    Returns:
        Public feedback items ordered newest first.
    """
    result = await db.execute(
        select(ViolationFeedback)
        .where(ViolationFeedback.violation_id == violation_id)
        .order_by(
            ViolationFeedback.created_at.desc(),
            ViolationFeedback.id.desc(),
        ),
    )
    return [_feedback_to_item(feedback) for feedback in result.scalars().all()]


def _encode_violation_cursor(item: ViolationListItem) -> str:
    """Encode a response item's order key for cursor pagination.

    Args:
        item: Last violation item returned in a page.

    Returns:
        URL-safe opaque keyset-pagination cursor.
    """
    payload = json.dumps(
        {
            'detection_time': item.detection_time.isoformat(),
            'id': item.id,
        },
        separators=(',', ':'),
    ).encode('utf-8')
    return base64.urlsafe_b64encode(payload).decode('ascii').rstrip('=')


def _decode_violation_cursor(cursor: str) -> tuple[datetime, int]:
    """Decode a cursor into the list ordering key.

    Args:
        cursor: URL-safe opaque keyset-pagination cursor.

    Returns:
        Detection time and violation identifier ordering key.

    Raises:
        HTTPException: If the cursor cannot be decoded or validated.
    """
    try:
        padded = cursor + '=' * (-len(cursor) % 4)
        raw_payload = base64.urlsafe_b64decode(padded.encode('ascii'))
        payload = json.loads(raw_payload.decode('utf-8'))
        detection_time = datetime.fromisoformat(payload['detection_time'])
        violation_id = int(payload['id'])
    except (
        binascii.Error,
        UnicodeDecodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise HTTPException(status_code=422, detail='Invalid cursor') from exc
    return detection_time, violation_id


async def _resolve_stream_filter(
    stream_id: str,
    site_name: str | None,
    site_names: list[str],
    user: _StreamScopeUser | User,
    db: AsyncSession,
) -> tuple[int, str]:
    """Resolve and authorise a stable camera before querying violations.

    Args:
        stream_id: Client-supplied stable stream identifier.
        site_name: Optional selected site name.
        site_names: Sites visible to the requesting user.
        user: User scope used for group access checks.
        db: Database session used to resolve the stream.

    Returns:
        Authorised stream configuration identifier and its site name.

    Raises:
        HTTPException: If the stream identifier or access scope is invalid.
    """
    stream_id_text = stream_id.strip()
    if not stream_id_text.isdigit() or int(stream_id_text) <= 0:
        raise HTTPException(
            status_code=422,
            detail='stream_id must be a positive stream configuration ID',
        )

    stream_result = await db.execute(
        select(
            StreamConfig.id,
            StreamConfig.group_id,
            Site.name,
        )
        .join(Site, StreamConfig.site_id == Site.id)
        .where(StreamConfig.id == int(stream_id_text)),
    )
    stream_row = stream_result.first()
    if (
        not stream_row
        or stream_row[2] not in site_names
        or (site_name is not None and stream_row[2] != site_name)
        or (
            user.role != 'super_admin'
            and (user.group_id is None or stream_row[1] != user.group_id)
        )
    ):
        raise HTTPException(status_code=403, detail='No access to stream_id')

    return int(stream_row[0]), str(stream_row[2])


async def _load_review_scope(
    username: str,
    db: AsyncSession,
) -> tuple[User, list[str]]:
    """Load a reviewer and the sites they may review.

    Args:
        username: Authenticated username.
        db: Database session used to load effective site access.

    Returns:
        Reviewer user and accessible site names.

    Raises:
        HTTPException: If the user is not an administrator.
    """
    user, sites = await _user_service.load_user_with_effective_sites(
        username,
        db,
    )
    if user.role not in {'admin', 'super_admin'}:
        raise HTTPException(
            status_code=403,
            detail='Only admin or super_admin can review violations',
        )
    return user, [site.name for site in sites]


async def _violation_site_names(
    username: str,
    flagged: bool | None,
    review_status: ViolationReviewStatus | None,
    db: AsyncSession,
) -> list[str]:
    """Load sites visible to a viewer or authorised reviewer.

    Args:
        username: Authenticated username.
        flagged: Optional flagged-record filter.
        review_status: Optional review-status filter.
        db: Database session used to load effective access.

    Returns:
        Site names visible for the selected query mode.
    """
    if flagged is True or review_status is not None:
        _, site_names = await _load_review_scope(username, db)
        return site_names
    return await _user_service.get_cached_effective_site_names(username, db)


async def _build_violation_conditions(
    username: str,
    site_names: list[str],
    site_id: int | None,
    stream_id: str | None,
    violation_type: str | None,
    keyword: str | None,
    start_time: datetime | None,
    end_time: datetime | None,
    flagged: bool | None,
    review_status: ViolationReviewStatus | None,
    cursor: str | None,
    db: AsyncSession,
) -> list:
    """Build the fully authorised filter set for a violation listing.

    Args:
        username: Authenticated username.
        site_names: Sites visible to the user.
        site_id: Optional site filter.
        stream_id: Optional stream filter.
        violation_type: Optional type filter.
        keyword: Optional keyword filter.
        start_time: Optional range start.
        end_time: Optional range end.
        flagged: Optional flag-state filter.
        review_status: Optional review-status filter.
        cursor: Optional pagination cursor.
        db: Database session used to resolve scoped filters.

    Returns:
        SQL predicates constrained by the user's authorisation scope.
    """
    conditions: list = [Violation.site.in_(site_names)]
    if flagged is True or review_status is not None:
        conditions.append(Violation.is_flagged.is_(True))
    if review_status is not None:
        conditions.append(Violation.review_status == review_status)
    site_name = await _filtered_violation_site_name(site_id, site_names, db)
    if site_name is not None:
        conditions.append(Violation.site == site_name)
    conditions.extend(
        await _stream_violation_conditions(
            username,
            stream_id,
            site_name,
            site_names,
            db,
        ),
    )
    conditions.extend(
        _optional_violation_conditions(
            violation_type,
            keyword,
            start_time,
            end_time,
            cursor,
        ),
    )
    return conditions


async def _filtered_violation_site_name(
    site_id: int | None,
    site_names: list[str],
    db: AsyncSession,
) -> str | None:
    """Resolve a site identifier and enforce caller scope.

    Args:
        site_id: Optional selected site identifier.
        site_names: Site names visible to the caller.
        db: Database session used to resolve the site.

    Returns:
        Authorised site name, or ``None`` when no site was selected.

    Raises:
        HTTPException: If the selected site is not accessible.
    """
    if site_id is None:
        return None
    site_stmt = select(Site.name).where(Site.id == site_id)
    site_name = (await db.execute(site_stmt)).scalar_one_or_none()
    if not site_name or site_name not in set(site_names):
        raise HTTPException(status_code=403, detail='No access to site_id')
    return site_name


async def _stream_violation_conditions(
    username: str,
    stream_id: str | None,
    site_name: str | None,
    site_names: list[str],
    db: AsyncSession,
) -> list:
    """Resolve a requested camera and return scoped SQL conditions.

    Args:
        username: Authenticated username.
        stream_id: Optional selected stable stream identifier.
        site_name: Optional selected site name.
        site_names: Sites visible to the caller.
        db: Database session used to resolve user and stream scope.

    Returns:
        Stream and site SQL predicates, or an empty list when unfiltered.
    """
    if not stream_id:
        return []
    user, _ = await _user_service.load_user_with_effective_sites(
        username,
        db,
        status_code=401,
        detail='Invalid user',
    )
    stream_config_id, stream_site_name = await _resolve_stream_filter(
        stream_id,
        site_name,
        site_names,
        user,
        db,
    )
    return [
        Violation.stream_config_id == stream_config_id,
        Violation.site == stream_site_name,
    ]


def _optional_violation_conditions(
    violation_type: str | None,
    keyword: str | None,
    start_time: datetime | None,
    end_time: datetime | None,
    cursor: str | None,
) -> list[ColumnElement[bool]]:
    """Return independent type, text, time, and cursor conditions.

    Args:
        violation_type: Optional type filter.
        keyword: Optional keyword filter.
        start_time: Optional range start.
        end_time: Optional range end.
        cursor: Optional pagination cursor.
    Returns:
        SQL predicates independent of authorisation scope.
    """
    conditions: list[ColumnElement[bool]] = []
    if violation_type:
        canonical_type = normalise_violation_type(violation_type)
        if canonical_type is None:
            valid = ', '.join(
                definition.code for definition in VIOLATION_TYPE_DEFINITIONS
            )
            raise HTTPException(
                status_code=422,
                detail=(
                    'Unsupported violation_type. Expected one of: '
                    f'{valid}'
                ),
            )
        conditions.append(
            cast(Violation.violation_type_codes, JSONB).contains(
                [canonical_type],
            ),
        )
    keyword_condition = _violation_keyword_condition(keyword)
    if keyword_condition is not None:
        conditions.append(keyword_condition)
    if start_time:
        conditions.append(Violation.detection_time >= start_time)
    if end_time:
        conditions.append(Violation.detection_time <= end_time)
    if cursor:
        conditions.append(_violation_cursor_condition(cursor))
    return conditions


def _violation_keyword_condition(
    keyword: str | None,
) -> ColumnElement[bool] | None:
    """Expand keyword synonyms into an OR search predicate.

    Args:
        keyword: Optional raw search keyword.

    Returns:
        SQL text-match predicate, or ``None`` for blank input.
    """
    keyword_text = keyword.strip() if keyword else ''
    if not keyword_text:
        return None
    synonyms: list[str] = _search_util().expand_synonyms(keyword_text)
    matches = [
        condition
        for synonym in dict.fromkeys(synonyms)
        for condition in (
            Violation.stream_name.ilike(f"%{synonym}%"),
            Violation.warnings_json.ilike(f"%{synonym}%"),
        )
    ]
    return or_(*matches) if matches else None


def _violation_cursor_condition(cursor: str) -> ColumnElement[bool]:
    """Build a keyset-pagination condition for a decoded cursor.

    Args:
        cursor: Opaque cursor from the previous response page.

    Returns:
        SQL predicate for records after the cursor's ordering key.
    """
    cursor_time, cursor_id = _decode_violation_cursor(cursor)
    return or_(
        Violation.detection_time < cursor_time,
        and_(
            Violation.detection_time == cursor_time,
            Violation.id < cursor_id,
        ),
    )


async def get_violation_analytics(
    start: datetime,
    end: datetime,
    db: AsyncSession,
    credentials: JwtAuthorizationCredentials,
    *,
    site_id: int | None = None,
    stream_id: str | None = None,
    violation_type: str | None = None,
    bucket: Literal['day', 'hour', 'week'] = 'day',
) -> ViolationAnalyticsResponse:
    """Return aggregated violation counts for charts and KPI widgets.

    The response intentionally excludes image paths, warning payloads, and
    individual violation records.
    """
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    user, sites = await _user_service.load_user_with_effective_sites(
        username,
        db,
        status_code=401,
        detail='Invalid user',
    )
    if user.role not in {'admin', 'super_admin'}:
        raise HTTPException(
            status_code=403,
            detail='violation_analytics_forbidden',
        )
    site_names = [site.name for site in sites]
    start_utc = (
        start.replace(tzinfo=timezone.utc)
        if start.tzinfo is None
        else start.astimezone(timezone.utc)
    )
    end_utc = (
        end.replace(tzinfo=timezone.utc)
        if end.tzinfo is None
        else end.astimezone(timezone.utc)
    )
    if start_utc >= end_utc:
        raise HTTPException(
            status_code=422,
            detail='start must be before end',
        )
    try:
        latest_end = start_utc.replace(year=start_utc.year + 5)
    except ValueError:
        latest_end = start_utc.replace(
            year=start_utc.year + 5,
            day=28,
        )
    if end_utc > latest_end:
        raise HTTPException(
            status_code=422,
            detail='Query range must not exceed 5 years',
        )
    if not site_names:
        return ViolationAnalyticsResponse(
            summary=ViolationAnalyticsSummary(total=0, today=0),
        )

    conditions: list[ColumnElement[bool]] = [
        Violation.site.in_(site_names),
        Violation.detection_time >= start_utc,
        Violation.detection_time <= end_utc,
    ]

    if site_id is not None:
        site_name_set = set(site_names)
        site_stmt = select(Site.name).where(Site.id == site_id)
        site_name = (await db.execute(site_stmt)).scalar_one_or_none()
        if not site_name or site_name not in site_name_set:
            raise HTTPException(status_code=403, detail='No access to site_id')
        conditions.append(Violation.site == site_name)

    if stream_id:
        stream_config_id, stream_site_name = await _resolve_stream_filter(
            stream_id,
            site_name if site_id is not None else None,
            site_names,
            user,
            db,
        )
        conditions.extend(
            [
                Violation.stream_config_id == stream_config_id,
                Violation.site == stream_site_name,
            ],
        )

    canonical_type = None
    if violation_type:
        canonical_type = normalise_violation_type(violation_type)
        if canonical_type is None:
            valid = ', '.join(
                definition.code for definition in VIOLATION_TYPE_DEFINITIONS
            )
            raise HTTPException(
                status_code=422,
                detail=(
                    'Unsupported violation_type. Expected one of: '
                    f'{valid}'
                ),
            )
        conditions.append(
            cast(Violation.violation_type_codes, JSONB).contains(
                [canonical_type],
            ),
        )

    where_clause = and_(*conditions)
    now_utc = datetime.now(timezone.utc)
    today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(days=1)
    type_names = (
        [canonical_type]
        if canonical_type is not None
        else [definition.code for definition in VIOLATION_TYPE_DEFINITIONS]
    )

    # One materialised CTE keeps dashboard aggregation to one database trip.
    filtered = (
        select(
            Violation.site.label('site'),
            Violation.detection_time.label('detection_time'),
            Violation.violation_type_codes.label('violation_type_codes'),
        )
        .where(where_clause)
        .cte('filtered_violations')
        .prefix_with('MATERIALIZED')
    )

    empty_text = cast(literal(None), String)
    zero = literal(0)
    bucket_format = {
        'hour': 'YYYY-MM-DD"T"HH24:00:00"Z"',
        'day': 'YYYY-MM-DD',
        'week': 'IYYY-"W"IW',
    }[bucket]
    bucket_expr = func.to_char(filtered.c.detection_time, bucket_format)
    hour_expr = cast(func.extract('hour', filtered.c.detection_time), Integer)
    aggregate_queries = [
        select(
            literal('summary').label('kind'),
            empty_text.label('value'),
            empty_text.label('label'),
            func.count().label('count'),
            cast(
                func.coalesce(
                    func.sum(
                        case(
                            (
                                and_(
                                    filtered.c.detection_time >= today_start,
                                    filtered.c.detection_time < today_end,
                                ),
                                1,
                            ),
                            else_=0,
                        ),
                    ),
                    0,
                ),
                Integer,
            ).label('today'),
        ).select_from(filtered),
        select(
            literal('trend').label('kind'),
            cast(bucket_expr, String).label('value'),
            empty_text.label('label'),
            func.count().label('count'),
            zero.label('today'),
        )
        .select_from(filtered)
        .group_by(bucket_expr),
        select(
            literal('site').label('kind'),
            cast(Site.id, String).label('value'),
            Site.name.label('label'),
            func.count().label('count'),
            zero.label('today'),
        )
        .select_from(filtered.join(Site, filtered.c.site == Site.name))
        .group_by(Site.id, Site.name),
        select(
            literal('hour').label('kind'),
            cast(hour_expr, String).label('value'),
            empty_text.label('label'),
            func.count().label('count'),
            zero.label('today'),
        )
        .select_from(filtered)
        .group_by(hour_expr),
    ]
    aggregate_queries.extend(
        select(
            literal('type').label('kind'),
            literal(type_name).label('value'),
            literal(VIOLATION_TYPE_BY_CODE[type_name].label).label('label'),
            cast(
                func.coalesce(
                    func.sum(
                        case(
                            (
                                cast(
                                    filtered.c.violation_type_codes,
                                    JSONB,
                                ).contains(
                                    [type_name],
                                ),
                                1,
                            ),
                            else_=0,
                        ),
                    ),
                    0,
                ),
                Integer,
            ).label('count'),
            zero.label('today'),
        ).select_from(filtered)
        for type_name in type_names
    )

    aggregate_rows = (await db.execute(union_all(*aggregate_queries))).all()
    total = 0
    today = 0
    trend: list[ViolationAnalyticsTrendItem] = []
    by_site: list[ViolationAnalyticsSiteItem] = []
    by_hour: list[ViolationAnalyticsHourItem] = []
    by_type: list[ViolationAnalyticsTypeItem] = []
    for kind, value, label, count, row_today in aggregate_rows:
        count_value = int(count or 0)
        if kind == 'summary':
            total = count_value
            today = int(row_today or 0)
        elif kind == 'trend':
            trend.append(
                ViolationAnalyticsTrendItem(
                    bucket=str(value),
                    count=count_value,
                ),
            )
        elif kind == 'site':
            by_site.append(
                ViolationAnalyticsSiteItem(
                    site_id=int(value),
                    site_name=str(label),
                    count=count_value,
                ),
            )
        elif kind == 'hour':
            by_hour.append(
                ViolationAnalyticsHourItem(
                    hour=int(value),
                    count=count_value,
                ),
            )
        elif kind == 'type' and count_value:
            by_type.append(
                ViolationAnalyticsTypeItem(
                    type=str(value),
                    label=str(label),
                    count=count_value,
                ),
            )

    if total == 0:
        return ViolationAnalyticsResponse(
            summary=ViolationAnalyticsSummary(total=0, today=0),
        )

    trend.sort(key=lambda item: item.bucket)
    by_site.sort(key=lambda item: (-item.count, item.site_id))
    by_hour.sort(key=lambda item: item.hour)
    by_type.sort(key=lambda item: (-item.count, item.type))
    top_site = (
        ViolationAnalyticsTopSite(**by_site[0].model_dump())
        if by_site
        else None
    )
    top_type = (
        ViolationAnalyticsTopType(**by_type[0].model_dump())
        if by_type
        else None
    )

    return ViolationAnalyticsResponse(
        summary=ViolationAnalyticsSummary(
            total=total,
            today=today,
            top_site=top_site,
            top_type=top_type,
        ),
        trend=trend,
        by_type=by_type,
        by_site=by_site,
        by_hour=by_hour,
    )
