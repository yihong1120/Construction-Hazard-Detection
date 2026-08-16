from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from typing import Final
from typing import Literal
from typing import Protocol
from urllib.parse import urlencode

from fastapi import HTTPException
from fastapi import Request
from fastapi import UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from PIL import ImageFile
from PIL import UnidentifiedImageError
from sqlalchemy import and_
from sqlalchemy import cast
from sqlalchemy import func
from sqlalchemy import Integer
from sqlalchemy import or_
from sqlalchemy import select
from sqlalchemy import String
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
from examples.shared.filename_utils import sanitize_filename
from examples.violation_records.path_utils import _determine_media_type
from examples.violation_records.path_utils import _normalize_safe_rel_path
from examples.violation_records.path_utils import _resolve_and_authorize
from examples.violation_records.schemas import FeedbackDetectionItem
from examples.violation_records.schemas import NormalizedBBox
from examples.violation_records.schemas import SiteOut
from examples.violation_records.schemas import UploadViolationResponse
from examples.violation_records.schemas import ViolationAnalyticsHourItem
from examples.violation_records.schemas import ViolationAnalyticsResponse
from examples.violation_records.schemas import ViolationAnalyticsSiteItem
from examples.violation_records.schemas import ViolationAnalyticsSummary
from examples.violation_records.schemas import ViolationAnalyticsTopSite
from examples.violation_records.schemas import ViolationAnalyticsTopType
from examples.violation_records.schemas import ViolationAnalyticsTrendItem
from examples.violation_records.schemas import ViolationAnalyticsTypeItem
from examples.violation_records.schemas import ViolationDetectionRows
from examples.violation_records.schemas import ViolationFeedbackCreate
from examples.violation_records.schemas import ViolationFeedbackItem
from examples.violation_records.schemas import ViolationFeedbackResponse
from examples.violation_records.schemas import ViolationFilterCamera
from examples.violation_records.schemas import ViolationFilterOptions
from examples.violation_records.schemas import ViolationItem
from examples.violation_records.schemas import ViolationList
from examples.violation_records.schemas import ViolationOverlayObject
from examples.violation_records.schemas import ViolationReviewAuditItem
from examples.violation_records.schemas import ViolationReviewStatus
from examples.violation_records.schemas import ViolationReviewUpdate
from examples.violation_records.schemas import ViolationTypeOption
from examples.violation_records.search_utils import SearchUtils
from examples.violation_records.settings import STATIC_DIR
from examples.violation_records.violation_manager import (
    EmptyViolationImageError,
)
from examples.violation_records.violation_manager import (
    ViolationImageReadError,
)
from examples.violation_records.violation_manager import ViolationManager
from examples.violation_records.violation_types import normalise_violation_type
from examples.violation_records.violation_types import parse_warning_payload
from examples.violation_records.violation_types import (
    VIOLATION_TYPE_BY_CODE,
)
from examples.violation_records.violation_types import (
    VIOLATION_TYPE_DEFINITIONS,
)

# Module-level aliases used by tests for patching
get_cached_effective_site_names = _user_service.get_cached_effective_site_names
load_user_with_effective_sites = _user_service.load_user_with_effective_sites
get_user_effective_sites = _user_service.load_user_with_effective_sites
get_user_sites_cached = _user_service.get_cached_effective_site_names

# Instantiate a global ViolationManager for handling image saving
# and record creation.
violation_manager: ViolationManager = ViolationManager(base_dir=STATIC_DIR)

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

_violation_columns = (
    Violation.id,
    Violation.site,
    Violation.stream_name,
    Violation.detection_time,
    Violation.image_path,
    Violation.created_at,
    Violation.detections_json,
    Violation.warnings_json,
    Violation.cone_polygon_json,
    Violation.pole_polygon_json,
    Violation.is_flagged,
    Violation.flag_reason,
    Violation.flagged_by,
    Violation.flagged_at,
    Violation.review_status,
    Violation.review_note,
    Violation.reviewed_by,
    Violation.reviewed_at,
    _latest_feedback_note,
)
_violation_column_count = len(_violation_columns)

AnalyticsBucket = Literal['day', 'hour', 'week']


class _StreamScopeUser(Protocol):
    """Define identity fields used to authorise a selected camera stream.

    Attributes:
        role: Role determining cross-group stream access.
        group_id: Optional group scope for an administrator.
    """

    role: str
    group_id: int | None


MAX_ANALYTICS_RANGE_YEARS = 5
ALLOWED_VIOLATION_ANALYTICS_ROLES: Final[frozenset[str]] = frozenset(
    {
        'admin',
        'super_admin',
    },
)
THUMBNAIL_DIR_NAME = '_thumbnails'
THUMBNAIL_MAX_EDGE = 360
THUMBNAIL_QUALITY = 78
THUMBNAIL_HEADER_SCAN_BYTES = 64 * 1024
_ISOBMFF_IMAGE_BRANDS: Final[frozenset[bytes]] = frozenset(
    {
        b'avif',
        b'avis',
        b'heic',
        b'heix',
        b'hevc',
        b'hevx',
        b'mif1',
        b'msf1',
    },
)


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
    payload = parse_warning_payload(value)
    if payload is None:
        return None
    parts = [
        f'{key}: {warning.count}'
        for key, warning in payload.root.items()
        if warning.count > 0
    ]
    return ', '.join(parts)[:200] if parts else None


def _media_endpoint_url(
    endpoint_name: str,
    image_path: str,
    request: Request | None,
) -> str:
    """Build a protected media endpoint URL for an image path.

    Args:
        endpoint_name: Named image or thumbnail endpoint.
        image_path: Stored relative image path.
        request: Optional request used to create an absolute URL.

    Returns:
        Protected media URL containing the encoded image path.
    """
    query = urlencode({'image_path': image_path})
    if request is None:
        return f"/{endpoint_name}?{query}"
    return f"{request.url_for(endpoint_name)}?{query}"


def _image_urls(
    image_path: str,
    request: Request | None,
) -> tuple[str, str]:
    """Return protected original-image and thumbnail URLs.

    Args:
        image_path: Stored relative image path.
        request: Optional request used to create absolute URLs.

    Returns:
        Original-image and thumbnail endpoint URLs.
    """
    return (
        _media_endpoint_url('get_violation_image', image_path, request),
        _media_endpoint_url('get_violation_thumbnail', image_path, request),
    )


def _bbox_from_detection_item(item: list[float]) -> list[float]:
    """Extract bounding-box columns from one canonical YOLO row.

    Args:
        item: Validated tracked YOLO row.

    Returns:
        ``[x1, y1, x2, y2]`` bounding-box values.
    """
    return item[:4]


def _feedback_detection_id_candidates(
    item: list[float],
    index: int,
) -> set[str]:
    """Return IDs accepted by feedback endpoints for one detection.

    Args:
        item: Validated tracked YOLO row.
        index: Zero-based detection position in the persisted row list.

    Returns:
        Synthetic and, when present, tracker detection identifiers.
    """
    candidates = {f"det_{index}"}
    track_id = int(item[6])
    if track_id != -1:
        candidates.add(str(track_id))
    return candidates


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
        id=f'det_{index}',
        label=f'class-{int(item[5])}',
        confidence=item[4],
        bbox=_bbox_from_detection_item(item),
    )


def _feedback_detections_from_json(
    value: str | None,
) -> list[FeedbackDetectionItem] | None:
    """Return normalised detection selections for violation responses.

    Args:
        value: Optional persisted detection JSON.

    Returns:
        Public detection items, or ``None`` when detections are absent.
    """
    items = _decode_detection_items(value)
    if items is None:
        return None
    return [
        _feedback_detection_from_item(item, index)
        for index, item in enumerate(items)
    ]


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
    ids: set[str] = set()
    for index, item in enumerate(items):
        ids.update(_feedback_detection_id_candidates(item, index))
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


def _feedback_for_detection(
    detection: FeedbackDetectionItem,
    feedbacks: list[ViolationFeedbackItem],
) -> ViolationFeedbackItem | None:
    """Return feedback targeting a detection, when available.

    Args:
        detection: Public detection selected by the frontend.
        feedbacks: Newest-first feedback items for the violation.

    Returns:
        Matching newest feedback, or ``None`` when none targets the detection.
    """
    for feedback in feedbacks:
        if feedback.target_detection_id == detection.id:
            return feedback
        if _bbox_nearly_equal(feedback.original_bbox, detection.bbox):
            return feedback
    return None


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

    for detection in detections or []:
        bbox = _bbox_to_normalized(detection.bbox, image_size)
        if bbox is None:
            continue
        feedback = _feedback_for_detection(detection, feedbacks)
        overlay_objects.append(
            ViolationOverlayObject(
                object_id=detection.id,
                label=detection.label,
                confidence=detection.confidence,
                bbox=bbox,
                is_flagged=feedback is not None,
                flag_reason=feedback.type if feedback else None,
                flag_note=feedback.note if feedback else None,
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


def _violation_to_item(
    row: Any,
    request: Request | None = None,
) -> ViolationItem:
    """Convert an ORM object or selected row into a response item.

    Args:
        row: Violation ORM object or selected-column result row.
        request: Optional request used to build protected media URLs.

    Returns:
        Public detailed violation response item.
    """
    if hasattr(row, 'site'):
        detections = _feedback_detections_from_json(row.detections_json)
        is_flagged = bool(getattr(row, 'is_flagged', False))
        image_url, thumbnail_url = _image_urls(row.image_path, request)
        return ViolationItem(
            id=row.id,
            site_name=row.site,
            stream_name=row.stream_name,
            detection_time=row.detection_time.astimezone(),
            detected_at=row.detection_time.astimezone(),
            image_path=row.image_path,
            image_url=image_url,
            thumbnail_url=thumbnail_url,
            created_at=row.created_at.astimezone(),
            detection_items=row.detections_json,
            warnings=row.warnings_json,
            warning_text=_warning_text_from_json(row.warnings_json),
            cone_polygons=row.cone_polygon_json,
            pole_polygons=row.pole_polygon_json,
            detections=detections,
            feedback_detections=detections,
            is_flagged=is_flagged,
            flag_reason=getattr(row, 'flag_reason', None),
            flagged_by=getattr(row, 'flagged_by', None),
            flagged_at=getattr(row, 'flagged_at', None),
            review_status=(
                getattr(row, 'review_status', None) if is_flagged else None
            ),
            review_note=getattr(row, 'review_note', None),
            reviewed_by=getattr(row, 'reviewed_by', None),
            reviewed_at=getattr(row, 'reviewed_at', None),
            feedback_note=getattr(row, 'feedback_note', None),
        )

    (
        violation_id,
        site,
        stream_name,
        detection_time,
        image_path,
        created_at,
        detections_json,
        warnings_json,
        cone_polygon_json,
        pole_polygon_json,
        is_flagged,
        flag_reason,
        flagged_by,
        flagged_at,
        review_status,
        review_note,
        reviewed_by,
        reviewed_at,
        feedback_note,
    ) = row
    detections = _feedback_detections_from_json(detections_json)
    flagged_value = bool(is_flagged)
    image_url, thumbnail_url = _image_urls(image_path, request)
    return ViolationItem(
        id=violation_id,
        site_name=site,
        stream_name=stream_name,
        detection_time=detection_time.astimezone(),
        detected_at=detection_time.astimezone(),
        image_path=image_path,
        image_url=image_url,
        thumbnail_url=thumbnail_url,
        created_at=created_at.astimezone(),
        detection_items=detections_json,
        warnings=warnings_json,
        warning_text=_warning_text_from_json(warnings_json),
        cone_polygons=cone_polygon_json,
        pole_polygons=pole_polygon_json,
        detections=detections,
        feedback_detections=detections,
        is_flagged=flagged_value,
        flag_reason=flag_reason,
        flagged_by=flagged_by,
        flagged_at=flagged_at,
        review_status=review_status if flagged_value else None,
        review_note=review_note,
        reviewed_by=reviewed_by,
        reviewed_at=reviewed_at,
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
        type=feedback.feedback_type,  # type: ignore[arg-type]
        note=feedback.note,
        target_detection_id=feedback.target_detection_id,
        original_label=feedback.original_label,
        corrected_label=feedback.corrected_label,
        original_bbox=feedback.original_bbox,
        corrected_bbox=feedback.corrected_bbox,
        model_version=feedback.model_version,
        confidence=feedback.confidence,
        status=feedback.status,  # type: ignore[arg-type]
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
        type=feedback.feedback_type,  # type: ignore[arg-type]
        target_detection_id=feedback.target_detection_id,
        original_label=feedback.original_label,
        corrected_label=feedback.corrected_label,
        original_bbox=feedback.original_bbox,
        corrected_bbox=feedback.corrected_bbox,
        model_version=feedback.model_version,
        confidence=feedback.confidence,
        note=feedback.note,
        status=feedback.status,  # type: ignore[arg-type]
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
    action = getattr(audit_log, 'action', None) or 'review_status_changed'
    return ViolationReviewAuditItem(
        id=audit_log.id,
        violation_id=audit_log.violation_id,
        actor_user_id=audit_log.reviewed_by,
        action=action,
        old_status=audit_log.old_status,  # type: ignore[arg-type]
        new_status=audit_log.new_status,  # type: ignore[arg-type]
        note=audit_log.review_note,
        flagged_reason=getattr(audit_log, 'flagged_reason', None),
        created_at=audit_log.reviewed_at.astimezone(),
    )


async def _load_latest_feedback_note(
    db: AsyncSession,
    violation_id: int,
) -> str | None:
    """Return the newest non-empty feedback note for a violation.

    Args:
        db: Database session used to load feedback.
        violation_id: Identifier of the violation.

    Returns:
        Latest feedback note, or ``None`` when none exists.
    """
    result = await db.execute(
        select(ViolationFeedback.note)
        .where(
            ViolationFeedback.violation_id == violation_id,
            ViolationFeedback.note.is_not(None),
        )
        .order_by(
            ViolationFeedback.created_at.desc(),
            ViolationFeedback.id.desc(),
        )
        .limit(1),
    )
    return result.scalar()


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


def _split_violation_row_total(row: Any) -> tuple[Any, int | None]:
    """Split an optional window-count column from a selected row.

    Args:
        row: ORM or tuple-like selected violation row.

    Returns:
        Violation columns and optional total count.
    """
    mapping = getattr(row, '_mapping', None)
    if mapping is not None and 'total_count' in mapping:
        try:
            row_length = len(row)
        except TypeError:
            return row, int(mapping['total_count'])
        if row_length == _violation_column_count + 1:
            return row[:_violation_column_count], int(mapping['total_count'])
        return row, int(mapping['total_count'])

    try:
        row_length = len(row)
    except TypeError:
        return row, None

    if row_length == _violation_column_count + 1:
        return row[:_violation_column_count], int(row[-1])
    return row, None


def _scalar_value(value: Any) -> Any:
    """Return a scalar result value, unwrapping named SQL values.

    Args:
        value: Database scalar or named SQL enum-like value.

    Returns:
        Underlying scalar value.
    """
    if hasattr(value, 'name'):
        return value.name
    return value


async def _authorize_violation_media_access(
    image_path: str,
    username: str,
    db: AsyncSession,
) -> tuple[Path, str]:
    """Resolve media and ensure it belongs to an accessible record.

    Args:
        image_path: Untrusted stored relative image path.
        username: Requesting username.
        db: Database session used to verify violation ownership.

    Returns:
        Authorised absolute media path and its response media type.

    Raises:
        HTTPException: If the path, media, record, or site access is invalid.
    """
    safe_rel_path = _normalize_safe_rel_path(image_path, path_cls=Path)
    base_dir: Path = Path(STATIC_DIR).resolve()
    full_path: Path = _resolve_and_authorize(
        base_dir,
        safe_rel_path,
        username,
        path_cls=Path,
    )

    if not full_path.exists():
        raise HTTPException(status_code=404, detail='Image not found')

    media_type = _determine_media_type(full_path)
    site_names = await get_user_sites_cached(username, db)
    if not site_names:
        raise HTTPException(status_code=403, detail='Access denied')

    result = await db.execute(
        select(Violation.id)
        .where(
            Violation.image_path == safe_rel_path.as_posix(),
            Violation.site.in_(site_names),
        )
        .limit(1),
    )
    if result.scalar() is None:
        raise HTTPException(status_code=403, detail='Access denied')

    return full_path, media_type


def _thumbnail_cache_path(full_path: Path) -> Path:
    """Return the deterministic cached thumbnail path for an image.

    Args:
        full_path: Authorised absolute original image path.

    Returns:
        JPEG thumbnail path below the dedicated cache directory.
    """
    base_dir = Path(STATIC_DIR).resolve()
    rel_path = full_path.relative_to(base_dir)
    return (base_dir / THUMBNAIL_DIR_NAME / rel_path).with_suffix('.jpg')


def _has_recognized_image_header(source_path: Path) -> bool:
    """Recognise image headers before using Pillow's lazy opener.

    Args:
        source_path: Original evidence image path.

    Returns:
        ``True`` when the header identifies a supported image container.
    """
    with source_path.open('rb') as source_file:
        header = source_file.read(THUMBNAIL_HEADER_SCAN_BYTES)

    parser = ImageFile.Parser()
    parser.feed(header)
    return parser.image is not None or (
        len(header) >= 12
        and header[4:8] == b'ftyp'
        and header[8:12] in _ISOBMFF_IMAGE_BRANDS
    )


def _generate_thumbnail_sync(source_path: Path, thumbnail_path: Path) -> None:
    """Generate or refresh a cached thumbnail on disk.

    Args:
        source_path: Original evidence image path.
        thumbnail_path: JPEG cache path to create or refresh.

    Raises:
        HTTPException: If the source is not supported image content.
    """
    if (
        thumbnail_path.exists()
        and thumbnail_path.stat().st_mtime >= source_path.stat().st_mtime
    ):
        return

    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not _has_recognized_image_header(source_path):
            raise UnidentifiedImageError(str(source_path))
        with Image.open(source_path) as opened_image:
            image: Image.Image = opened_image
            image.thumbnail((THUMBNAIL_MAX_EDGE, THUMBNAIL_MAX_EDGE))
            if image.mode not in {'RGB', 'L'}:
                image = image.convert('RGB')
            image.save(
                thumbnail_path,
                format='JPEG',
                quality=THUMBNAIL_QUALITY,
                optimize=True,
            )
    except (ModuleNotFoundError, OSError) as exc:
        raise HTTPException(
            status_code=400,
            detail='Unsupported image content',
        ) from exc


async def _ensure_thumbnail(source_path: Path) -> Path:
    """Generate a thumbnail in a worker thread and return its path.

    Args:
        source_path: Original evidence image path.

    Returns:
        Generated or current cached JPEG thumbnail path.
    """
    thumbnail_path = _thumbnail_cache_path(source_path)
    await asyncio.to_thread(
        _generate_thumbnail_sync,
        source_path,
        thumbnail_path,
    )
    return thumbnail_path


def _image_size_for_violation(image_path: str) -> tuple[int, int] | None:
    """Load image dimensions for a detailed violation response.

    Args:
        image_path: Stored relative evidence-image path.

    Returns:
        Image width and height, or ``None`` when unavailable.
    """
    try:
        safe_rel_path = _normalize_safe_rel_path(image_path, path_cls=Path)
        base_dir: Path = Path(STATIC_DIR).resolve()
        full_path: Path = _resolve_and_authorize(
            base_dir,
            safe_rel_path,
            '_internal',
            path_cls=Path,
        )
        if not full_path.exists():
            return None
        with Image.open(full_path) as image:
            return image.size
    except Exception:
        return None


def _encode_violation_cursor(item: ViolationItem) -> str:
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


def _empty_analytics_response() -> ViolationAnalyticsResponse:
    """Return the canonical empty analytics payload.

    Returns:
        Analytics response with zero summary and empty aggregate series.
    """
    return ViolationAnalyticsResponse(
        summary=ViolationAnalyticsSummary(total=0, today=0),
        trend=[],
        by_type=[],
        by_site=[],
        by_hour=[],
    )


def _normalise_utc(value: datetime) -> datetime:
    """Treat naive values as UTC and convert aware values to UTC.

    Args:
        value: Datetime supplied by an analytics client.

    Returns:
        Timezone-aware UTC datetime.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _validate_analytics_range(start: datetime, end: datetime) -> tuple[
    datetime,
    datetime,
]:
    """Validate and normalise an analytics query range.

    Args:
        start: Requested inclusive range start.
        end: Requested inclusive range end.

    Returns:
        Validated UTC start and end values.

    Raises:
        HTTPException: If the range is reversed or exceeds the maximum span.
    """
    start_utc = _normalise_utc(start)
    end_utc = _normalise_utc(end)
    if start_utc >= end_utc:
        raise HTTPException(
            status_code=422,
            detail='start must be before end',
        )
    try:
        latest_end = start_utc.replace(
            year=start_utc.year + MAX_ANALYTICS_RANGE_YEARS,
        )
    except ValueError:
        # February 29 has no matching date in a non-leap target year.
        latest_end = start_utc.replace(
            year=start_utc.year + MAX_ANALYTICS_RANGE_YEARS,
            day=28,
        )
    if end_utc > latest_end:
        raise HTTPException(
            status_code=422,
            detail='Query range must not exceed 5 years',
        )
    return start_utc, end_utc


def _analytics_dialect_name(db: AsyncSession) -> str:
    """Return the SQL dialect name for analytics expressions.

    Args:
        db: Database session whose bound dialect is inspected.

    Returns:
        Lower-case SQL dialect name, or an empty string when unbound.
    """
    bind = getattr(db, 'bind', None)
    dialect = getattr(bind, 'dialect', None)
    return str(getattr(dialect, 'name', '') or '')


def _analytics_bucket_expr(
    bucket: AnalyticsBucket,
    db: AsyncSession,
) -> ColumnElement[Any]:
    """Build a dialect-aware UTC bucket expression for detection time.

    Args:
        bucket: Requested analytics time bucket.
        db: Database session whose SQL dialect is inspected.

    Returns:
        SQL expression that labels each violation's time bucket.
    """
    dialect_name = _analytics_dialect_name(db)
    if dialect_name == 'postgresql':
        formats = {
            'hour': 'YYYY-MM-DD"T"HH24:00:00"Z"',
            'day': 'YYYY-MM-DD',
            'week': 'IYYY-"W"IW',
        }
        return func.to_char(Violation.detection_time, formats[bucket])
    if dialect_name in {'mysql', 'mariadb'}:
        formats = {
            'hour': '%Y-%m-%dT%H:00:00Z',
            'day': '%Y-%m-%d',
            'week': '%x-W%v',
        }
        return func.date_format(Violation.detection_time, formats[bucket])
    if dialect_name == 'sqlite':
        formats = {
            'hour': '%Y-%m-%dT%H:00:00Z',
            'day': '%Y-%m-%d',
            'week': '%Y-W%W',
        }
        return func.strftime(formats[bucket], Violation.detection_time)

    formats = {
        'hour': 'YYYY-MM-DD"T"HH24:00:00"Z"',
        'day': 'YYYY-MM-DD',
        'week': 'IYYY-"W"IW',
    }
    return func.to_char(Violation.detection_time, formats[bucket])


def _analytics_hour_expr(db: AsyncSession) -> ColumnElement[Any]:
    """Build a dialect-aware UTC-hour expression for detection time.

    Args:
        db: Database session whose SQL dialect is inspected.

    Returns:
        SQL expression that extracts the UTC hour.
    """
    dialect_name = _analytics_dialect_name(db)
    if dialect_name == 'postgresql':
        return cast(func.extract('hour', Violation.detection_time), Integer)
    if dialect_name in {'mysql', 'mariadb'}:
        return func.hour(Violation.detection_time)
    if dialect_name == 'sqlite':
        return cast(func.strftime('%H', Violation.detection_time), Integer)
    return cast(func.extract('hour', Violation.detection_time), Integer)


def _canonical_violation_type(violation_type: str) -> str:
    """Validate and normalise a supported violation-type code.

    Args:
        violation_type: Client-supplied type code.

    Returns:
        Canonical violation-type code.

    Raises:
        HTTPException: If the type code is unsupported.
    """
    canonical = normalise_violation_type(violation_type)
    if canonical is not None:
        return canonical
    valid = ', '.join(
        definition.code for definition in VIOLATION_TYPE_DEFINITIONS
    )
    raise HTTPException(
        status_code=422,
        detail=f"Unsupported violation_type. Expected one of: {valid}",
    )


def _type_condition(
    violation_type: str,
    db: AsyncSession,
) -> ColumnElement[bool]:
    """Build a dialect-aware filter for a canonical violation type.

    Args:
        violation_type: Client-supplied type code.
        db: Database session whose SQL dialect is inspected.

    Returns:
        SQL predicate matching stored canonical type codes.
    """
    canonical = _canonical_violation_type(violation_type)
    dialect_name = _analytics_dialect_name(db)
    if dialect_name == 'postgresql':
        return cast(Violation.violation_type_codes, JSONB).contains(
            [canonical],
        )
    if dialect_name in {'mysql', 'mariadb'}:
        return (
            func.json_contains(
                Violation.violation_type_codes,
                json.dumps([canonical]),
            )
            == 1
        )
    return cast(Violation.violation_type_codes, String).like(
        f'%"{canonical}"%',
    )


async def _resolve_stream_filter(
    stream_id: str,
    site_name: str | None,
    site_names: list[str],
    user: _StreamScopeUser,
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
    user, sites = await load_user_with_effective_sites(username, db)
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
    return await get_user_sites_cached(username, db)


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
            db,
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
    site_name = _scalar_value((await db.execute(site_stmt)).scalar())
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
    user, _ = await load_user_with_effective_sites(
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
    db: AsyncSession,
) -> list:
    """Return independent type, text, time, and cursor conditions.

    Args:
        violation_type: Optional type filter.
        keyword: Optional keyword filter.
        start_time: Optional range start.
        end_time: Optional range end.
        cursor: Optional pagination cursor.
        db: Database session used for dialect-aware type filtering.

    Returns:
        SQL predicates independent of authorisation scope.
    """
    conditions: list = []
    if violation_type:
        conditions.append(_type_condition(violation_type, db))
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


async def _query_violation_page(
    db: AsyncSession,
    where_clause: ColumnElement[bool],
    limit: int,
    offset: int,
    cursor: str | None,
) -> list[Any]:
    """Fetch one offset- or keyset-paginated violation result window.

    Args:
        db: Database session used to query violations.
        where_clause: Fully authorised SQL filter predicate.
        limit: Requested page size.
        offset: Legacy offset-pagination position.
        cursor: Optional keyset-pagination cursor.

    Returns:
        Result rows including one extra row for next-cursor detection.
    """
    statement = (
        select(*_violation_columns, func.count().over().label('total_count'))
        .where(where_clause)
        .order_by(Violation.detection_time.desc(), Violation.id.desc())
        .limit(limit + 1)
    )
    if not cursor:
        statement = statement.offset(offset)
    return (await db.execute(statement)).all()


def _violation_page_response(
    rows: list[Any],
    request: Request,
    limit: int,
) -> tuple[int, list[ViolationItem], str | None]:
    """Convert a query window into items and an optional next cursor.

    Args:
        rows: Query rows including the optional look-ahead record.
        request: Request used to build protected media URLs.
        limit: Requested page size.

    Returns:
        Total count, response items, and optional keyset cursor.
    """
    rows_to_return = rows[:limit]
    total = 0
    items: list[ViolationItem] = []
    for row in rows_to_return:
        item_row, row_total = _split_violation_row_total(row)
        if row_total is not None:
            total = row_total
        items.append(_violation_to_item(item_row, request))
    next_cursor = None
    if len(rows) > limit and items:
        next_cursor = _encode_violation_cursor(items[-1])
    return total, items, next_cursor


async def _empty_offset_total(
    db: AsyncSession,
    rows: list[Any],
    offset: int,
    cursor: str | None,
    where_clause: ColumnElement[bool],
    current_total: int,
) -> int:
    """Query total only for an empty offset page.

    Args:
        db: Database session used for a fallback count.
        rows: Current page query rows.
        offset: Legacy offset-pagination position.
        cursor: Optional keyset-pagination cursor.
        where_clause: Fully authorised SQL filter predicate.
        current_total: Total supplied by a window count, if available.

    Returns:
        Existing or fallback total count.
    """
    if rows or not offset or cursor:
        return current_total
    result = await db.execute(
        select(func.count()).select_from(Violation).where(where_clause),
    )
    return int(result.scalar() or 0)


async def require_violation_analytics_access(
    username: str,
    db: AsyncSession,
) -> tuple[User, list[str]]:
    """Return accessible sites when a user may view violation analytics.

    Args:
        username: Authenticated username.
        db: Database session used to load effective site access.

    Returns:
        Authorised user and accessible site names.

    Raises:
        HTTPException: If the user lacks the required analytics role.
    """
    user, sites = await load_user_with_effective_sites(
        username,
        db,
        status_code=401,
        detail='Invalid user',
    )
    if user.role not in ALLOWED_VIOLATION_ANALYTICS_ROLES:
        raise HTTPException(
            status_code=403,
            detail='violation_analytics_forbidden',
        )
    return user, [site.name for site in sites]


async def get_my_sites(
    db: AsyncSession,
    credentials: JwtAuthorizationCredentials,
) -> list[SiteOut]:
    """Retrieve all sites accessible by the currently logged-in user.

    Args:
        db (AsyncSession): The SQLAlchemy async session.
        credentials (JwtAuthorizationCredentials): The JWT credentials from
            the request.

    Returns:
        list[dict]: A list of dictionaries containing the site's ID, name,
            creation timestamp, and update timestamp.

    Raises:
        HTTPException: If the token is invalid (401) or the user is not found
            (404).
    """
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    _, sites = await get_user_effective_sites(username, db)

    return [
        SiteOut(
            id=s.id,
            name=s.name,
            created_at=s.created_at,
            updated_at=s.updated_at,
        )
        for s in sites
    ]


async def get_violation_filter_options(
    site_id: int,
    group_id: int | None,
    db: AsyncSession,
    credentials: JwtAuthorizationCredentials,
) -> ViolationFilterOptions:
    """Return type codes and cameras visible within a selected site.

    Args:
        site_id: Selected site identifier.
        group_id: Optional group restriction.
        db: Database session used to load stream configurations.
        credentials: Validated requesting-user credentials.

    Returns:
        Authorised camera and violation-type options.
    """
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    user, sites = await load_user_with_effective_sites(
        username,
        db,
        status_code=401,
        detail='Invalid user',
    )
    site = next(
        (candidate for candidate in sites if candidate.id == site_id),
        None,
    )
    if site is None:
        raise HTTPException(status_code=403, detail='No access to site_id')

    if (
        group_id is not None
        and user.role != 'super_admin'
        and group_id != user.group_id
    ):
        raise HTTPException(status_code=403, detail='No access to group_id')

    visible_group_id = group_id
    if user.role != 'super_admin':
        visible_group_id = user.group_id

    stream_statement = (
        select(StreamConfig.id, StreamConfig.stream_name)
        .where(StreamConfig.site_id == site.id)
        .order_by(StreamConfig.stream_name, StreamConfig.id)
    )
    if visible_group_id is not None:
        stream_statement = stream_statement.where(
            StreamConfig.group_id == visible_group_id,
        )
    stream_result = await db.execute(stream_statement)

    return ViolationFilterOptions(
        cameras=[
            ViolationFilterCamera(stream_id=str(row[0]), name=str(row[1]))
            for row in stream_result.all()
        ],
        violation_types=[
            ViolationTypeOption(code=definition.code, label=definition.label)
            for definition in VIOLATION_TYPE_DEFINITIONS
        ],
    )


async def get_violations(
    request: Request,
    db: AsyncSession,
    credentials: JwtAuthorizationCredentials,
    *,
    site_id: int | None = None,
    stream_id: str | None = None,
    violation_type: str | None = None,
    keyword: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: int = 20,
    offset: int = 0,
    flagged: bool | None = None,
    review_status: ViolationReviewStatus | None = None,
    cursor: str | None = None,
) -> ViolationList:
    """Retrieve a paginated list of violation records.

    Args:
        site_id (int | None): The ID of the site to filter violations by.
        keyword (str | None): A keyword to search for in violation records.
        start_time (datetime | None): The start of the detection time range.
        end_time (datetime | None): The end of the detection time range.
        limit (int): The maximum number of records to return (default is 20).
        offset (int): The starting record offset (default is 0).
        db (AsyncSession): The SQLAlchemy async session.
        credentials (JwtAuthorizationCredentials):
            The JWT credentials from the request.

    Returns:
        ViolationList: A dictionary with:
            - 'total': the total count of matching violations,
            - 'items': a list of violation records (paginated).

    Raises:
        HTTPException: If the token is invalid (401), if the user is not found
            (404), if the user lacks access to the site (403), or if any other
            error occurs.
    """
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    site_names = await _violation_site_names(
        username,
        flagged,
        review_status,
        db,
    )
    if not site_names:
        return ViolationList(total=0, items=[])
    conditions = await _build_violation_conditions(
        username,
        site_names,
        site_id,
        stream_id,
        violation_type,
        keyword,
        start_time,
        end_time,
        flagged,
        review_status,
        cursor,
        db,
    )
    where_clause = and_(*conditions)
    rows = await _query_violation_page(
        db,
        where_clause,
        limit,
        offset,
        cursor,
    )
    total, items, next_cursor = _violation_page_response(rows, request, limit)
    total = await _empty_offset_total(
        db,
        rows,
        offset,
        cursor,
        where_clause,
        total,
    )
    return ViolationList(total=total, items=items, next_cursor=next_cursor)


async def get_violation_analytics(
    start: datetime,
    end: datetime,
    db: AsyncSession,
    credentials: JwtAuthorizationCredentials,
    *,
    site_id: int | None = None,
    stream_id: str | None = None,
    violation_type: str | None = None,
    bucket: AnalyticsBucket = 'day',
) -> ViolationAnalyticsResponse:
    """Return aggregated violation counts for charts and KPI widgets.

    The response intentionally excludes image paths, warning payloads, and
    individual violation records.
    """
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    user, site_names = await require_violation_analytics_access(username, db)
    start_utc, end_utc = _validate_analytics_range(start, end)
    if not site_names:
        return _empty_analytics_response()

    conditions: list[ColumnElement[bool]] = [
        Violation.site.in_(site_names),
        Violation.detection_time >= start_utc,
        Violation.detection_time <= end_utc,
    ]

    if site_id is not None:
        site_name_set = set(site_names)
        site_stmt = select(Site.name).where(Site.id == site_id)
        site_name = _scalar_value((await db.execute(site_stmt)).scalar())
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

    if violation_type:
        conditions.append(_type_condition(violation_type, db))

    where_clause = and_(*conditions)
    total_result = await db.execute(
        select(func.count()).select_from(Violation).where(where_clause),
    )
    total = int(total_result.scalar() or 0)
    if total == 0:
        return _empty_analytics_response()

    now_utc = datetime.now(timezone.utc)
    today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(days=1)
    today_result = await db.execute(
        select(func.count())
        .select_from(Violation)
        .where(
            where_clause,
            Violation.detection_time >= today_start,
            Violation.detection_time < today_end,
        ),
    )
    today = int(today_result.scalar() or 0)

    bucket_expr = _analytics_bucket_expr(bucket, db).label('bucket')
    trend_result = await db.execute(
        select(bucket_expr, func.count().label('count'))
        .select_from(Violation)
        .where(where_clause)
        .group_by(bucket_expr)
        .order_by(bucket_expr),
    )
    trend = [
        ViolationAnalyticsTrendItem(
            bucket=str(row[0]),
            count=int(row[1] or 0),
        )
        for row in trend_result.all()
    ]

    site_result = await db.execute(
        select(Site.id, Site.name, func.count().label('count'))
        .select_from(Violation)
        .join(Site, Violation.site == Site.name)
        .where(where_clause)
        .group_by(Site.id, Site.name)
        .order_by(func.count().desc(), Site.id),
    )
    by_site = [
        ViolationAnalyticsSiteItem(
            site_id=int(row[0]),
            site_name=str(row[1]),
            count=int(row[2] or 0),
        )
        for row in site_result.all()
    ]

    hour_expr = _analytics_hour_expr(db).label('hour')
    hour_result = await db.execute(
        select(hour_expr, func.count().label('count'))
        .select_from(Violation)
        .where(where_clause)
        .group_by(hour_expr)
        .order_by(hour_expr),
    )
    by_hour = [
        ViolationAnalyticsHourItem(hour=int(row[0]), count=int(row[1] or 0))
        for row in hour_result.all()
    ]

    by_type: list[ViolationAnalyticsTypeItem] = []
    type_names = (
        [_canonical_violation_type(violation_type)]
        if violation_type
        else [definition.code for definition in VIOLATION_TYPE_DEFINITIONS]
    )
    for type_name in type_names:
        definition = VIOLATION_TYPE_BY_CODE[type_name]
        type_result = await db.execute(
            select(func.count())
            .select_from(Violation)
            .where(where_clause, _type_condition(type_name, db)),
        )
        type_count = int(type_result.scalar() or 0)
        if type_count:
            by_type.append(
                ViolationAnalyticsTypeItem(
                    type=type_name,
                    label=definition.label,
                    count=type_count,
                ),
            )

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


async def get_next_review_violation(
    request: Request,
    db: AsyncSession,
    credentials: JwtAuthorizationCredentials,
    *,
    review_status: ViolationReviewStatus = 'pending',
    site_id: int | None = None,
    current_id: int | None = None,
) -> ViolationItem | None:
    """Return the next flagged record an administrator may review.

    Args:
        request: Request used to build protected media URLs.
        db: Database session used to query violations.
        credentials: Validated reviewer credentials.
        review_status: Required review state.
        site_id: Optional selected site identifier.
        current_id: Optional current record to exclude.

    Returns:
        Next reviewable violation, or ``None`` when none matches.
    """
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    _, site_names = await _load_review_scope(username, db)
    if not site_names:
        return None

    conditions: list = [
        Violation.site.in_(site_names),
        Violation.is_flagged.is_(True),
        Violation.review_status == review_status,
    ]
    if current_id is not None:
        conditions.append(Violation.id != current_id)
    if site_id is not None:
        site_name_set = set(site_names)
        site_stmt = select(Site.name).where(Site.id == site_id)
        site_name = _scalar_value((await db.execute(site_stmt)).scalar())
        if not site_name or site_name not in site_name_set:
            raise HTTPException(status_code=403, detail='No access to site_id')
        conditions.append(Violation.site == site_name)

    result = await db.execute(
        select(*_violation_columns)
        .where(and_(*conditions))
        .order_by(
            Violation.flagged_at.asc().nullslast(),
            Violation.detection_time.asc(),
            Violation.id.asc(),
        )
        .limit(1),
    )
    row = result.first()
    if not row:
        return None

    item = _violation_to_item(row, request)
    item.feedbacks = await _load_violation_feedbacks(db, item.id)
    item.review_audit_logs = await _load_review_audit_logs(db, item.id)
    item.overlay_objects = _overlay_objects_from_feedback(
        item.detections,
        item.feedbacks or [],
        _image_size_for_violation(item.image_path),
    )
    return item


async def get_violation_review_audit_log(
    violation_id: int,
    db: AsyncSession,
    credentials: JwtAuthorizationCredentials,
) -> list[ViolationReviewAuditItem]:
    """Return review history for a flagged record in reviewer scope.

    Args:
        violation_id: Identifier of the reviewed violation.
        db: Database session used to load audit records.
        credentials: Validated reviewer credentials.

    Returns:
        Newest-first public review audit items.
    """
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    _, site_names = await _load_review_scope(username, db)
    if not site_names:
        raise HTTPException(
            status_code=403,
            detail='No access to this violation',
        )

    result = await db.execute(
        select(Violation.id).where(
            Violation.id == violation_id,
            Violation.site.in_(site_names),
            Violation.is_flagged.is_(True),
        ),
    )
    if result.scalar() is None:
        raise HTTPException(
            status_code=403,
            detail='No access to this violation',
        )

    return await _load_review_audit_logs(db, violation_id)


async def get_single_violation(
    violation_id: int,
    request: Request,
    db: AsyncSession,
    credentials: JwtAuthorizationCredentials,
) -> ViolationItem:
    """Retrieve detailed information for a specific violation record.

    Args:
        violation_id (int):
            The ID of the violation to retrieve.
        db (AsyncSession):
            The SQLAlchemy async session.
        credentials (JwtAuthorizationCredentials):
            The JWT credentials from the request.

    Returns:
        dict: A dictionary containing details of the violation, including:
            - 'id': The ID of the violation.
            - 'site_name': The name of the site.
            - 'stream_name': The name of the stream.
            - 'detection_time': The time of detection.
            - 'image_path': The path to the image.
            - 'created_at': The creation timestamp.
            - 'detection_items': JSON string with detection items.
            - 'warnings': JSON string with warnings.
            - 'cone_polygons': JSON string with cone polygons.
            - 'pole_polygons': JSON string with pole polygons.

    Raises:
        HTTPException: If the token is invalid (401), if the user is not found
            (404), if the user lacks access to the violation's site (403), or
            if the violation ID does not exist (404).
    """
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    # Retrieve user sites using the cache
    site_names: list[str] = await get_user_sites_cached(username, db)
    if not site_names:
        raise HTTPException(
            status_code=403,
            detail='No access to this violation',
        )

    stmt_violation = select(*_violation_columns).where(
        Violation.id == violation_id,
        Violation.site.in_(site_names),
    )
    result = await db.execute(stmt_violation)
    row = result.first() if hasattr(result, 'first') else result.scalar()

    if not row:
        print(
            f"[get_single_violation] No access to violation_id {violation_id}",
        )
        raise HTTPException(
            status_code=403,
            detail='No access to this violation',
        )
    if hasattr(row, 'site') and row.site not in site_names:
        print(
            f"[get_single_violation] No access to violation_id {violation_id}",
        )
        raise HTTPException(
            status_code=403,
            detail='No access to this violation',
        )

    item = _violation_to_item(row, request)
    item.feedbacks = await _load_violation_feedbacks(db, violation_id)
    if item.feedback_note is None:
        item.feedback_note = next(
            (feedback.note for feedback in item.feedbacks if feedback.note),
            None,
        )
    item.overlay_objects = _overlay_objects_from_feedback(
        item.detections,
        item.feedbacks or [],
        _image_size_for_violation(item.image_path),
    )
    if item.is_flagged:
        item.review_audit_logs = await _load_review_audit_logs(
            db,
            violation_id,
        )
    return item


async def submit_violation_feedback(
    violation_id: int,
    payload: ViolationFeedbackCreate,
    db: AsyncSession,
    credentials: JwtAuthorizationCredentials,
) -> ViolationFeedbackResponse:
    """Store structured feedback against a persisted violation record.

    Feedback is intentionally not accepted as training data immediately. The
    row starts as ``pending`` and can later be reviewed by a human workflow.
    """
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    site_names: list[str] = await get_user_sites_cached(username, db)
    if not site_names:
        raise HTTPException(
            status_code=403,
            detail='No access to this violation',
        )

    stmt_violation = select(Violation).where(
        Violation.id == violation_id,
        Violation.site.in_(site_names),
    )
    violation_result = await db.execute(stmt_violation)
    violation = violation_result.scalar_one_or_none()
    if not violation:
        raise HTTPException(
            status_code=403,
            detail='No access to this violation',
        )

    feedback_detection_ids = _feedback_detection_ids_from_json(
        violation.detections_json,
    )
    if (
        feedback_detection_ids is not None
        and payload.target_detection_id
        and payload.target_detection_id not in feedback_detection_ids
    ):
        raise HTTPException(
            status_code=422,
            detail='target_detection_id does not belong to this violation',
        )

    user_result = await db.execute(
        select(User.id).where(User.username == username),
    )
    user_id = user_result.scalar()
    if user_id is None:
        raise HTTPException(status_code=404, detail='User not found')

    created_at = datetime.now(timezone.utc)
    violation.is_flagged = True
    violation.flag_reason = payload.type
    violation.flagged_by = int(_scalar_value(user_id))
    violation.flagged_at = created_at
    violation.review_status = 'pending'

    feedback = ViolationFeedback(
        violation_id=violation.id,
        user_id=int(_scalar_value(user_id)),
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

    try:
        db.add(feedback)
        await db.commit()
        await db.refresh(feedback)
    except Exception as exc:
        try:
            await db.rollback()
        except Exception:
            pass
        logging.error(
            f"[submit_violation_feedback] create feedback failed: {exc}",
        )
        raise HTTPException(
            status_code=500,
            detail='Failed to create violation feedback',
        ) from exc

    return _feedback_to_response(feedback)


async def review_violation(
    violation_id: int,
    payload: ViolationReviewUpdate,
    request: Request,
    db: AsyncSession,
    credentials: JwtAuthorizationCredentials,
) -> ViolationItem:
    """Update review state for a flagged violation within reviewer scope.

    Args:
        violation_id: Identifier of the violation to update.
        payload: Validated review status and optional note.
        request: Request used to build protected media URLs.
        db: Database session used to persist review and audit state.
        credentials: Validated reviewer credentials.

    Returns:
        Updated detailed violation item.
    """
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    reviewer, site_names = await _load_review_scope(username, db)
    if not site_names:
        raise HTTPException(
            status_code=403,
            detail='No access to this violation',
        )

    result = await db.execute(
        select(Violation).where(
            Violation.id == violation_id,
            Violation.site.in_(site_names),
        ),
    )
    violation = result.scalar_one_or_none()
    if not violation:
        raise HTTPException(
            status_code=403,
            detail='No access to this violation',
        )
    if not violation.is_flagged:
        raise HTTPException(
            status_code=404,
            detail='Flagged violation not found',
        )

    reviewed_at = datetime.now(timezone.utc)
    old_status = violation.review_status
    violation.review_status = payload.review_status
    violation.review_note = payload.review_note
    violation.reviewed_by = reviewer.id
    violation.reviewed_at = reviewed_at

    audit_log = ViolationReviewAuditLog(
        violation_id=violation.id,
        action='review_status_changed',
        old_status=old_status,
        new_status=payload.review_status,
        review_note=payload.review_note,
        flagged_reason=violation.flag_reason,
        reviewed_by=reviewer.id,
        reviewed_at=reviewed_at,
    )

    try:
        db.add(audit_log)
        await db.commit()
        await db.refresh(violation)
    except Exception as exc:
        try:
            await db.rollback()
        except Exception:
            pass
        logging.error(f"[review_violation] review update failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail='Failed to update violation review',
        ) from exc

    setattr(
        violation,
        'feedback_note',
        await _load_latest_feedback_note(db, violation.id),
    )
    return _violation_to_item(violation, request)


async def get_violation_image(
    image_path: str,
    db: AsyncSession,
    credentials: JwtAuthorizationCredentials,
) -> FileResponse:
    """Retrieve a violation image file from the "static" directory.

    Args:
        image_path (str): The relative path of the image within the "static"
            directory.
        credentials (JwtAuthorizationCredentials): The JWT credentials from the
            request.

    Returns:
        FileResponse: The requested image file with inline Content-Disposition.

    Raises:
        HTTPException: If the token is invalid (401), if the path contains '..'
            (400), if the path is outside "static" (403), or if the file is not
            found (404).
    """
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    full_path, media_type = await _authorize_violation_media_access(
        image_path,
        username,
        db,
    )

    return FileResponse(
        path=full_path,
        media_type=media_type,
        headers={
            'Content-Disposition': (
                f'inline; filename="{sanitize_filename(full_path.name)}"'
            ),
        },
    )


async def get_violation_thumbnail(
    image_path: str,
    db: AsyncSession,
    credentials: JwtAuthorizationCredentials,
) -> FileResponse:
    """Return an authorised thumbnail, generating it on first request.

    Args:
        image_path: Stored relative evidence-image path.
        db: Database session used to verify media access.
        credentials: Validated requesting-user credentials.

    Returns:
        Protected JPEG thumbnail response.
    """
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    full_path, _ = await _authorize_violation_media_access(
        image_path,
        username,
        db,
    )
    thumbnail_path = await _ensure_thumbnail(full_path)
    return FileResponse(
        path=thumbnail_path,
        media_type='image/jpeg',
        headers={
            'Content-Disposition': (
                f'inline; filename="{sanitize_filename(thumbnail_path.name)}"'
            ),
            'Cache-Control': 'private, max-age=86400',
        },
    )


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
    """Upload a new violation record, including an image and associated
    metadata.

    Args:
        site (str):
            The name of the site where the violation occurred.
        stream_name (str):
            The name of the video stream or camera.
        detection_time (datetime | None):
            The detection time; defaults to local now.
        warnings_json (str | None):
            JSON string describing warnings.
        detections_json (str | None):
            JSON string describing detected items.
        cone_polygon_json (str | None):
            JSON string with cone polygon data.
        pole_polygon_json (str | None):
            JSON string with pole polygon data.
        image (UploadFile):
            The violation image file.
        db (AsyncSession):
            The SQLAlchemy async session.
        credentials (JwtAuthorizationCredentials):
            The JWT credentials.

    Returns:
        dict: A dictionary containing a success message and the violation ID.

    Raises:
        HTTPException: If the token is invalid (401), if the user has no access
            to the site (403), or if any error occurs during file reading or
            database operations (400, 500).
    """
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    site_names: list[str] = await get_user_sites_cached(username, db)
    if site not in site_names:
        print(f"[upload_violation] No access to site {site}")
        raise HTTPException(status_code=403, detail='No access to this site')

    detection_time = (
        detection_time.astimezone()
        if detection_time is not None
        else datetime.now().astimezone()
    )

    try:
        violation_id: int | None = await violation_manager.save_violation(
            db=db,
            site=site,
            stream_name=stream_name,
            detection_time=detection_time,
            image_file=image,
            warnings_json=warnings_json,
            detections_json=detections_json,
            cone_polygon_json=cone_polygon_json,
            pole_polygon_json=pole_polygon_json,
        )
    except (EmptyViolationImageError, ViolationImageReadError) as exc:
        logging.error(f"[upload_violation] read error: {exc}")
        raise HTTPException(
            status_code=400,
            detail='Failed to read image file',
        )
    if not violation_id:
        raise HTTPException(
            status_code=500,
            detail='Failed to create violation record',
        )

    return UploadViolationResponse(
        message='Violation uploaded successfully.',
        violation_id=violation_id,
    )
