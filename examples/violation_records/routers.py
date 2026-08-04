from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
from collections.abc import Sequence
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Final
from typing import Literal
from typing import Protocol
from urllib.parse import urlencode

from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import Query
from fastapi import Request
from fastapi import Security
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
from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
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

# Create a global SearchUtils instance for expanding synonyms in query filters.
search_util: SearchUtils = SearchUtils(device=-1)

# Create a FastAPI router for violations-related endpoints.
router: APIRouter = APIRouter()

# Note: effective site access helpers are provided by
# examples.auth.user_service

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
    """Identity fields needed to authorize a selected camera stream."""

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


def _decode_detection_items(value: str | None) -> list[Any] | None:
    """Decode stored detections into a list, when the shape is known."""
    if not value:
        return None
    try:
        data = json.loads(value)
    except Exception:
        return None

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ('detections', 'detection_items', 'items'):
            items = data.get(key)
            if isinstance(items, list):
                return items
    return None


def _warning_text_from_json(value: str | None) -> str | None:
    """Build a short warning summary for lightweight list rows."""
    if not value:
        return None
    try:
        data = json.loads(value)
    except Exception:
        return value[:200]

    if isinstance(data, str):
        return data[:200]
    if isinstance(data, dict):
        parts: list[str] = []
        for key, raw_value in data.items():
            if isinstance(raw_value, dict) and 'count' in raw_value:
                parts.append(f'{key}: {raw_value["count"]}')
            elif raw_value not in {None, '', False}:
                parts.append(str(key))
        return ', '.join(parts)[:200] if parts else None
    if isinstance(data, list):
        return ', '.join(str(item) for item in data[:5])[:200]
    return None


def _media_endpoint_url(
    endpoint_name: str,
    image_path: str,
    request: Request | None,
) -> str:
    """Build a protected media endpoint URL for an image path."""
    query = urlencode({'image_path': image_path})
    if request is None:
        return f"/{endpoint_name}?{query}"
    return f"{request.url_for(endpoint_name)}?{query}"


def _image_urls(
    image_path: str,
    request: Request | None,
) -> tuple[str, str]:
    """Return original image and thumbnail API URLs."""
    return (
        _media_endpoint_url('get_violation_image', image_path, request),
        _media_endpoint_url('get_violation_thumbnail', image_path, request),
    )


def _bbox_from_dict(raw: dict[str, Any]) -> list[float] | None:
    """Normalise common dict bbox shapes into [x1, y1, x2, y2]."""
    if {'x1', 'y1', 'x2', 'y2'}.issubset(raw):
        return [
            float(raw['x1']),
            float(raw['y1']),
            float(raw['x2']),
            float(raw['y2']),
        ]
    if (
        {'x', 'y'}.issubset(raw)
        and ('width' in raw or 'w' in raw)
        and ('height' in raw or 'h' in raw)
    ):
        x = float(raw['x'])
        y = float(raw['y'])
        width_value = raw.get('width', raw.get('w'))
        height_value = raw.get('height', raw.get('h'))
        if width_value is None or height_value is None:
            return None
        width = float(width_value)
        height = float(height_value)
        return [x, y, x + width, y + height]
    return None


def _bbox_from_detection_item(item: Any) -> list[float] | None:
    """Extract a bbox from dict or YOLO-list detection metadata."""
    try:
        if isinstance(item, dict):
            bbox = item.get('bbox') or item.get('box')
            if isinstance(bbox, dict):
                return _bbox_from_dict(bbox)
            if isinstance(bbox, list | tuple) and len(bbox) >= 4:
                return [float(bbox[i]) for i in range(4)]
        if isinstance(item, list | tuple) and len(item) >= 4:
            return [float(item[i]) for i in range(4)]
    except (TypeError, ValueError):
        return None
    return None


def _feedback_detection_id_candidates(item: Any, index: int) -> set[str]:
    """Return all IDs the feedback endpoint can accept for one detection."""
    candidates = {f"det_{index}"}
    if isinstance(item, dict):
        for key in ('id', 'detection_id', 'target_detection_id', 'track_id'):
            value = item.get(key)
            if value not in {None, ''}:
                candidates.add(str(value))
    elif isinstance(item, list | tuple) and len(item) >= 7:
        track_id = item[6]
        if track_id not in {None, '', -1}:
            candidates.add(str(track_id))
    return candidates


def _feedback_detection_from_item(
    item: Any,
    index: int,
) -> FeedbackDetectionItem | None:
    """Build a compact detection item that the frontend can select."""
    detection_id = f"det_{index}"
    label: str | None = None
    confidence: float | None = None

    try:
        if isinstance(item, dict):
            raw_id = item.get('id') or item.get('detection_id')
            if raw_id not in {None, ''}:
                detection_id = str(raw_id)
            raw_label = (
                item.get('class_name')
                or item.get('class')
                or item.get('label')
            )
            if raw_label not in {None, ''}:
                label = str(raw_label)
            raw_confidence = item.get('confidence', item.get('conf'))
            if raw_confidence is not None:
                confidence = float(raw_confidence)
        elif isinstance(item, list | tuple) and len(item) >= 6:
            confidence = float(item[4])
            label = f"class-{int(float(item[5]))}"
    except (TypeError, ValueError):
        pass

    bbox = _bbox_from_detection_item(item)
    if bbox is None and label is None and confidence is None:
        return None

    return FeedbackDetectionItem(
        id=detection_id,
        label=label,
        confidence=confidence,
        bbox=bbox,
    )


def _feedback_detections_from_json(
    value: str | None,
) -> list[FeedbackDetectionItem] | None:
    """Return normalised detection selections for violation responses."""
    items = _decode_detection_items(value)
    if items is None:
        return None
    return [
        detection
        for index, item in enumerate(items)
        if (detection := _feedback_detection_from_item(item, index))
        is not None
    ]


def _feedback_detection_ids_from_json(value: str | None) -> set[str] | None:
    """Return target_detection_id values accepted for a stored record."""
    items = _decode_detection_items(value)
    if items is None:
        return None
    ids: set[str] = set()
    for index, item in enumerate(items):
        ids.update(_feedback_detection_id_candidates(item, index))
    return ids


def _clamp_ratio(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _bbox_to_normalized(
    bbox: Sequence[object] | None,
    image_size: tuple[int, int] | None,
) -> NormalizedBBox | None:
    """Convert [x1, y1, x2, y2] pixels or ratios to normalized bbox."""
    if bbox is None or len(bbox) != 4:
        return None

    try:
        x1, y1, x2, y2 = (float(str(value)) for value in bbox)
    except (TypeError, ValueError):
        return None

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
    """Compare bboxes with tiny tolerance for serialized floats."""
    if left is None or right is None or len(left) != 4 or len(right) != 4:
        return False
    return all(abs(float(a) - float(b)) <= 1e-6 for a, b in zip(left, right))


def _feedback_for_detection(
    detection: FeedbackDetectionItem,
    feedbacks: list[ViolationFeedbackItem],
) -> ViolationFeedbackItem | None:
    """Return newest feedback targeting a detection, when available."""
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
    """Build structured overlay rows for the frontend painter."""
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
    """Convert an ORM object or selected-column row into a response item."""
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
    """Convert a feedback ORM row to a detail response item."""
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
    """Convert a feedback ORM row to its public response model."""
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
    """Convert one review audit row into the public timeline shape."""
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
    """Return the newest non-empty feedback note for a violation."""
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
    """Return review audit rows in newest-first timeline order."""
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
    """Return feedback rows for a violation in newest-first order."""
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
    """Split an optional window-count column from a selected violation row."""
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
    """Return a scalar result value."""
    if hasattr(value, 'name'):
        return value.name
    return value


def _path_candidates_for_db(
    safe_rel_path: Path,
    full_path: Path,
) -> list[str]:
    """Return path variants that may exist in older violation rows."""
    safe_posix = safe_rel_path.as_posix()
    prefixed = (Path(Path(STATIC_DIR).name) / safe_rel_path).as_posix()
    candidates = [safe_posix, prefixed, str(full_path)]
    return list(dict.fromkeys(candidates))


async def _authorize_violation_media_access(
    image_path: str,
    username: str,
    db: AsyncSession,
) -> tuple[Path, str]:
    """Resolve a media path and ensure it belongs to an accessible record."""
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
            Violation.image_path.in_(
                _path_candidates_for_db(safe_rel_path, full_path),
            ),
            Violation.site.in_(site_names),
        )
        .limit(1),
    )
    if result.scalar() is None:
        raise HTTPException(status_code=403, detail='Access denied')

    return full_path, media_type


def _thumbnail_cache_path(full_path: Path) -> Path:
    """Return the deterministic cached thumbnail path for an image."""
    base_dir = Path(STATIC_DIR).resolve()
    rel_path = full_path.relative_to(base_dir)
    return (base_dir / THUMBNAIL_DIR_NAME / rel_path).with_suffix('.jpg')


def _has_recognized_image_header(source_path: Path) -> bool:
    """Recognize image headers before invoking Pillow's lazy opener."""
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
    """Generate or refresh a cached thumbnail on disk."""
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
    """Generate a thumbnail in a worker thread and return its path."""
    thumbnail_path = _thumbnail_cache_path(source_path)
    await asyncio.to_thread(
        _generate_thumbnail_sync,
        source_path,
        thumbnail_path,
    )
    return thumbnail_path


def _image_size_for_violation(image_path: str) -> tuple[int, int] | None:
    """Load image dimensions for a single detail response when available."""
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


def _cursor_payload(row: Any) -> tuple[datetime, int]:
    """Extract cursor ordering values from a selected row or item."""
    if isinstance(row, ViolationItem):
        return row.detection_time, row.id
    if hasattr(row, 'detection_time') and hasattr(row, 'id'):
        return row.detection_time, row.id
    return row[3], int(row[0])


def _encode_violation_cursor(row: Any) -> str:
    """Encode the last row's order key for cursor pagination."""
    detection_time, violation_id = _cursor_payload(row)
    payload = json.dumps(
        {
            'detection_time': detection_time.isoformat(),
            'id': violation_id,
        },
        separators=(',', ':'),
    ).encode('utf-8')
    return base64.urlsafe_b64encode(payload).decode('ascii').rstrip('=')


def _decode_violation_cursor(cursor: str) -> tuple[datetime, int]:
    """Decode a cursor into the list ordering key."""
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
    """Return the canonical empty analytics payload."""
    return ViolationAnalyticsResponse(
        summary=ViolationAnalyticsSummary(total=0, today=0),
        trend=[],
        by_type=[],
        by_site=[],
        by_hour=[],
    )


def _normalise_utc(value: datetime) -> datetime:
    """Treat naive values as UTC and convert aware values to UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _validate_analytics_range(start: datetime, end: datetime) -> tuple[
    datetime,
    datetime,
]:
    """Validate and normalise the analytics query range."""
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
    bind = getattr(db, 'bind', None)
    dialect = getattr(bind, 'dialect', None)
    return str(getattr(dialect, 'name', '') or '')


def _analytics_bucket_expr(
    bucket: AnalyticsBucket,
    db: AsyncSession,
) -> ColumnElement[Any]:
    """Build a dialect-aware UTC bucket expression for detection_time."""
    dialect_name = _analytics_dialect_name(db)
    if dialect_name == 'postgresql':
        return func.date_trunc(bucket, Violation.detection_time)
    if dialect_name in {'mysql', 'mariadb'}:
        formats = {
            'hour': '%Y-%m-%d %H:00:00',
            'day': '%Y-%m-%d',
            'week': '%x-W%v',
        }
        return func.date_format(Violation.detection_time, formats[bucket])
    if dialect_name == 'sqlite':
        formats = {
            'hour': '%Y-%m-%d %H:00:00',
            'day': '%Y-%m-%d',
            'week': '%Y-W%W',
        }
        return func.strftime(formats[bucket], Violation.detection_time)

    if bucket == 'hour':
        return func.date_trunc('hour', Violation.detection_time)
    if bucket == 'week':
        return func.date_trunc('week', Violation.detection_time)
    return func.date(Violation.detection_time)


def _analytics_hour_expr(db: AsyncSession) -> ColumnElement[Any]:
    dialect_name = _analytics_dialect_name(db)
    if dialect_name == 'postgresql':
        return cast(func.extract('hour', Violation.detection_time), Integer)
    if dialect_name in {'mysql', 'mariadb'}:
        return func.hour(Violation.detection_time)
    if dialect_name == 'sqlite':
        return cast(func.strftime('%H', Violation.detection_time), Integer)
    return cast(func.extract('hour', Violation.detection_time), Integer)


def _format_bucket(value: Any, bucket: AnalyticsBucket) -> str:
    if isinstance(value, datetime):
        value = _normalise_utc(value)
        if bucket == 'hour':
            return value.strftime('%Y-%m-%dT%H:00:00Z')
        if bucket == 'week':
            year, week, _ = value.isocalendar()
            return f"{year}-W{week:02d}"
        return value.strftime('%Y-%m-%d')
    return str(value)


def _canonical_violation_type(violation_type: str) -> str:
    """Validate a type code and normalise supported legacy aliases."""
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
    """Filter by a canonical code stored in violation_type_codes."""
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
    """Resolve and authorize a stable camera ID before querying violations."""
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
    """Return the reviewer and sites they may review, or raise 403."""
    user, sites = await load_user_with_effective_sites(username, db)
    if user.role not in {'admin', 'super_admin'}:
        raise HTTPException(
            status_code=403,
            detail='Only admin or super_admin can review violations',
        )
    return user, [site.name for site in sites]


async def require_violation_analytics_access(
    username: str,
    db: AsyncSession,
) -> tuple[User, list[str]]:
    """Return accessible sites when the user may view violation analytics."""
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


@router.get(
    '/my_sites',
    response_model=list[SiteOut],
    summary='Get all accessible sites',
    description='Return a list of sites the user has access to.',
)
async def get_my_sites(
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
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


@router.get(
    '/violations/filter-options',
    response_model=ViolationFilterOptions,
    summary='Get authorized camera and violation type filter options',
)
@router.get(
    '/filter-options',
    response_model=ViolationFilterOptions,
    include_in_schema=False,
)
async def get_violation_filter_options(
    site_id: int = Query(..., gt=0),
    group_id: int | None = Query(None, gt=0),
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
) -> ViolationFilterOptions:
    """Return fixed type codes and cameras visible within one selected site."""
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


@router.get(
    '/violations',
    response_model=ViolationList,
    summary='Get paginated violation records',
    description='Supports filtering by site_id, keyword, and time range.',
)
async def get_violations(
    request: Request,
    site_id: int | None = None,
    stream_id: str | None = None,
    violation_type: str | None = None,
    keyword: str | None = None,
    start_time: datetime | None = Query(None),
    end_time: datetime | None = Query(None),
    limit: int = Query(
        20,
        gt=0,
        le=100,
        description='Records per page (1-100)',
    ),
    offset: int = Query(0, ge=0, description='Starting record offset'),
    flagged: bool | None = Query(
        None,
        description='When true, return only flagged records for reviewers.',
    ),
    review_status: ViolationReviewStatus | None = Query(
        None,
        description='Filter flagged review records by review status.',
    ),
    cursor: str | None = Query(
        None,
        description='Cursor returned by the previous page.',
    ),
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
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

    if flagged is True or review_status is not None:
        _, site_names = await _load_review_scope(username, db)
    else:
        # Retrieve user sites using the cache
        site_names = await get_user_sites_cached(username, db)
    if not site_names:
        return ViolationList(total=0, items=[])

    conditions: list = [Violation.site.in_(site_names)]

    if flagged is True or review_status is not None:
        conditions.append(Violation.is_flagged.is_(True))
    if review_status is not None:
        conditions.append(Violation.review_status == review_status)

    site_name: str | None = None
    if site_id is not None:
        site_name_set = set(site_names)
        site_stmt = select(Site.name).where(Site.id == site_id)
        site_name = _scalar_value((await db.execute(site_stmt)).scalar())
        if not site_name or site_name not in site_name_set:
            raise HTTPException(status_code=403, detail='No access to site_id')
        conditions.append(Violation.site == site_name)

    if stream_id:
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
        conditions.extend(
            [
                Violation.stream_config_id == stream_config_id,
                Violation.site == stream_site_name,
            ],
        )

    if violation_type:
        conditions.append(_type_condition(violation_type, db))

    keyword_text = keyword.strip() if keyword else None
    if keyword_text:
        synonyms: list[str] = search_util.expand_synonyms(keyword_text)
        or_list: list = []
        for syn in dict.fromkeys(synonyms):
            or_list.append(Violation.stream_name.ilike(f"%{syn}%"))
            or_list.append(Violation.warnings_json.ilike(f"%{syn}%"))
        if or_list:
            conditions.append(or_(*or_list))

    if start_time:
        conditions.append(Violation.detection_time >= start_time)
    if end_time:
        conditions.append(Violation.detection_time <= end_time)
    if cursor:
        cursor_time, cursor_id = _decode_violation_cursor(cursor)
        conditions.append(
            or_(
                Violation.detection_time < cursor_time,
                and_(
                    Violation.detection_time == cursor_time,
                    Violation.id < cursor_id,
                ),
            ),
        )

    where_clause = and_(*conditions)

    rows_stmt = (
        select(*_violation_columns, func.count().over().label('total_count'))
        .where(where_clause)
        .order_by(Violation.detection_time.desc(), Violation.id.desc())
        .limit(limit + 1)
    )
    if not cursor:
        rows_stmt = rows_stmt.offset(offset)
    rows_result = await db.execute(rows_stmt)
    rows = rows_result.all()
    has_more = len(rows) > limit
    rows_to_return = rows[:limit]
    total = 0
    items: list[ViolationItem] = []
    next_cursor = None
    for row in rows_to_return:
        item_row, row_total = _split_violation_row_total(row)
        if row_total is not None:
            total = row_total
        items.append(_violation_to_item(item_row, request))

    if has_more and rows_to_return:
        item_row, _ = _split_violation_row_total(rows_to_return[-1])
        next_cursor = _encode_violation_cursor(item_row)

    if not rows and offset and not cursor:
        total_result = await db.execute(
            select(func.count()).select_from(Violation).where(where_clause),
        )
        total = int(total_result.scalar() or 0)

    return ViolationList(total=total, items=items, next_cursor=next_cursor)


@router.get(
    '/violations/analytics',
    response_model=ViolationAnalyticsResponse,
    summary='Get aggregated violation analytics',
    description=(
        'Return aggregates with the same authorized site, camera, violation '
        'type, time-range, and bucket filters applied to every result.'
    ),
)
@router.get(
    '/analytics',
    response_model=ViolationAnalyticsResponse,
    include_in_schema=False,
)
@router.get(
    '/hazard/api/detection/violations/analytics',
    response_model=ViolationAnalyticsResponse,
    include_in_schema=False,
)
async def get_violation_analytics(
    start: datetime = Query(..., description='Inclusive UTC start datetime'),
    end: datetime = Query(..., description='Inclusive UTC end datetime'),
    site_id: int | None = None,
    stream_id: str | None = None,
    violation_type: str | None = None,
    bucket: AnalyticsBucket = Query('day'),
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
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
            bucket=_format_bucket(row[0], bucket),
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


@router.get(
    '/violations/next',
    response_model=ViolationItem | None,
    summary='Get next violation review item',
    description='Return the next flagged record inside reviewer scope.',
)
async def get_next_review_violation(
    request: Request,
    review_status: ViolationReviewStatus = Query('pending'),
    site_id: int | None = None,
    current_id: int | None = Query(
        None,
        description='Optional record id to exclude from the next result.',
    ),
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
) -> ViolationItem | None:
    """Return the next flagged record an admin/super_admin may review."""
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


@router.get(
    '/violations/{violation_id}/audit-log',
    response_model=list[ViolationReviewAuditItem],
    summary='Get violation review audit log',
    description=(
        'Return review history for a flagged record in reviewer scope.'
    ),
)
async def get_violation_review_audit_log(
    violation_id: int,
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
) -> list[ViolationReviewAuditItem]:
    """Return review history for one flagged record inside reviewer scope."""
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


@router.get(
    '/violations/{violation_id}',
    response_model=ViolationItem,
    summary='Get single violation details',
    description='Retrieve a single violation record by its ID.',
)
async def get_single_violation(
    violation_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
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


@router.post(
    '/violations/{violation_id}/feedback',
    response_model=ViolationFeedbackResponse,
    status_code=201,
    summary='Submit feedback for a violation record',
    description='Store structured feedback in pending status for review.',
)
async def submit_violation_feedback(
    violation_id: int,
    payload: ViolationFeedbackCreate,
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
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


@router.patch(
    '/violations/{violation_id}/review',
    response_model=ViolationItem,
    summary='Review a flagged violation record',
    description='Resolve or dismiss a flagged violation with audit logging.',
)
async def review_violation(
    violation_id: int,
    payload: ViolationReviewUpdate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
) -> ViolationItem:
    """Update review state for a flagged violation within reviewer scope."""
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


@router.get(
    '/get_violation_image',
    summary='Get a violation image file',
    description="Retrieve an image file from the 'static' directory.",
)
async def get_violation_image(
    image_path: str,
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
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


@router.get(
    '/get_violation_thumbnail',
    summary='Get a cached violation thumbnail',
    description='Return a small cached JPEG thumbnail for a violation image.',
)
async def get_violation_thumbnail(
    image_path: str,
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
) -> FileResponse:
    """Return an authorized thumbnail, generating it on first request."""
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


@router.post(
    '/upload',
    response_model=UploadViolationResponse,
    summary='Upload a new violation record',
    description='Upload a violation image and associated metadata.',
)
async def upload_violation(
    site: str = Form(...),
    stream_name: str = Form(...),
    detection_time: datetime | None = Form(None),
    warnings_json: str | None = Form(None),
    detections_json: str | None = Form(None),
    cone_polygon_json: str | None = Form(None),
    pole_polygon_json: str | None = Form(None),
    image: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
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
