from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast as type_cast
from urllib.parse import urlencode

from fastapi import Request
from pydantic import ValidationError

from examples.auth.models import Violation
from examples.auth.models import ViolationFeedback
from examples.auth.models import ViolationReviewAuditLog
from examples.shared.class_colors import normalize_color
from examples.shared.class_colors import overlay_style
from examples.shared.overlay_geometry import clip_pixel_bbox
from examples.shared.overlay_visibility import object_visibility
from examples.violation_records.record_policy import historical_classes
from examples.violation_records.schemas import FeedbackDetectionItem
from examples.violation_records.schemas import FeedbackStatus
from examples.violation_records.schemas import FeedbackType
from examples.violation_records.schemas import NormalizedBBox
from examples.violation_records.schemas import ViolationDetectionRows
from examples.violation_records.schemas import ViolationFeedbackItem
from examples.violation_records.schemas import ViolationFeedbackResponse
from examples.violation_records.schemas import ViolationItem
from examples.violation_records.schemas import ViolationListItem
from examples.violation_records.schemas import ViolationOverlayObject
from examples.violation_records.schemas import ViolationReviewAuditItem
from examples.violation_records.schemas import ViolationReviewStatus
from examples.violation_records.violation_projection import ViolationListRow
from examples.violation_records.violation_types import WARNING_PAYLOAD_ADAPTER


@dataclass(frozen=True)
class HistoricalDetection:
    index: int
    values: list[float]

    @property
    def object_id(self) -> str:
        return f'det_{self.index}'


def decode_indexed_detections(
    value: str | None,
) -> list[HistoricalDetection] | None:
    if value is None:
        return None
    try:
        rows = ViolationDetectionRows.model_validate_json(value).root
        return [
            HistoricalDetection(index, row) for index, row in enumerate(rows)
        ]
    except ValidationError:
        try:
            raw = json.loads(value)
        except (ValueError, TypeError):
            return None
        if not isinstance(raw, list):
            return None
        valid = []
        for index, row in enumerate(raw):
            try:
                parsed = ViolationDetectionRows.model_validate([row]).root[0]
            except ValidationError:
                continue
            valid.append(HistoricalDetection(index, parsed))
        return valid or None


def decode_detection_items(value: str | None) -> list[list[float]] | None:
    rows = decode_indexed_detections(value)
    return [row.values for row in rows] if rows is not None else None


def warning_text_from_json(value: str | None) -> str | None:
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
        f'{key}: {warning.count}'
        for key, warning in payload.items()
        if warning.count > 0
    ]
    return ', '.join(parts)[:200] if parts else None


def media_endpoint_url(
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
    return f'{request.url_for(endpoint_name)}?{query}'


def feedback_detection_from_item(
    detection: HistoricalDetection,
    class_codes: dict[int, str] | None = None,
) -> FeedbackDetectionItem:
    """Retain the persisted row index for frontend feedback selection."""
    item = detection.values
    return FeedbackDetectionItem(
        id=detection.object_id,
        label=(class_codes or {}).get(int(item[5]), f'class-{int(item[5])}'),
        confidence=item[4],
        bbox=item[:4],
    )


def feedback_detection_ids_from_json(value: str | None) -> set[str] | None:
    """Return target-detection IDs accepted for a stored record.

    Args:
        value: Optional persisted detection JSON.

    Returns:
        Accepted target identifiers, or ``None`` when detections are absent.
    """
    items = decode_indexed_detections(value)
    if items is None:
        return None
    ids = {item.object_id for item in items}
    ids.update(
        str(int(item.values[6])) for item in items if int(item.values[6]) != -1
    )
    return ids


def clamp_ratio(value: float) -> float:
    """Clamp a coordinate ratio to the inclusive 0..1 range.

    Args:
        value: Candidate ratio.

    Returns:
        Ratio constrained to the valid image-relative range.
    """
    return min(max(value, 0.0), 1.0)


def bbox_to_normalized(
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

    normalized = all(0 <= value <= 1 for value in bbox)
    size = (1, 1) if normalized else image_size
    if size is None:
        return None
    clipped = clip_pixel_bbox(bbox, size)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    width, height = size
    return NormalizedBBox(
        x=x1 / width,
        y=y1 / height,
        w=(x2 - x1) / width,
        h=(y2 - y1) / height,
    )


def bbox_nearly_equal(
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


def overlay_objects_from_feedback(
    detections: list[FeedbackDetectionItem] | None,
    feedbacks: list[ViolationFeedbackItem],
    image_size: tuple[int, int] | None,
    class_metadata: list[dict] | None = None,
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
        bbox = bbox_to_normalized(detection.bbox, image_size)
        if bbox is None:
            continue
        matching_feedback = feedback_by_detection_id.get(detection.id)
        if matching_feedback is None:
            matching_feedback = next(
                (
                    candidate
                    for candidate in feedbacks_without_target
                    if bbox_nearly_equal(
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
        bbox = bbox_to_normalized(feedback.corrected_bbox, image_size)
        if bbox is None:
            continue
        overlay_objects.append(
            ViolationOverlayObject(
                object_id=f'feedback_{feedback.id}',
                label=feedback.corrected_label,
                confidence=None,
                bbox=bbox,
                is_flagged=True,
                flag_reason=feedback.type,
                flag_note=feedback.note,
            ),
        )

    style = overlay_style()
    colors = dict(style['class_colors'])
    names = {}
    codes = {}
    for item in class_metadata or []:
        color = normalize_color(item.get('color'))
        if color:
            for key in (item['code'], str(item['id']), str(float(item['id']))):
                colors[key] = color
                names[key] = item.get('display_name', item['code'])
                codes[key] = item['code']
    for overlay in overlay_objects:
        overlay.code = codes.get(overlay.label)
        overlay.display_name = names.get(overlay.label, overlay.label)
        overlay.color = colors.get(overlay.label, style['unknown_class_color'])
        for field, value in object_visibility(overlay.code).items():
            setattr(overlay, field, value)
    return overlay_objects


def violation_to_detail_item(
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
    snapshot = historical_classes(violation)
    class_codes = {c.id: c.code for c in snapshot}
    rows = decode_indexed_detections(violation.detections_json)
    detections = (
        [feedback_detection_from_item(item, class_codes) for item in rows]
        if rows is not None
        else None
    )
    return ViolationItem(
        model_id=getattr(violation, 'model_id', None),
        model_version=getattr(violation, 'model_version', None),
        model_classes=getattr(violation, 'model_classes', None),
        class_metadata=[c.model_dump(exclude_none=True) for c in snapshot],
        overlay_style=overlay_style([c.model_dump() for c in snapshot]),
        id=violation.id,
        site_name=violation.site,
        stream_name=violation.stream_name,
        detection_time=violation.detection_time.astimezone(),
        image_path=violation.image_path,
        image_url=media_endpoint_url(
            'get_violation_image',
            violation.image_path,
            request,
        ),
        thumbnail_url=media_endpoint_url(
            'get_violation_thumbnail',
            violation.image_path,
            request,
        ),
        created_at=violation.created_at.astimezone(),
        detection_items=violation.detections_json,
        warnings=violation.warnings_json,
        warning_text=warning_text_from_json(violation.warnings_json),
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


def violation_to_list_item(
    row: ViolationListRow,
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
    ) = row[:9]
    version, classes = row[9:11] if len(row) > 9 else (None, None)
    model_id = row[11] if len(row) > 11 else None
    snapshot = historical_classes(
        SimpleNamespace(
            model_classes=classes, model_version=version, model_id=model_id,
        ),
    )
    # The shipped web/native list localises structured warning keys. Keep
    # positive counts alongside the text summary without loading full detail.
    warnings = (
        WARNING_PAYLOAD_ADAPTER.validate_json(warnings_json)
        if warnings_json is not None else {}
    )
    return ViolationListItem(
        model_version=version,
        class_metadata=[c.model_dump(exclude_none=True) for c in snapshot],
        overlay_style=overlay_style([c.model_dump() for c in snapshot]),
        id=violation_id,
        site_name=site,
        stream_name=stream_name,
        detection_time=detection_time.astimezone(),
        thumbnail_url=media_endpoint_url(
            'get_violation_thumbnail',
            image_path,
            request,
        ),
        warnings={
            key: warning for key, warning in warnings.items()
            if warning.count > 0
        },
        warning_text=warning_text_from_json(warnings_json),
        is_flagged=is_flagged,
        review_status=review_status if is_flagged else None,
        feedback_note=feedback_note,
    )


def feedback_to_item(feedback: ViolationFeedback) -> ViolationFeedbackItem:
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
        confidence=feedback.confidence,
        status=type_cast(FeedbackStatus, feedback.status),
        submitted_by=feedback.user_id,
        submitted_at=feedback.created_at.astimezone(),
    )


def feedback_to_response(
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
        confidence=feedback.confidence,
        note=feedback.note,
        status=type_cast(FeedbackStatus, feedback.status),
        created_at=feedback.created_at.astimezone(),
    )


def review_audit_to_item(
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
            ViolationReviewStatus | None,
            audit_log.old_status,
        ),
        new_status=type_cast(ViolationReviewStatus, audit_log.new_status),
        note=audit_log.review_note,
        flagged_reason=audit_log.flagged_reason,
        created_at=audit_log.reviewed_at.astimezone(),
    )
