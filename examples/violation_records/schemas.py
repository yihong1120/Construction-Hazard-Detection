from __future__ import annotations

import math
from datetime import datetime
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator


FeedbackType = Literal[
    'false_positive',
    'false_negative',
    'wrong_class',
    'bad_bbox',
]
FeedbackStatus = Literal['pending', 'reviewed', 'accepted', 'rejected']
ViolationReviewStatus = Literal['pending', 'resolved', 'dismissed']


class FeedbackDetectionItem(BaseModel):
    """Detection data normalised for feedback UI selection."""
    id: str
    label: str | None = None
    confidence: float | None = None
    bbox: list[float] | None = None


class NormalizedBBox(BaseModel):
    """Normalized bounding box using image-relative 0..1 co-ordinates."""
    x: float
    y: float
    w: float
    h: float

    @field_validator('x', 'y', 'w', 'h')
    @classmethod
    def validate_ratio(cls, value: float) -> float:
        """Keep normalized co-ordinates in the 0..1 range."""
        if value < 0 or value > 1:
            raise ValueError('bbox ratio must be between 0 and 1')
        return value


class ViolationOverlayObject(BaseModel):
    """Structured overlay object for frontend painters."""
    object_id: str
    label: str | None = None
    confidence: float | None = None
    bbox: NormalizedBBox
    is_flagged: bool = False
    flag_reason: str | None = None
    flag_note: str | None = None


class ViolationFeedbackCreate(BaseModel):
    """Structured feedback submitted against a stored violation record."""
    type: FeedbackType
    anonymous_id: str | None = Field(default=None, max_length=80)
    target_detection_id: str | None = Field(default=None, max_length=120)
    original_label: str | None = Field(default=None, max_length=120)
    corrected_label: str | None = Field(default=None, max_length=120)
    original_bbox: list[float] | None = None
    corrected_bbox: list[float] | None = None
    model_version: str | None = Field(default=None, max_length=120)
    confidence: float | None = None
    note: str | None = Field(default=None, max_length=1000)

    @field_validator('original_bbox', 'corrected_bbox')
    @classmethod
    def validate_bbox(
        cls,
        value: list[float] | None,
    ) -> list[float] | None:
        """Validate pixel or normalised [x1, y1, x2, y2] boxes."""
        if value is None:
            return None
        if len(value) != 4:
            raise ValueError('bbox must contain exactly 4 values')

        bbox = [float(number) for number in value]
        if any(not math.isfinite(number) for number in bbox):
            raise ValueError('bbox values must be finite numbers')
        if any(number < 0 for number in bbox):
            raise ValueError('bbox values must be non-negative')
        if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
            raise ValueError('bbox must be ordered as [x1, y1, x2, y2]')
        return bbox

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, value: float | None) -> float | None:
        """Keep confidence values in the standard 0..1 range."""
        if value is None:
            return None
        if value < 0 or value > 1:
            raise ValueError('confidence must be between 0 and 1')
        return value

    @model_validator(mode='after')
    def validate_type_requirements(self) -> ViolationFeedbackCreate:
        """Require the fields needed by each feedback type."""
        has_original_target = (
            bool(self.target_detection_id)
            or self.original_bbox is not None
        )
        if self.type == 'false_negative':
            if not self.corrected_label or self.corrected_bbox is None:
                raise ValueError(
                    'false_negative requires corrected_label and '
                    'corrected_bbox',
                )
        if self.type == 'wrong_class':
            if not has_original_target or not self.corrected_label:
                raise ValueError(
                    'wrong_class requires a target and corrected_label',
                )
        if self.type == 'bad_bbox':
            if not has_original_target or self.corrected_bbox is None:
                raise ValueError(
                    'bad_bbox requires a target and corrected_bbox',
                )
        return self


class ViolationFeedbackResponse(BaseModel):
    """Response returned after feedback is stored for review."""
    id: int
    violation_id: int
    type: FeedbackType
    target_detection_id: str | None = None
    original_label: str | None = None
    corrected_label: str | None = None
    original_bbox: list[float] | None = None
    corrected_bbox: list[float] | None = None
    model_version: str | None = None
    confidence: float | None = None
    note: str | None = None
    status: FeedbackStatus
    created_at: datetime


class ViolationFeedbackItem(BaseModel):
    """Feedback summary returned with a violation detail response."""
    id: int
    type: FeedbackType
    note: str | None = None
    target_detection_id: str | None = None
    original_label: str | None = None
    corrected_label: str | None = None
    original_bbox: list[float] | None = None
    corrected_bbox: list[float] | None = None
    model_version: str | None = None
    confidence: float | None = None
    status: FeedbackStatus
    submitted_by: int | None = None
    submitted_at: datetime


class ViolationReviewUpdate(BaseModel):
    """Review state update for a flagged violation record."""
    review_status: ViolationReviewStatus
    review_note: str | None = Field(default=None, max_length=1000)


class ViolationReviewAuditItem(BaseModel):
    """Audit event returned for violation review history."""
    id: int
    violation_id: int
    actor_user_id: int | None = None
    action: str = 'review_status_changed'
    old_status: ViolationReviewStatus | None = None
    new_status: ViolationReviewStatus
    note: str | None = None
    flagged_reason: str | None = None
    created_at: datetime


class SiteOut(BaseModel):
    """
    Schema for returning information about a single site.

    Args:
        id (int): Unique identifier for the site.
        name (str): Name of the site.
        created_at (datetime): Timestamp when the site was created.
        updated_at (datetime): Timestamp when the site was last updated.
    """
    id: int
    name: str
    created_at: datetime
    updated_at: datetime


class ViolationFilterCamera(BaseModel):
    """A camera option scoped to an accessible construction site."""

    stream_id: str
    name: str


class ViolationTypeOption(BaseModel):
    """A canonical violation type exposed to client filter controls."""

    code: str
    label: str


class ViolationFilterOptions(BaseModel):
    """Filter options available to the authenticated user at one site."""

    cameras: list[ViolationFilterCamera]
    violation_types: list[ViolationTypeOption]


class ViolationItem(BaseModel):
    """
    Schema for returning a single Violation record, with details such as
    site_name, stream_name, detection_time, etc.

    Args:
        id (int): Unique identifier for the violation record.
        site_name (str): Name of the site where the violation occurred.
        stream_name (str): Name of the stream associated with the violation.
        detection_time (datetime): Timestamp when the violation was detected.
        image_path (str): Path to the image associated with the violation.
        created_at (datetime): Timestamp when the violation record was created.
        detection_items (str | None): Details of detected items, if any.
        warnings (str | None): Any warnings associated with the violation.
        cone_polygons (str | None): Polygons representing cones, if applicable.
        pole_polygons (str | None): Polygons representing poles, if applicable.
    """
    id: int
    site_name: str
    stream_name: str
    detection_time: datetime
    detected_at: datetime | None = None
    image_path: str
    image_url: str | None = None
    thumbnail_url: str | None = None
    created_at: datetime
    detection_items: str | None = None
    warnings: str | None = None
    warning_text: str | None = None
    cone_polygons: str | None = None
    pole_polygons: str | None = None
    detections: list[FeedbackDetectionItem] | None = None
    feedback_detections: list[FeedbackDetectionItem] | None = None
    overlay_objects: list[ViolationOverlayObject] | None = None
    feedbacks: list[ViolationFeedbackItem] | None = None
    is_flagged: bool = False
    flag_reason: str | None = None
    flagged_by: int | None = None
    flagged_at: datetime | None = None
    review_status: ViolationReviewStatus | None = None
    review_note: str | None = None
    reviewed_by: int | None = None
    reviewed_at: datetime | None = None
    feedback_note: str | None = None
    review_audit_logs: list[ViolationReviewAuditItem] | None = None


class ViolationList(BaseModel):
    """
    Schema for returning a paginated list of violation records.

    Args:
        total (int): Total number of violation records available.
        items (list[ViolationItem]): List of violation records.
    """
    total: int
    items: list[ViolationItem]
    next_cursor: str | None = None


class UploadViolationResponse(BaseModel):
    """
    Schema for the response after uploading a violation record.

    Args:
        message (str):
            Message indicating the result of the upload.
        violation_id (int):
            Unique identifier for the uploaded violation record.
    """
    message: str
    violation_id: int


class ViolationAnalyticsTopSite(BaseModel):
    site_id: int
    site_name: str
    count: int


class ViolationAnalyticsTopType(BaseModel):
    type: str
    label: str
    count: int


class ViolationAnalyticsSummary(BaseModel):
    total: int
    today: int
    top_site: ViolationAnalyticsTopSite | None = None
    top_type: ViolationAnalyticsTopType | None = None


class ViolationAnalyticsTrendItem(BaseModel):
    bucket: str
    count: int


class ViolationAnalyticsTypeItem(BaseModel):
    type: str
    label: str
    count: int


class ViolationAnalyticsSiteItem(BaseModel):
    site_id: int
    site_name: str
    count: int


class ViolationAnalyticsHourItem(BaseModel):
    hour: int
    count: int


class ViolationAnalyticsResponse(BaseModel):
    summary: ViolationAnalyticsSummary
    trend: list[ViolationAnalyticsTrendItem]
    by_type: list[ViolationAnalyticsTypeItem]
    by_site: list[ViolationAnalyticsSiteItem]
    by_hour: list[ViolationAnalyticsHourItem]
