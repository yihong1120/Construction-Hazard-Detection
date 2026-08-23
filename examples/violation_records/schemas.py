from __future__ import annotations

import math
from datetime import datetime
from typing import Annotated
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from pydantic import RootModel
from pydantic import StrictInt

FeedbackType = Literal[
    'false_positive',
    'false_negative',
    'wrong_class',
    'bad_bbox',
]
FeedbackStatus = Literal['pending', 'reviewed', 'accepted', 'rejected']
ViolationReviewStatus = Literal['pending', 'resolved', 'dismissed']
ViolationDetectionRow = Annotated[list[float], Field(min_length=7)]


class ViolationDetectionRows(RootModel[list[ViolationDetectionRow]]):
    """Validate tracked YOLO rows stored with a violation record.

    Attributes:
        root: Detection rows with at least seven numeric values each.
    """


class ViolationWarning(BaseModel):
    """Represent one structured warning emitted by the danger detector.

    Attributes:
        count: Number of objects or events triggering the warning.
    """

    count: StrictInt


class FeedbackDetectionItem(BaseModel):
    """Represent a detection normalised for feedback selection.

    Attributes:
        id: Stable detection identifier accepted by feedback endpoints.
        label: Optional detected class label.
        confidence: Optional detector confidence.
        bbox: Optional ``[x1, y1, x2, y2]`` bounding box.
    """

    id: str
    label: str | None = None
    confidence: float | None = None
    bbox: list[float] | None = None


class NormalizedBBox(BaseModel):
    """Represent an image-relative bounding box using 0..1 co-ordinates.

    Attributes:
        x: Left edge ratio.
        y: Top edge ratio.
        w: Width ratio.
        h: Height ratio.
    """

    x: float
    y: float
    w: float
    h: float

    @field_validator('x', 'y', 'w', 'h')
    @classmethod
    def validate_ratio(cls, value: float) -> float:
        """Validate an image-relative co-ordinate ratio.

        Args:
            value: Candidate ratio.

        Returns:
            Ratio constrained to the inclusive 0..1 range.

        Raises:
            ValueError: If the ratio falls outside the accepted range.
        """
        if value < 0 or value > 1:
            raise ValueError('bbox ratio must be between 0 and 1')
        return value


class ViolationOverlayObject(BaseModel):
    """Represent one frontend overlay object for a violation image.

    Attributes:
        object_id: Stable detection identifier.
        label: Optional detected class label.
        confidence: Optional detector confidence.
        bbox: Image-relative bounding box.
        is_flagged: Whether feedback flagged this object.
        flag_reason: Optional reason for the flag.
        flag_note: Optional reviewer or feedback note.
    """

    object_id: str
    label: str | None = None
    confidence: float | None = None
    bbox: NormalizedBBox
    is_flagged: bool = False
    flag_reason: str | None = None
    flag_note: str | None = None


class ViolationFeedbackCreate(BaseModel):
    """Define structured feedback submitted against a violation record.

    Attributes:
        type: Classification of feedback supplied by the reviewer.
        anonymous_id: Optional non-identifying client feedback identifier.
        target_detection_id: Optional detection selected by the reviewer.
        original_label: Optional detector label before correction.
        corrected_label: Optional reviewer-supplied replacement label.
        original_bbox: Optional detector bounding box.
        corrected_bbox: Optional reviewer-supplied bounding box.
        model_version: Optional detector model version.
        confidence: Optional detector confidence.
        note: Optional reviewer explanation.
    """

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
        """Validate pixel or normalised ``[x1, y1, x2, y2]`` boxes.

        Args:
            value: Optional candidate bounding box.

        Returns:
            Validated bounding box, or ``None`` when absent.

        Raises:
            ValueError: If the box is incomplete, non-finite, negative, or
                unordered.
        """
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
        """Validate a detector confidence in the inclusive 0..1 range.

        Args:
            value: Optional candidate confidence.

        Returns:
            Validated confidence, or ``None`` when absent.

        Raises:
            ValueError: If the confidence falls outside the accepted range.
        """
        if value is None:
            return None
        if value < 0 or value > 1:
            raise ValueError('confidence must be between 0 and 1')
        return value

    @model_validator(mode='after')
    def validate_type_requirements(self) -> ViolationFeedbackCreate:
        """Validate fields required by the selected feedback type.

        Returns:
            The validated feedback payload.

        Raises:
            ValueError: If the selected feedback type lacks required fields.
        """
        has_original_target = (
            bool(self.target_detection_id) or self.original_bbox is not None
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
    """Represent feedback persisted for later review.

    Attributes:
        id: Feedback database identifier.
        violation_id: Associated violation identifier.
        type: Submitted feedback classification.
        target_detection_id: Optional targeted detection identifier.
        original_label: Optional detector label.
        corrected_label: Optional reviewer correction.
        original_bbox: Optional detector bounding box.
        corrected_bbox: Optional reviewer bounding box.
        model_version: Optional detector model version.
        confidence: Optional detector confidence.
        note: Optional reviewer explanation.
        status: Feedback review status.
        created_at: Time at which feedback was submitted.
    """

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
    """Represent feedback embedded in a violation-detail response.

    Attributes:
        id: Feedback database identifier.
        type: Submitted feedback classification.
        note: Optional reviewer explanation.
        target_detection_id: Optional targeted detection identifier.
        original_label: Optional detector label.
        corrected_label: Optional reviewer correction.
        original_bbox: Optional detector bounding box.
        corrected_bbox: Optional reviewer bounding box.
        model_version: Optional detector model version.
        confidence: Optional detector confidence.
        status: Feedback review status.
        submitted_by: Optional user identifier of the submitter.
        submitted_at: Time at which feedback was submitted.
    """

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
    """Define a review-state update for a flagged violation record.

    Attributes:
        review_status: New flagged-record review status.
        review_note: Optional reviewer explanation.
    """

    review_status: ViolationReviewStatus
    review_note: str | None = Field(default=None, max_length=1000)


class ViolationReviewAuditItem(BaseModel):
    """Represent one immutable review-history audit event.

    Attributes:
        id: Audit database identifier.
        violation_id: Reviewed violation identifier.
        actor_user_id: Optional identifier of the reviewing user.
        action: Stable audit action name.
        old_status: Optional review status before the event.
        new_status: Review status after the event.
        note: Optional reviewer explanation.
        flagged_reason: Optional original flag reason.
        created_at: Time at which the event was recorded.
    """

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
    """Schema for returning information about a single site.

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
    """Represent a camera option scoped to an accessible site.

    Attributes:
        stream_id: Stable stream configuration identifier.
        name: Human-readable camera name.
    """

    stream_id: str
    name: str


class ViolationTypeOption(BaseModel):
    """Represent a canonical type exposed to client filter controls.

    Attributes:
        code: Stable violation-type code.
        label: Localised display label.
    """

    code: str
    label: str


class ViolationFilterOptions(BaseModel):
    """Represent filters available to an authenticated user at one site.

    Attributes:
        cameras: Authorised camera options.
        violation_types: Canonical violation-type options.
    """

    cameras: list[ViolationFilterCamera]
    violation_types: list[ViolationTypeOption]


class ViolationListItem(BaseModel):
    """Represent the compact data required by a violation-list row.

    Attributes:
        id: Violation database identifier.
        site_name: Site where the violation was detected.
        stream_name: Camera that produced the evidence.
        detection_time: Timestamp of the detection.
        thumbnail_url: Protected thumbnail resource URL.
        warning_text: Bounded active-warning summary.
        is_flagged: Whether feedback has flagged the record.
        review_status: Optional status for a flagged record.
        feedback_note: Latest non-empty feedback note.
    """

    id: int
    site_name: str
    stream_name: str
    detection_time: datetime
    thumbnail_url: str
    warning_text: str | None = None
    is_flagged: bool = False
    review_status: ViolationReviewStatus | None = None
    feedback_note: str | None = None


class ViolationItem(BaseModel):
    """Represent full violation evidence, feedback, and review detail.

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
    """Schema for returning a paginated list of violation records.

    Args:
        items: Compact violation rows.
        has_more (bool): Whether another cursor page is available.
    """

    items: list[ViolationListItem]
    next_cursor: str | None = None
    has_more: bool = False


class UploadViolationResponse(BaseModel):
    """Schema for the response after uploading a violation record.

    Args:
        message (str):
            Message indicating the result of the upload.
        violation_id (int):
            Unique identifier for the uploaded violation record.
    """

    message: str
    violation_id: int


class ViolationAnalyticsTopSite(BaseModel):
    """Represent the most frequent site in an analytics result.

    Attributes:
        site_id: Database identifier of the site.
        site_name: Human-readable site name.
        count: Number of matching violations.
    """

    site_id: int
    site_name: str
    count: int


class ViolationAnalyticsTopType(BaseModel):
    """Represent the most frequent type in an analytics result.

    Attributes:
        type: Canonical violation-type code.
        label: Localised display label.
        count: Number of matching violations.
    """

    type: str
    label: str
    count: int


class ViolationAnalyticsSummary(BaseModel):
    """Represent headline counts for a violation analytics result.

    Attributes:
        total: Total matching violations in the requested range.
        today: Matching violations in the current UTC day.
        top_site: Optional most frequent matching site.
        top_type: Optional most frequent matching type.
    """

    total: int
    today: int
    top_site: ViolationAnalyticsTopSite | None = None
    top_type: ViolationAnalyticsTopType | None = None


class ViolationAnalyticsTrendItem(BaseModel):
    """Represent one time-bucket count in an analytics trend.

    Attributes:
        bucket: ISO time bucket label.
        count: Number of matching violations in the bucket.
    """

    bucket: str
    count: int


class ViolationAnalyticsTypeItem(BaseModel):
    """Represent one canonical type count in analytics.

    Attributes:
        type: Canonical violation-type code.
        label: Localised display label.
        count: Number of matching violations.
    """

    type: str
    label: str
    count: int


class ViolationAnalyticsSiteItem(BaseModel):
    """Represent one site count in analytics.

    Attributes:
        site_id: Database identifier of the site.
        site_name: Human-readable site name.
        count: Number of matching violations.
    """

    site_id: int
    site_name: str
    count: int


class ViolationAnalyticsHourItem(BaseModel):
    """Represent one UTC-hour count in analytics.

    Attributes:
        hour: UTC hour from zero to twenty-three.
        count: Number of matching violations.
    """

    hour: int
    count: int


class ViolationAnalyticsResponse(BaseModel):
    """Represent all aggregates returned by a violation analytics query.

    Attributes:
        summary: Headline totals and top values.
        trend: Counts grouped by requested time bucket.
        by_type: Counts grouped by violation type.
        by_site: Counts grouped by site.
        by_hour: Counts grouped by UTC hour.
    """

    summary: ViolationAnalyticsSummary
    trend: list[ViolationAnalyticsTrendItem] = Field(default_factory=list)
    by_type: list[ViolationAnalyticsTypeItem] = Field(default_factory=list)
    by_site: list[ViolationAnalyticsSiteItem] = Field(default_factory=list)
    by_hour: list[ViolationAnalyticsHourItem] = Field(default_factory=list)
