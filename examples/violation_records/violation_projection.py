from __future__ import annotations

from datetime import datetime

from sqlalchemy import select

from examples.auth.models import Violation
from examples.auth.models import ViolationFeedback
from examples.violation_records.schemas import ViolationReviewStatus


latest_feedback_note = (
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


violation_list_columns = (
    Violation.id.label('id'),
    Violation.site.label('site'),
    Violation.stream_name.label('stream_name'),
    Violation.detection_time.label('detection_time'),
    Violation.image_path.label('image_path'),
    Violation.warnings_json.label('warnings_json'),
    Violation.is_flagged.label('is_flagged'),
    Violation.review_status.label('review_status'),
    latest_feedback_note,
    Violation.model_version,
    Violation.model_classes,
    Violation.model_id,
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
    str | None,
    list[dict] | None,
    str | None,
]
