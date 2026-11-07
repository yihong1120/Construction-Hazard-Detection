from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean
from sqlalchemy import CheckConstraint
from sqlalchemy import DateTime
from sqlalchemy import event
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Index
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import select
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from examples.auth.database import Base
from examples.auth.db_models.shared import utc_now
from examples.auth.db_models.sites import Site


class Violation(Base):
    """Represents a safety violation detected at a specific site and time.

    Attributes:
        id (int): Primary key.
        stream_name (str): Name of the video stream or camera.
        detection_time (datetime): Timestamp when violation was detected.
        image_path (str): Path to the saved image of the violation.
        detections_json (str | None): JSON string of detected objects.
        cone_polygon_json (str | None): JSON of safety cone polygon data.
        pole_polygon_json (str | None): JSON of safety pole polygon data.
        warnings_json (str | None): JSON of warning content (translated).
        stream_config_id (int | None): Stable camera configuration identifier.
        violation_type_codes (list[str]): Canonical violation type codes.
        created_at (datetime): Time of record creation.
        site (str): Site-name snapshot at the time evidence was recorded.
        site_obj (Site): ORM relationship to the site object.
    """

    __tablename__ = 'violations'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    stream_name: Mapped[str] = mapped_column(String(80), nullable=False)
    detection_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    event_fingerprint: Mapped[str | None] = mapped_column(String(64))
    image_path: Mapped[str] = mapped_column(String(255), nullable=False)

    model_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    model_version: Mapped[str | None] = mapped_column(
        String(120),
        nullable=True,
    )
    model_classes: Mapped[list[dict] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    # Optional JSON fields for detection and warning results
    detections_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    cone_polygon_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    pole_polygon_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    warnings_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    stream_config_id: Mapped[int | None] = mapped_column(
        ForeignKey('stream_configs.id', ondelete='SET NULL'),
        nullable=True,
    )
    violation_type_codes: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )

    is_flagged: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
    flag_reason: Mapped[str | None] = mapped_column(
        String(120),
        nullable=True,
    )
    flagged_by: Mapped[int | None] = mapped_column(
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
    )
    flagged_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    review_status: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        default=None,
    )
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_by: Mapped[int | None] = mapped_column(
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=utc_now,
    )

    event_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    site_id: Mapped[int | None] = mapped_column(
        ForeignKey('sites.id', ondelete='CASCADE'),
        nullable=True,
    )

    # Preserve the historical label even when the site is renamed.
    site: Mapped[str] = mapped_column(
        String(80),
        nullable=False,
    )

    # ORM relationship to the actual Site object
    site_obj: Mapped[Site] = relationship(
        'Site',
        back_populates='violations',
    )

    __table_args__ = (
        Index('uq_violation_event_id', 'event_id', unique=True),
        Index(
            'idx_vio_site_id_detection_id',
            'site_id',
            detection_time.desc(),
            id.desc(),
        ),
        CheckConstraint(
            "review_status IN ('pending', 'resolved', 'dismissed')",
            name='chk_vio_review_status',
        ),
        # Match the list keyset ordering exactly.  The trailing primary key
        # keeps timestamp ties index-only and cursor seeks deterministic.
        Index(
            'idx_vio_site_detection_id',
            'site',
            detection_time.desc(),
            id.desc(),
        ),
        Index(
            'idx_vio_stream_config_detection_id',
            'stream_config_id',
            detection_time.desc(),
            id.desc(),
        ),
        Index('idx_vio_detection_id', detection_time.desc(), id.desc()),
        Index(
            'idx_vio_review_queue',
            'review_status',
            flagged_at.asc().nullslast(),
            detection_time.asc(),
            id.asc(),
            postgresql_where=text('is_flagged IS TRUE'),
        ).ddl_if(dialect='postgresql'),
        Index(
            'idx_vio_site_review_queue',
            'site',
            'review_status',
            flagged_at.asc().nullslast(),
            detection_time.asc(),
            id.asc(),
            postgresql_where=text('is_flagged IS TRUE'),
        ).ddl_if(dialect='postgresql'),
        # pg_trgm is installed by the PostgreSQL baseline/migration before
        # metadata DDL. Keep these extension indexes out of SQLite tests.
        Index(
            'idx_vio_stream_name_trgm',
            'stream_name',
            postgresql_using='gin',
            postgresql_ops={'stream_name': 'gin_trgm_ops'},
        ).ddl_if(dialect='postgresql'),
        Index(
            'idx_vio_warnings_trgm',
            'warnings_json',
            postgresql_using='gin',
            postgresql_ops={'warnings_json': 'gin_trgm_ops'},
        ).ddl_if(dialect='postgresql'),
        Index('idx_vio_flagged_status', 'is_flagged', 'review_status'),
        Index('idx_vio_reviewed_at', 'reviewed_at'),
    )


class ViolationFeedback(Base):
    """Structured user feedback for a stored violation record.

    Feedback is collected in a pending state first so it can be reviewed before
    becoming training data.
    """

    __tablename__ = 'violation_feedback'
    __table_args__ = (
        CheckConstraint(
            "feedback_type IN ('false_positive', 'false_negative', "
            "'wrong_class', 'bad_bbox')",
            name='chk_vf_feedback_type',
        ),
        CheckConstraint(
            "status IN ('pending', 'reviewed', 'accepted', 'rejected')",
            name='chk_vf_status',
        ),
        Index('idx_vf_violation_created', 'violation_id', 'created_at'),
        Index('idx_vf_type_status', 'feedback_type', 'status'),
        Index('idx_vf_user_created', 'user_id', 'created_at'),
        Index('idx_vf_status_created', 'status', 'created_at'),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    violation_id: Mapped[int] = mapped_column(
        ForeignKey('violations.id', ondelete='CASCADE'),
        nullable=False,
    )
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
    )
    anonymous_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    target_detection_id: Mapped[str | None] = mapped_column(
        String(120),
        nullable=True,
    )
    feedback_type: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )
    original_label: Mapped[str | None] = mapped_column(
        String(120),
        nullable=True,
    )
    corrected_label: Mapped[str | None] = mapped_column(
        String(120),
        nullable=True,
    )
    original_bbox: Mapped[list[float] | None] = mapped_column(
        JSON,
        nullable=True,
    )
    corrected_bbox: Mapped[list[float] | None] = mapped_column(
        JSON,
        nullable=True,
    )
    model_version: Mapped[str | None] = mapped_column(
        String(120),
        nullable=True,
    )
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default='pending',
    )
    reviewer_id: Mapped[int | None] = mapped_column(
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )


class ViolationReviewAuditLog(Base):
    """Audit trail for review status changes on violation records."""

    __tablename__ = 'violation_review_audit_logs'
    __table_args__ = (
        CheckConstraint(
            "new_status IN ('pending', 'resolved', 'dismissed')",
            name='chk_vral_new_status',
        ),
        Index('idx_vral_violation_time', 'violation_id', 'reviewed_at'),
        Index('idx_vral_reviewer_time', 'reviewed_by', 'reviewed_at'),
        Index('idx_vral_action_time', 'action', 'reviewed_at'),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    violation_id: Mapped[int] = mapped_column(
        ForeignKey('violations.id', ondelete='CASCADE'),
        nullable=False,
    )
    action: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
        default='review_status_changed',
    )
    old_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    new_status: Mapped[str] = mapped_column(String(20), nullable=False)
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    flagged_reason: Mapped[str | None] = mapped_column(
        String(120),
        nullable=True,
    )
    reviewed_by: Mapped[int | None] = mapped_column(
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
    )
    reviewed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )


@event.listens_for(Violation, 'before_insert')
def _bind_violation_site(
    _mapper: object,
    connection: Connection,
    target: Violation,
) -> None:
    if target.site_id is None:
        target.site_id = connection.scalar(
            select(Site.__table__.c.id).where(
                Site.__table__.c.name == target.site,
            ),
        )
