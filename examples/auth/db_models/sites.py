from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger
from sqlalchemy import Boolean
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import Index
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import text
from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from examples.auth.database import Base
from examples.auth.db_models.associations import site_groups_table
from examples.auth.db_models.associations import user_sites_table
from examples.auth.db_models.shared import utc_now

if TYPE_CHECKING:
    from examples.auth.db_models.identity import Group
    from examples.auth.db_models.identity import User
    from examples.auth.db_models.notifications import (
        SiteNotificationPreference,
    )
    from examples.auth.db_models.violations import Violation


class Site(Base):
    """Represents a construction site, including its name and associated users
    and safety violations.

    Attributes:
        id (int): Primary key.
        name (str): Name of the site.
        created_at (datetime): Creation timestamp.
        updated_at (datetime): Last update timestamp.
        users (list[User]): Users assigned to this site.
        violations (list[Violation]): Safety violations detected at the site.
    """

    __tablename__ = 'sites'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    recipient_revision: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default=text('0'),
    )
    name: Mapped[str] = mapped_column(
        String(80),
        unique=True,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    # Many-to-many relationship to Group
    groups: Mapped[list[Group]] = relationship(
        'Group',
        secondary=site_groups_table,
        back_populates='sites',
        lazy='joined',
    )

    # Many-to-many relationship to User
    # This is an association table linking users to sites
    users: Mapped[list[User]] = relationship(
        'User',
        secondary=user_sites_table,
        back_populates='sites',
    )

    notification_preferences: Mapped[list[SiteNotificationPreference]] = (
        relationship(
            'SiteNotificationPreference',
            back_populates='site',
            cascade='all, delete-orphan',
        )
    )

    # One-to-many relationship to Violation
    # This is a foreign key to the violations table
    violations: Mapped[list[Violation]] = relationship(
        'Violation',
        back_populates='site_obj',
        passive_deletes='all',
    )

    stream_configs: Mapped[list[StreamConfig]] = relationship(
        'StreamConfig',
        back_populates='site',
        cascade='all, delete-orphan',
    )


class StreamConfig(Base):
    """Represents the configuration for a video stream, including detection
    capabilities and scheduling.

    Attributes:
        id (int): Primary key.
        group_id (int): Foreign key to the group_info table.
        site_id (int): Foreign key to the sites table.
        stream_name (str): Name of the video stream.
        video_url (str): URL of the video stream.
        model_key (str): Key for the detection model.
        detect_no_safety_vest_or_helmet (bool): Detection capability.
        detect_near_machinery_or_vehicle (bool): Detection capability.
        detect_in_restricted_area (bool): Detection capability.
        detect_in_utility_pole_restricted_area (bool): Detection capability.
        detect_machinery_close_to_pole (bool): Detection capability.
        recognition_enabled (bool): Whether recognition processing is enabled.
        expire_date (datetime | None): Expiration date for the configuration.
        work_start_hour (int): Start hour for work scheduling.
        work_end_hour (int): End hour for work scheduling.
        created_at (datetime): Timestamp of creation.
        updated_at (datetime): Timestamp of last update.
    """

    __tablename__ = 'stream_configs'

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    group_id: Mapped[int] = mapped_column(
        ForeignKey('group_info.id', ondelete='CASCADE'),
        nullable=False,
    )
    site_id: Mapped[int] = mapped_column(
        ForeignKey('sites.id', ondelete='CASCADE'),
        nullable=False,
    )

    stream_name: Mapped[str] = mapped_column(String(80), nullable=False)
    video_url: Mapped[str] = mapped_column(String(255), nullable=False)
    model_key: Mapped[str] = mapped_column(String(80), nullable=False)

    # Detection capabilities
    # These fields are used to determine the types of violations
    detect_no_safety_vest_or_helmet: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )
    detect_near_machinery_or_vehicle: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )
    detect_in_restricted_area: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )
    detect_in_utility_pole_restricted_area: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )
    detect_machinery_close_to_pole: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )

    recognition_enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        server_default=text('TRUE'),
        nullable=False,
    )
    expire_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    work_start_hour: Mapped[int] = mapped_column(Integer)
    work_end_hour: Mapped[int] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text('CURRENT_TIMESTAMP'),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    # Foreign key to the group_info table
    group: Mapped[Group] = relationship(
        'Group',
        back_populates='stream_configs',
    )
    site: Mapped[Site] = relationship(
        'Site',
        back_populates='stream_configs',
    )

    __table_args__ = (
        # Ensure that the combination of site_id and stream_name is unique
        # across the table to prevent duplicate stream configurations
        UniqueConstraint('site_id', 'stream_name', name='uq_sc_site_stream'),
    )


class SiteMediaCleanupJob(Base):
    """Durable post-commit cleanup task for site-owned evidence files."""

    __tablename__ = 'site_media_cleanup_jobs'
    __table_args__ = (
        Index(
            'idx_site_media_cleanup_jobs_pending',
            'completed_at',
            'lease_expires_at',
            'id',
            postgresql_where=text('completed_at IS NULL'),
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )
    path: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
        unique=True,
    )
    attempt_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    lease_token: Mapped[str | None] = mapped_column(String(36), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
