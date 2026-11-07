from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean
from sqlalchemy import CheckConstraint
from sqlalchemy import DateTime
from sqlalchemy import event
from sqlalchemy import ForeignKey
from sqlalchemy import Index
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import text
from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from examples.auth.database import Base
from examples.auth.db_models.notification_schema import (
    install_recipient_versions,
)
from examples.auth.db_models.shared import utc_now

if TYPE_CHECKING:
    from examples.auth.db_models.identity import User
    from examples.auth.db_models.sites import Site


class SiteNotificationPreference(Base):
    """Per-user notification preference for a specific site."""

    __tablename__ = 'site_notification_preferences'

    user_id: Mapped[int] = mapped_column(
        ForeignKey('users.id', ondelete='CASCADE'),
        primary_key=True,
    )
    site_id: Mapped[int] = mapped_column(
        ForeignKey('sites.id', ondelete='CASCADE'),
        primary_key=True,
    )
    is_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

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

    user: Mapped[User] = relationship(
        'User',
        back_populates='notification_preferences',
    )
    site: Mapped[Site] = relationship(
        'Site',
        back_populates='notification_preferences',
    )


class Notification(Base):
    """In-app notification record for notification center history."""

    __tablename__ = 'notifications'
    __table_args__ = (
        UniqueConstraint(
            'user_id',
            'event_key',
            name='uq_notification_user_event',
        ),
        CheckConstraint(
            "type IN ('signature', 'violation', 'document', 'site_alert', "
            "'system')",
            name='chk_notifications_type',
        ),
        Index(
            'idx_notifications_user_created_id',
            'user_id',
            text('created_at DESC'),
            text('id DESC'),
        ),
        Index(
            'idx_notifications_user_read_created_id',
            'user_id',
            'is_read',
            text('created_at DESC'),
            text('id DESC'),
        ),
        Index(
            'idx_notifications_user_type_created_id',
            'user_id',
            'type',
            text('created_at DESC'),
            text('id DESC'),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
    )
    type: Mapped[str] = mapped_column(String(30), nullable=False)
    title: Mapped[str] = mapped_column(String(120), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    deep_link: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_read: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
    metadata_json: Mapped[dict[str, object]] = mapped_column(
        'metadata',
        JSON,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    user: Mapped[User] = relationship(
        'User',
        back_populates='notifications',
    )


class FcmDeviceToken(Base):
    """Push-notification device token stored as encrypted source of truth."""

    __tablename__ = 'fcm_device_tokens'
    __table_args__ = (
        UniqueConstraint(
            'device_token_hash',
            name='uq_fcm_device_tokens_token_hash',
        ),
        CheckConstraint(
            "platform IN ('android', 'ios', 'web', 'unknown')",
            name='chk_fcm_device_tokens_platform',
        ),
        Index(
            'idx_fcm_device_tokens_user_active',
            'user_id',
            'disabled_at',
        ),
        Index(
            'idx_fcm_device_tokens_user_seen',
            'user_id',
            text('last_seen_at DESC'),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
    )
    device_token_encrypted: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    device_token_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    platform: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default='unknown',
    )
    device_lang: Mapped[str] = mapped_column(String(20), nullable=False)
    permission_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default='unknown',
    )
    app_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    web_vapid_key_available: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
    )
    web_service_worker_registered: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    last_success_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    last_failure_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    failure_reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    disabled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
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

    user: Mapped[User] = relationship(
        'User',
        back_populates='fcm_device_tokens',
    )


class NotificationDeliveryReceipt(Base):
    __tablename__ = 'notification_delivery_receipts'
    __table_args__ = (Index('notification_receipts_expiry', 'expires_at'),)
    event_digest: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey(
            'users.id',
            ondelete='CASCADE',
        ),
        primary_key=True,
    )
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    registration_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        primary_key=True,
    )
    delivered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )


event.listen(Base.metadata, 'after_create', install_recipient_versions)
