from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime
from sqlalchemy import Index
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import text
from sqlalchemy import Uuid
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column

from examples.auth.database import Base
from examples.auth.db_models.shared import utc_now


class AccountOperation(Base):
    """Durable identity work; credentials never enter the journal."""

    __tablename__ = 'account_operations'
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    phase: Mapped[str] = mapped_column(String(32), nullable=False)
    user_id: Mapped[int | None] = mapped_column(Integer)
    tenant_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    username: Mapped[str] = mapped_column(String(80), nullable=False)
    group_id: Mapped[int | None] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(
        String(32),
        default='user',
        nullable=False,
    )
    subject: Mapped[str | None] = mapped_column(String(255))
    target_revision: Mapped[int | None] = mapped_column(Integer)
    sync_parts: Mapped[dict[str, str]] = mapped_column(
        JSON,
        default=dict,
        server_default=text("'{}'"),
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
    )
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_error: Mapped[str | None] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )
    next_attempt: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )
    __table_args__ = (
        Index(
            'account_operation_due',
            'next_attempt',
            'created_at',
            postgresql_where=text("phase NOT IN ('completed','uncertain')"),
            sqlite_where=text("phase NOT IN ('completed','uncertain')"),
        ),
        Index(
            'account_operation_completed',
            'completed_at',
            postgresql_where=text("phase='completed'"),
        ),
        Index(
            'account_deletion_one_pending',
            'user_id',
            unique=True,
            postgresql_where=text("kind='delete' AND phase<>'completed'"),
            sqlite_where=text("kind='delete' AND phase<>'completed'"),
        ),
    )


class EvidenceWriteOperation(Base):
    __tablename__ = 'evidence_write_operations'
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    path: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )
    attempts: Mapped[int] = mapped_column(
        Integer,
        default=0,
        server_default=text('0'),
        nullable=False,
    )
    next_attempt: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=text('CURRENT_TIMESTAMP'),
        nullable=False,
    )
    last_error: Mapped[str | None] = mapped_column(String(80))
    __table_args__ = (
        Index('evidence_write_due', 'next_attempt', 'created_at'),
    )


class AccountDeletionReceipt(Base):
    __tablename__ = 'account_deletion_receipts'
    user_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=False,
    )
    tenant_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    group_id: Mapped[int | None] = mapped_column(Integer)
    username: Mapped[str] = mapped_column(String(80), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
