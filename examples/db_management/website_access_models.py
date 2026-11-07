from __future__ import annotations

from datetime import datetime
from uuid import UUID
from uuid import uuid4

from sqlalchemy import Boolean
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import String
from sqlalchemy import UniqueConstraint
from sqlalchemy import Uuid
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from examples.auth.database import Base
from examples.auth.models import utc_now


class WebsiteAccessPolicy(Base):
    """One independently versioned policy per configured public website."""

    __tablename__ = 'website_access_policies'
    site_key: Mapped[str] = mapped_column(String(32), primary_key=True)
    restricted: Mapped[bool] = mapped_column(Boolean, default=False)
    version: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now,
    )
    rules: Mapped[list[WebsiteAccessRule]] = relationship(
        cascade='all, delete-orphan',
        lazy='selectin',
        order_by='WebsiteAccessRule.network',
    )


class WebsiteAccessRule(Base):
    """A labelled IP address or CIDR belonging to one website allowlist."""

    __tablename__ = 'website_access_rules'
    __table_args__ = (
        UniqueConstraint(
            'site_key', 'network', name='uq_website_access_network',
        ),
    )
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    site_key: Mapped[str] = mapped_column(
        ForeignKey('website_access_policies.site_key', ondelete='CASCADE'),
    )
    network: Mapped[str] = mapped_column(String(43))
    label: Mapped[str] = mapped_column(String(120))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now,
    )


class WebsiteAccessAudit(Base):
    """Keep a before/after record of every committed management operation."""

    __tablename__ = 'website_access_audit_logs'
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    site_key: Mapped[str] = mapped_column(
        ForeignKey('website_access_policies.site_key'),
    )
    actor_user_id: Mapped[int | None] = mapped_column(
        ForeignKey('users.id', ondelete='SET NULL'),
    )
    actor_ip: Mapped[str] = mapped_column(String(45))
    action: Mapped[str] = mapped_column(String(32))
    before: Mapped[dict] = mapped_column(JSON)
    after: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True,
    )
