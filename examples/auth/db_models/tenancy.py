from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID
from uuid import uuid4

from sqlalchemy import BigInteger
from sqlalchemy import CheckConstraint
from sqlalchemy import DateTime
from sqlalchemy import event
from sqlalchemy import ForeignKey
from sqlalchemy import Index
from sqlalchemy import inspect
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import text
from sqlalchemy import UniqueConstraint
from sqlalchemy import Uuid
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from examples.auth.database import Base
from examples.auth.db_models.shared import DEPLOYMENT_STATUS_ACTIVE
from examples.auth.db_models.shared import TENANT_STATUS_ACTIVE
from examples.auth.db_models.shared import utc_now
lazy from examples.auth.deployment_context import canonical_api_base_url

if TYPE_CHECKING:
    from examples.auth.db_models.identity import User


class Tenant(Base):
    """Formal tenant boundary used by every authenticated account."""

    __tablename__ = 'tenants'
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'disabled')",
            name='chk_tenants_status',
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    name: Mapped[str] = mapped_column(String(160), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=TENANT_STATUS_ACTIVE,
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
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

    deployments: Mapped[list[Deployment]] = relationship(
        'Deployment',
        back_populates='tenant',
        cascade='all, delete-orphan',
    )
    users: Mapped[list[User]] = relationship('User', back_populates='tenant')


class Deployment(Base):
    """One canonical API deployment published through the signed Registry."""

    __tablename__ = 'deployments'
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'revoked')",
            name='chk_deployments_status',
        ),
        CheckConstraint(
            'config_revision >= 1',
            name='chk_deployments_config_revision',
        ),
        UniqueConstraint('api_base_url', name='uq_deployments_api_base_url'),
        Index('idx_deployments_tenant', 'tenant_id'),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    tenant_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey('tenants.id', ondelete='RESTRICT'),
        nullable=False,
    )
    api_base_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    config_revision: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=DEPLOYMENT_STATUS_ACTIVE,
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

    tenant: Mapped[Tenant] = relationship(
        'Tenant',
        back_populates='deployments',
        lazy='joined',
    )


class DeploymentEnrollmentCode(Base):
    """One-time native-device enrollment verifier for a deployment.

    The raw enrollment code is deliberately never represented by this model.
    Only its HMAC-SHA256 verifier is persisted, so a database backup cannot be
    used as a list of usable company activation codes.
    """

    __tablename__ = 'deployment_enrollment_codes'
    __table_args__ = (
        CheckConstraint(
            'expires_at > created_at',
            name='chk_deployment_enrollment_codes_expiry',
        ),
        UniqueConstraint(
            'code_verifier_hash',
            name='uq_deployment_enrollment_codes_verifier',
        ),
        Index(
            'uq_deployment_enrollment_codes_public_id',
            'public_id',
            unique=True,
        ),
        Index(
            'idx_deployment_enrollment_codes_deployment',
            'deployment_id',
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    # The internal bigint key remains private.  Management clients address an
    # invitation by this opaque UUID so database row counts are not exposed.
    public_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        nullable=False,
        default=uuid4,
    )
    deployment_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey('deployments.id', ondelete='RESTRICT'),
        nullable=False,
    )
    code_verifier_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    redeemed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    created_by: Mapped[str] = mapped_column(String(160), nullable=False)


class DeploymentEnrollmentCodeAuditLog(Base):
    """Immutable, non-secret audit event for an enrollment invitation."""

    __tablename__ = 'deployment_enrollment_code_audit_logs'
    __table_args__ = (
        CheckConstraint(
            "action IN ('created', 'revoked')",
            name='chk_deployment_enrollment_code_audit_action',
        ),
        Index(
            'idx_deployment_enrollment_code_audit_code',
            'enrollment_code_id',
            text('created_at DESC'),
        ),
        Index(
            'idx_deployment_enrollment_code_audit_deployment',
            'deployment_id',
            text('created_at DESC'),
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    enrollment_code_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey('deployment_enrollment_codes.id', ondelete='RESTRICT'),
        nullable=False,
    )
    deployment_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey('deployments.id', ondelete='RESTRICT'),
        nullable=False,
    )
    tenant_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey('tenants.id', ondelete='RESTRICT'),
        nullable=False,
    )
    actor_user_id: Mapped[int | None] = mapped_column(
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
    )
    action: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )


@event.listens_for(Deployment, 'before_insert')
def _canonicalise_new_deployment(
    _mapper: object,
    _connection: object,
    target: Deployment,
) -> None:
    """Reject non-canonical deployment origins before they reach storage."""

    target.api_base_url = canonical_api_base_url(target.api_base_url)
    if target.config_revision < 1:
        raise ValueError('config_revision must start at 1')


@event.listens_for(Deployment, 'before_update')
def _revision_on_deployment_change(
    _mapper: object,
    _connection: object,
    target: Deployment,
) -> None:
    """Invalidate tokens whenever the deployment security contract changes."""

    target.api_base_url = canonical_api_base_url(target.api_base_url)
    state = inspect(target)
    changed = any(
        state.attrs[name].history.has_changes()
        for name in ('tenant_id', 'api_base_url', 'status')
    )
    if changed and not state.attrs.config_revision.history.has_changes():
        target.config_revision += 1
