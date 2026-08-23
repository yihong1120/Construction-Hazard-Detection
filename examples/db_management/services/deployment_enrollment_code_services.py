from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.models import Deployment
from examples.auth.models import DeploymentEnrollmentCode
from examples.auth.models import DeploymentEnrollmentCodeAuditLog
from examples.auth.models import utc_now
from examples.db_management.deps import TenantDeploymentAdministrator
from examples.deployment_registry.enrollments import provision_enrollment_code
from examples.deployment_registry.enrollments import ProvisionedEnrollmentCode

_CANONICAL_UUID = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
)
_STATUS_ACTIVE = 'active'
_STATUS_REDEEMED = 'redeemed'
_STATUS_EXPIRED = 'expired'
_STATUS_REVOKED = 'revoked'


class EnrollmentManagementUnavailable(RuntimeError):
    """Raised when management cannot safely persist an invitation change."""


class EnrollmentManagementConflict(RuntimeError):
    """Raised when the authenticated deployment cannot issue invitations."""


@dataclass(frozen=True, slots=True)
class ManagedEnrollmentCode:
    """Raw code paired with only the metadata permitted in the create reply."""

    id: UUID
    enrollment_code: str
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class ManagedEnrollmentCodeItem:
    """Safe invitation metadata for a list response."""

    id: UUID
    expires_at: datetime
    status: str


def parse_canonical_enrollment_code_id(value: str) -> UUID:
    """Require a lower-case canonical UUID rather than normalising input."""
    if not _CANONICAL_UUID.fullmatch(value):
        raise ValueError('canonical enrollment code id is required')
    return UUID(value)


def enrollment_code_status(
    enrollment: DeploymentEnrollmentCode,
    *,
    now: datetime | None = None,
) -> str:
    """Return a stable public lifecycle status without exposing verifier
    data."""
    current = now or utc_now()
    if enrollment.revoked_at is not None:
        return _STATUS_REVOKED
    if enrollment.redeemed_at is not None:
        return _STATUS_REDEEMED
    if enrollment.expires_at <= current:
        return _STATUS_EXPIRED
    return _STATUS_ACTIVE


async def create_managed_enrollment_code(
    db: AsyncSession,
    *,
    administrator: TenantDeploymentAdministrator,
    expires_in_minutes: int,
    pepper: str,
) -> ManagedEnrollmentCode:
    """Create and audit one tenant-scoped code in one database commit."""
    expires_at = utc_now() + timedelta(minutes=expires_in_minutes)
    try:
        provisioned: ProvisionedEnrollmentCode = (
            await provision_enrollment_code(
                db,
                deployment_id=administrator.deployment_id,
                tenant_id=administrator.tenant_id,
                expires_at=expires_at,
                created_by=administrator.user.username,
                pepper=pepper,
            )
        )
        # Assign the database bigint before referencing it from the audit row.
        await db.flush()
        enrollment = provisioned.enrollment
        db.add(
            DeploymentEnrollmentCodeAuditLog(
                enrollment_code_id=enrollment.id,
                deployment_id=administrator.deployment_id,
                tenant_id=administrator.tenant_id,
                actor_user_id=administrator.user.id,
                action='created',
            ),
        )
        await db.commit()
    except ValueError as exc:
        await db.rollback()
        raise EnrollmentManagementConflict(
            'deployment is unavailable for enrollment',
        ) from exc
    except SQLAlchemyError as exc:
        await db.rollback()
        raise EnrollmentManagementUnavailable(
            'enrollment management storage is unavailable',
        ) from exc
    return ManagedEnrollmentCode(
        id=enrollment.public_id,
        enrollment_code=provisioned.raw_code,
        expires_at=enrollment.expires_at,
    )


async def list_managed_enrollment_codes(
    db: AsyncSession,
    *,
    administrator: TenantDeploymentAdministrator,
) -> list[ManagedEnrollmentCodeItem]:
    """Return only non-secret invitation metadata for one deployment."""
    try:
        enrollments = list(
            (
                await db.execute(
                    select(DeploymentEnrollmentCode)
                    .join(
                        Deployment,
                        Deployment.id
                        == DeploymentEnrollmentCode.deployment_id,
                    )
                    .where(
                        DeploymentEnrollmentCode.deployment_id
                        == administrator.deployment_id,
                        Deployment.tenant_id == administrator.tenant_id,
                    )
                    .order_by(DeploymentEnrollmentCode.created_at.desc()),
                )
            ).scalars(),
        )
    except SQLAlchemyError as exc:
        raise EnrollmentManagementUnavailable(
            'enrollment management storage is unavailable',
        ) from exc
    return [
        ManagedEnrollmentCodeItem(
            id=enrollment.public_id,
            expires_at=enrollment.expires_at,
            status=enrollment_code_status(enrollment),
        )
        for enrollment in enrollments
    ]


async def revoke_managed_enrollment_code(
    db: AsyncSession,
    *,
    administrator: TenantDeploymentAdministrator,
    public_id: UUID,
) -> None:
    """Idempotently revoke one unredeemed code owned by this deployment."""
    try:
        async with db.begin():
            enrollment = await db.scalar(
                select(DeploymentEnrollmentCode)
                .join(
                    Deployment,
                    Deployment.id == DeploymentEnrollmentCode.deployment_id,
                )
                .where(
                    DeploymentEnrollmentCode.public_id == public_id,
                    DeploymentEnrollmentCode.deployment_id
                    == administrator.deployment_id,
                    Deployment.tenant_id == administrator.tenant_id,
                )
                .with_for_update(),
            )
            if enrollment is None:
                return
            if (
                enrollment.revoked_at is None
                and enrollment.redeemed_at is None
            ):
                enrollment.revoked_at = utc_now()
                db.add(
                    DeploymentEnrollmentCodeAuditLog(
                        enrollment_code_id=enrollment.id,
                        deployment_id=administrator.deployment_id,
                        tenant_id=administrator.tenant_id,
                        actor_user_id=administrator.user.id,
                        action='revoked',
                    ),
                )
    except SQLAlchemyError as exc:
        raise EnrollmentManagementUnavailable(
            'enrollment management storage is unavailable',
        ) from exc
