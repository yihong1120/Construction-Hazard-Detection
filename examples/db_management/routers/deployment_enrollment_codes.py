from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.config import Settings
from examples.auth.database import get_db
from examples.auth.models import User
from examples.db_management.deps import get_current_user
from examples.db_management.deps import require_admin
from examples.db_management.deps import require_tenant_deployment_administrator
from examples.db_management.deps import TenantDeploymentAdministrator
from examples.db_management.schemas.deployment_enrollment_code import (
    DeploymentEnrollmentCodeCreate,
)
from examples.db_management.schemas.deployment_enrollment_code import (
    DeploymentEnrollmentCodeCreated,
)
from examples.db_management.schemas.deployment_enrollment_code import (
    DeploymentEnrollmentCodeItem,
)
from examples.db_management.schemas.deployment_enrollment_code import (
    DeploymentEnrollmentCodeList,
)
from examples.db_management.schemas.resource_options import Option
from examples.db_management.schemas.resource_options import resource_options
from examples.db_management.schemas.resource_options import ResourceOptions
from examples.db_management.services import (
    deployment_enrollment_code_services as code_services,
)
from examples.db_management.services.invitation_policy import invitation_policy

router = APIRouter(tags=['deployment-enrollment-codes'])


@router.get(
    '/deployment-enrollment-codes/expiry-options',
    response_model=ResourceOptions,
)
async def get_invitation_expiry(
    request: Request,
    response: Response,
    me: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ResourceOptions:
    await require_tenant_deployment_administrator(request, me, db)
    require_admin(me)
    invitation = invitation_policy(str(me.tenant_id))
    options = [Option(id=str(n), label=f'{n} min') for n in invitation.minutes]
    default = (
        str(invitation.default_minutes)
        if invitation.default_minutes is not None
        else None
    )
    policy = {'max_minutes': invitation.max_minutes}
    response.headers['Cache-Control'] = 'private, no-store'
    return resource_options(me, None, options, default, policy)


settings = Settings()


def _unavailable() -> HTTPException:
    """Return a safe, non-secret outage response."""
    return HTTPException(
        status_code=503,
        detail={'code': 'enrollment_management_unavailable'},
    )


@router.post(
    '/deployment-enrollment-codes',
    response_model=DeploymentEnrollmentCodeCreated,
)
async def create_deployment_enrollment_code(
    payload: DeploymentEnrollmentCodeCreate,
    response: Response,
    administrator: TenantDeploymentAdministrator = Depends(
        require_tenant_deployment_administrator,
    ),
    db: AsyncSession = Depends(get_db),
) -> DeploymentEnrollmentCodeCreated:
    """Issue one raw code once for the deployment bound to this login."""
    try:
        created = await code_services.create_managed_enrollment_code(
            db,
            administrator=administrator,
            expires_in_minutes=payload.expires_in_minutes,
            pepper=settings.deployment_enrollment_code_pepper,
        )
    except code_services.EnrollmentManagementConflict as exc:
        raise HTTPException(
            status_code=409,
            detail={'code': 'deployment_configuration_changed'},
        ) from exc
    except (code_services.EnrollmentManagementUnavailable, ValueError) as exc:
        raise _unavailable() from exc
    response.headers['Cache-Control'] = 'no-store'
    return DeploymentEnrollmentCodeCreated(
        id=created.id,
        enrollment_code=created.enrollment_code,
        expires_at=created.expires_at,
    )


@router.get(
    '/deployment-enrollment-codes',
    response_model=DeploymentEnrollmentCodeList,
)
async def list_deployment_enrollment_codes(
    response: Response,
    administrator: TenantDeploymentAdministrator = Depends(
        require_tenant_deployment_administrator,
    ),
    db: AsyncSession = Depends(get_db),
) -> DeploymentEnrollmentCodeList:
    """List current deployment invitations without raw code or verifier
    data."""
    try:
        items = await code_services.list_managed_enrollment_codes(
            db,
            administrator=administrator,
        )
    except code_services.EnrollmentManagementUnavailable as exc:
        raise _unavailable() from exc
    response.headers['Cache-Control'] = 'no-store'
    return DeploymentEnrollmentCodeList(
        items=[
            DeploymentEnrollmentCodeItem(
                id=item.id,
                expires_at=item.expires_at,
                status=item.status,
            )
            for item in items
        ],
    )


@router.delete('/deployment-enrollment-codes/{code_id}', status_code=204)
async def delete_deployment_enrollment_code(
    code_id: str,
    response: Response,
    administrator: TenantDeploymentAdministrator = Depends(
        require_tenant_deployment_administrator,
    ),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Idempotently revoke an invitation owned by this deployment."""
    try:
        public_id = code_services.parse_canonical_enrollment_code_id(code_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={'code': 'canonical_enrollment_code_id_required'},
        ) from exc
    try:
        await code_services.revoke_managed_enrollment_code(
            db,
            administrator=administrator,
            public_id=public_id,
        )
    except code_services.EnrollmentManagementUnavailable as exc:
        raise _unavailable() from exc
    response.headers['Cache-Control'] = 'no-store'
