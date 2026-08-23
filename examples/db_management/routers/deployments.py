from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.database import get_db
from examples.auth.deployment_context import canonical_api_base_url
from examples.auth.models import Deployment
from examples.auth.models import DEPLOYMENT_STATUS_ACTIVE
from examples.auth.models import DEPLOYMENT_STATUS_REVOKED
from examples.auth.models import Tenant
from examples.auth.models import TENANT_STATUS_ACTIVE
from examples.auth.models import TENANT_STATUS_DISABLED
from examples.db_management.deps import require_super_admin
from examples.db_management.schemas.deployment import DeploymentCreate
from examples.db_management.schemas.deployment import DeploymentRead
from examples.db_management.schemas.deployment import DeploymentUpdate
from examples.db_management.schemas.deployment import TenantCreate
from examples.db_management.schemas.deployment import TenantRead
from examples.db_management.schemas.deployment import TenantUpdate

router = APIRouter(prefix='/admin', tags=['tenant-deployment-mgmt'])


def _invalid_status(resource: str, status: str) -> HTTPException:
    """Return a precise validation response for a management status value."""
    return HTTPException(
        status_code=422,
        detail={'code': f'invalid_{resource}_status', 'status': status},
    )


@router.get('/tenants', response_model=list[TenantRead])
async def list_tenants(
    _operator: object = Depends(require_super_admin),
    db: AsyncSession = Depends(get_db),
) -> list[TenantRead]:
    """List tenant records for the privileged deployment administrator."""
    tenants = list((await db.execute(select(Tenant).order_by(Tenant.name))).scalars())
    return [TenantRead.model_validate(tenant) for tenant in tenants]


@router.post('/tenants', response_model=TenantRead, status_code=201)
async def create_tenant(
    payload: TenantCreate,
    _operator: object = Depends(require_super_admin),
    db: AsyncSession = Depends(get_db),
) -> TenantRead:
    """Create a tenant deployment boundary."""
    tenant = Tenant(name=payload.name.strip(), description=payload.description)
    db.add(tenant)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail={'code': 'tenant_name_conflict'},
        ) from exc
    await db.refresh(tenant)
    return TenantRead.model_validate(tenant)


@router.patch('/tenants/{tenant_id}', response_model=TenantRead)
async def update_tenant(
    tenant_id: UUID,
    payload: TenantUpdate,
    _operator: object = Depends(require_super_admin),
    db: AsyncSession = Depends(get_db),
) -> TenantRead:
    """Update a tenant and invalidate deployment sessions on status changes."""
    tenant = await db.get(Tenant, tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail={'code': 'tenant_not_found'})
    if payload.status is not None:
        if payload.status not in {TENANT_STATUS_ACTIVE, TENANT_STATUS_DISABLED}:
            raise _invalid_status('tenant', payload.status)
        if tenant.status != payload.status:
            tenant.status = payload.status
            # A tenant lifecycle transition must also invalidate sessions that
            # would become valid again after a later reactivation.
            deployments = list(
                (
                    await db.execute(
                        select(Deployment).where(Deployment.tenant_id == tenant.id),
                    )
                ).scalars(),
            )
            for deployment in deployments:
                deployment.config_revision += 1
    if payload.name is not None:
        tenant.name = payload.name.strip()
    if payload.description is not None:
        tenant.description = payload.description
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail={'code': 'tenant_name_conflict'},
        ) from exc
    await db.refresh(tenant)
    return TenantRead.model_validate(tenant)


@router.get('/deployments', response_model=list[DeploymentRead])
async def list_deployments(
    _operator: object = Depends(require_super_admin),
    db: AsyncSession = Depends(get_db),
) -> list[DeploymentRead]:
    """List canonical deployment settings published by the Registry."""
    statement = select(Deployment).order_by(Deployment.created_at)
    deployments = list((await db.execute(statement)).scalars())
    return [DeploymentRead.model_validate(deployment) for deployment in deployments]


@router.post('/deployments', response_model=DeploymentRead, status_code=201)
async def create_deployment(
    payload: DeploymentCreate,
    _operator: object = Depends(require_super_admin),
    db: AsyncSession = Depends(get_db),
) -> DeploymentRead:
    """Create one active canonical API deployment at revision one."""
    tenant = await db.get(Tenant, payload.tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail={'code': 'tenant_not_found'})
    try:
        api_base_url = canonical_api_base_url(payload.api_base_url)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={'code': 'invalid_api_base_url'},
        ) from exc
    deployment = Deployment(tenant_id=tenant.id, api_base_url=api_base_url)
    db.add(deployment)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail={'code': 'api_base_url_conflict'},
        ) from exc
    await db.refresh(deployment)
    return DeploymentRead.model_validate(deployment)


@router.patch('/deployments/{deployment_id}', response_model=DeploymentRead)
async def update_deployment(
    deployment_id: UUID,
    payload: DeploymentUpdate,
    _operator: object = Depends(require_super_admin),
    db: AsyncSession = Depends(get_db),
) -> DeploymentRead:
    """Update a deployment and advance its session-invalidating revision."""
    deployment = await db.get(Deployment, deployment_id)
    if deployment is None:
        raise HTTPException(
            status_code=404,
            detail={'code': 'deployment_not_found'},
        )
    if payload.tenant_id is not None:
        tenant = await db.get(Tenant, payload.tenant_id)
        if tenant is None:
            raise HTTPException(status_code=404, detail={'code': 'tenant_not_found'})
        deployment.tenant_id = tenant.id
    if payload.api_base_url is not None:
        try:
            deployment.api_base_url = canonical_api_base_url(payload.api_base_url)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail={'code': 'invalid_api_base_url'},
            ) from exc
    if payload.status is not None:
        if payload.status not in {DEPLOYMENT_STATUS_ACTIVE, DEPLOYMENT_STATUS_REVOKED}:
            raise _invalid_status('deployment', payload.status)
        deployment.status = payload.status
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail={'code': 'api_base_url_conflict'},
        ) from exc
    await db.refresh(deployment)
    return DeploymentRead.model_validate(deployment)
