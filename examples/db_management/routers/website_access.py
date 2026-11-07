from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query
from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.database import get_db
from examples.auth.models import User
from examples.db_management.deps import require_super_admin
from examples.db_management.schemas.website_access import AuditRead
from examples.db_management.schemas.website_access import canonical_ip
from examples.db_management.schemas.website_access import PolicyInput
from examples.db_management.schemas.website_access import PolicyRead
from examples.db_management.schemas.website_access import RuleInput
from examples.db_management.schemas.website_access import SiteKey
from examples.db_management.services import website_access as service

router = APIRouter(prefix='/admin/website-access', tags=['website-access'])
Db = Annotated[AsyncSession, Depends(get_db)]
Admin = Annotated[User, Depends(require_super_admin)]


def client_ip(request: Request) -> str:
    """Use the ASGI client supplied by the trusted proxy configuration."""
    try:
        return canonical_ip(request.client.host if request.client else '')
    except ValueError:
        raise HTTPException(
            400, detail={'code': 'client_ip_unavailable'},
        ) from None


@router.get('/context')
async def context(request: Request, _admin: Admin) -> dict:
    """Show the IP the gateway sees; clients must not submit their own IP."""
    return {'client_ip': client_ip(request), 'can_manage_website_access': True}


@router.get('/gateway')
async def gateway(_admin: Admin) -> dict:
    """Report the revision applied by a healthy gateway."""
    return service.gateway_status()


@router.get('', response_model=list[PolicyRead])
async def list_policies(db: Db, _admin: Admin):
    """List each website and its labelled allowlist."""
    return [service.projection(p) for p in await service.policies(db)]


@router.get('/audit', response_model=list[AuditRead])
async def audit(
    db: Db,
    _admin: Admin,
    limit: int = Query(50, ge=1, le=200),
    site_key: SiteKey | None = None,
):
    """Read recent management changes, optionally filtered by website."""
    return await service.list_audit(db, limit, site_key)


@router.put('/{site_key}', response_model=PolicyRead)
async def update_policy(
    site_key: SiteKey,
    payload: PolicyInput,
    request: Request,
    db: Db,
    admin: Admin,
):
    """Enable/disable IP restrictions while rejecting stale forms."""
    return await service.update_policy(
        db,
        site_key,
        payload.restricted,
        payload.expected_version,
        admin.id,
        client_ip(request),
    )


@router.post('/{site_key}/rules', response_model=PolicyRead, status_code=201)
async def create_rule(
    site_key: SiteKey,
    payload: RuleInput,
    request: Request,
    db: Db,
    admin: Admin,
):
    """Add a labelled single address or CIDR to one website."""
    return await service.create_rule(
        db,
        site_key,
        payload,
        admin.id,
        client_ip(request),
    )


@router.put('/{site_key}/rules/{rule_id}', response_model=PolicyRead)
async def update_rule(
    site_key: SiteKey,
    rule_id: UUID,
    payload: RuleInput,
    request: Request,
    db: Db,
    admin: Admin,
):
    """Edit a rule's address, label or enabled state within its own website."""
    return await service.update_rule(
        db,
        site_key,
        rule_id,
        payload,
        admin.id,
        client_ip(request),
    )


@router.delete('/{site_key}/rules/{rule_id}', response_model=PolicyRead)
async def delete_rule(
    site_key: SiteKey,
    rule_id: UUID,
    request: Request,
    db: Db,
    admin: Admin,
    expected_version: int = Query(ge=0),
):
    """Delete one website's rule while protecting the caller from lockout."""
    return await service.delete_rule(
        db,
        site_key,
        rule_id,
        expected_version,
        admin.id,
        client_ip(request),
    )
