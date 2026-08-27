from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from fastapi import Security
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from examples.auth.database import get_db
from examples.auth.deployment_context import DeploymentBinding
from examples.auth.deployment_context import resolve_request_deployment
from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.models import Site
from examples.auth.models import User

SUPER_ADMIN_NAME = 'ChangDar'


class _NamedRoleUser(Protocol):
    """Define identity fields needed to recognise the super administrator.

    Attributes:
        username: Account username.
        role: Role assigned to the account.
    """

    username: str
    role: str


class _GroupedRoleUser(Protocol):
    """Define fields needed for group-scoped administrator checks.

    Attributes:
        role: Role assigned to the account.
        group_id: Optional group identifier assigned to the account.
    """

    role: str
    group_id: int | None


@dataclass(frozen=True, slots=True)
class TenantDeploymentAdministrator:
    """Verified tenant/deployment scope for one invitation administrator."""

    user: User
    tenant_id: UUID
    deployment_id: UUID


async def get_current_user(
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Retrieve the current authenticated user from JWT credentials.

    Args:
        credentials: JWT credentials obtained from the request.
        db: Database session used to load the account.

    Returns:
        Authenticated user including group, profile, and site relationships.

    Raises:
        HTTPException: If the token subject is invalid or the user no longer
            exists.
    """
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token subject')

    result = await db.execute(
        select(User)
        .options(
            selectinload(User.group),
            selectinload(User.profile),
            selectinload(User.sites),
        )
        .where(User.username == username),
    )
    user: User | None = result.unique().scalar_one_or_none()

    if user is None:
        raise HTTPException(status_code=401, detail='User not found')

    token_tenant_id = credentials.subject.get('tenant_id')
    # ``jwt_access`` rejects a real HTTP token without this claim.  Keeping
    # the check conditional lets non-HTTP service tests inject a minimal
    # credential double without weakening the production authentication path.
    if (
        isinstance(token_tenant_id, str)
        and str(user.tenant_id) != token_tenant_id
    ):
        raise HTTPException(
            status_code=409,
            detail={
                'code': 'deployment_configuration_changed',
                'message': 'Account tenant changed; sign in again.',
            },
        )

    return user


def is_super_admin(user: _NamedRoleUser) -> bool:
    """Return whether a user is the configured super administrator.

    Args:
        user: User-like object containing a username and role.

    Returns:
        ``True`` only for the configured administrator with an administrator
        role. Keycloak canonicalises usernames to lower case in many realm
        configurations, while the original Visionnaire record used
        ``ChangDar``.
    """
    return (
        user.username.casefold() == SUPER_ADMIN_NAME.casefold()
        and user.role in {'admin', 'super_admin'}
    )


def require_admin(user: User = Depends(get_current_user)) -> User:
    """Require administrator-level permission for a request.

    Args:
        user: Currently authenticated user.

    Returns:
        Authenticated user authorised to perform administrator operations.

    Raises:
        HTTPException: If the user does not have administrator privileges.
    """
    if user.role != 'admin' and not is_super_admin(user):
        raise HTTPException(status_code=403, detail='Admin required')

    return user


def require_super_admin(user: User = Depends(get_current_user)) -> User:
    """Require the configured super administrator for a request.

    Args:
        user: Currently authenticated user.

    Returns:
        Authenticated user authorised as the super administrator.

    Raises:
        HTTPException: If the user is not the super administrator.
    """
    if not is_super_admin(user):
        raise HTTPException(status_code=403, detail='Super admin only')

    return user


async def require_tenant_deployment_administrator(
    request: Request,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> TenantDeploymentAdministrator:
    """Resolve an admin's invitation scope without client-selected IDs.

    ``require_admin`` authenticates a deployment-bound JWT.  The binding below
    is independently resolved from the server-recognised API origin and must
    still match that user's tenant, so neither an input field nor a forwarded
    header can choose a different tenant or deployment.
    """
    binding: DeploymentBinding = await resolve_request_deployment(request, db)
    if user.tenant_id != binding.tenant_id:
        raise HTTPException(
            status_code=409,
            detail={
                'code': 'deployment_configuration_changed',
                'message': 'Account tenant changed; sign in again.',
            },
        )
    return TenantDeploymentAdministrator(
        user=user,
        tenant_id=binding.tenant_id,
        deployment_id=binding.deployment_id,
    )


def ensure_not_super(target: User) -> None:
    """Prevent an operation from targeting the super administrator.

    Args:
        target: User targeted by the requested operation.

    Raises:
        HTTPException: If the target is the configured super administrator.
    """
    if target.username.casefold() == SUPER_ADMIN_NAME.casefold():
        raise HTTPException(
            status_code=403,
            detail='Cannot operate on super admin',
        )


def ensure_admin_with_group(user: _GroupedRoleUser) -> None:
    """Require an administrator who is assigned to a group.

    Args:
        user: User-like object whose role and group are checked.

    Raises:
        HTTPException: If the user is not an administrator or has no assigned
            group.
    """
    if user.role != 'admin':
        raise HTTPException(status_code=403, detail='Admin required')

    if user.group_id is None:
        raise HTTPException(status_code=403, detail='Admin without group')


def _site_permission(
    op: User,
    site: Site | None = None,
    group_id: int | None = None,
) -> None:
    """Enforce site and group scope for an administrator operation.

    Args:
        op: User performing the operation.
        site: Optional site whose group membership is being checked.
        group_id: Optional group identifier being checked.

    Raises:
        HTTPException: If the user is not permitted to operate on the supplied
            site or group.
    """
    # The configured super administrator has cross-group management rights.
    if is_super_admin(op):
        return

    if op.role != 'admin':
        raise HTTPException(status_code=403, detail='Admin required')

    ensure_admin_with_group(op)

    # A site can be shared by multiple groups, so authorisation compares the
    # operator's group against the complete site membership set.
    if site:
        site_group_ids = {g.id for g in site.groups}
    else:
        site_group_ids = set()

    if site and op.group_id not in site_group_ids:
        raise HTTPException(
            status_code=403,
            detail="Cannot manage other group's site",
        )

    # Direct group operations must stay within the operator's assigned group.
    if group_id is not None and group_id != op.group_id:
        raise HTTPException(
            status_code=403,
            detail='Cannot operate on other group',
        )
