"""Translate Open WebUI account actions into canonical identity operations."""
from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.security import HTTPBearer
from jwt.exceptions import InvalidTokenError
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .users import add_user
from .users import admin_update_pwd_by_id
from .users import change_group
from .users import change_role
from .users import remove_user
from .users import update_profile
from .users import update_user_status
from examples.auth.config import Settings
from examples.auth.database import get_db
from examples.auth.models import User
from examples.auth.models import USER_STATUS_ACTIVE
from examples.auth.models import UserIdentity
from examples.auth.oidc import OidcTokenVerifier
from examples.db_management.deps import is_super_admin
from examples.db_management.schemas.user import SetUserStatus
from examples.db_management.schemas.user import UpdatePasswordById
from examples.db_management.schemas.user import UpdateUserGroup
from examples.db_management.schemas.user import UpdateUserRole
from examples.db_management.schemas.user import UserCreate
from examples.db_management.schemas.user import UserDelete
from examples.db_management.schemas.user import UserProfileUpdate
from examples.db_management.schemas.user import UserRead
from examples.db_management.services.keycloak_user_management_services import (
    assign_keycloak_client_role,
)
from examples.db_management.services.keycloak_user_management_services import (
    delete_keycloak_user,
)
from examples.db_management.services.keycloak_user_management_services import (
    logout_keycloak_user,
)
from examples.db_management.services.keycloak_user_management_services import (
    remove_keycloak_client_role,
)
from examples.db_management.services.keycloak_user_management_services import (
    reset_keycloak_password,
)
from examples.db_management.services.password_policy import (
    validate_password_minimum,
)
from examples.db_management.services.site_services import (
    list_site_ids_for_group,
)
from examples.db_management.services.user_services import (
    ensure_keycloak_subject_for_user,
)

settings = Settings()
bearer = HTTPBearer(auto_error=False)
verifier = OidcTokenVerifier(
    issuer=settings.oidc_issuer_url,
    jwks_url=settings.oidc_jwks_url,
    audiences=('account',),
    algorithms=settings.oidc_algorithms,
    jwks_cache_seconds=settings.oidc_jwks_cache_seconds,
    jwks_timeout_seconds=settings.oidc_jwks_timeout_seconds,
)
router = APIRouter(prefix='/internal/openwebui', include_in_schema=False)
Database = Annotated[AsyncSession, Depends(get_db)]
BearerCredential = Annotated[
    HTTPAuthorizationCredentials | None,
    Depends(bearer),
]


class OpenWebUIUserUpdate(BaseModel):
    role: str | None = None
    password: str | None = None
    group_id: int | None = None
    name: str | None = None
    email: str | None = None


class OpenWebUIUserCreate(UserCreate):
    """Native Open WebUI form plus its local activation state."""

    openwebui_role: str = 'user'


class OpenWebUIAccessScope(BaseModel):
    """Authoritative company scope inherited by the new identity."""

    tenant_id: UUID
    group_id: int | None
    group_name: str | None
    site_ids: list[int]


class OpenWebUIPasswordChange(BaseModel):
    new_password: str


async def openwebui_user(
    credentials: BearerCredential,
    db: Database,
) -> User:
    """Resolve a verified Open WebUI OIDC subject to its active user."""
    if credentials is None or credentials.scheme.lower() != 'bearer':
        raise HTTPException(401, 'Open WebUI sign-in required.')
    try:
        claims = await verifier.decode_access_token(credentials.credentials)
    except InvalidTokenError as exc:
        raise HTTPException(401, 'Invalid Open WebUI identity.') from exc
    if claims.get('azp') != 'open-webui':
        raise HTTPException(401, 'Invalid Open WebUI client.')
    identity = await db.scalar(
        select(UserIdentity)
        .options(selectinload(UserIdentity.user))
        .where(
            UserIdentity.provider == settings.oidc_identity_provider,
            UserIdentity.provider_user_id == claims['sub'],
        ),
    )
    if identity is None:
        raise HTTPException(401, 'Visionnaire identity is not linked.')
    user = identity.user
    if user.status != USER_STATUS_ACTIVE:
        raise HTTPException(403, 'Visionnaire account is not active.')
    return user


async def openwebui_admin(
    credentials: BearerCredential,
    db: Database,
) -> User:
    """Resolve a verified Open WebUI OIDC subject to a Visionnaire admin."""
    user = await openwebui_user(credentials, db)
    if user.role != 'admin' and not is_super_admin(user):
        raise HTTPException(403, 'Visionnaire administrator required.')
    return user


OpenWebUIUser = Annotated[User, Depends(openwebui_user)]
OpenWebUIAdmin = Annotated[User, Depends(openwebui_admin)]


@router.put('/users/me/password')
async def change_own_password_from_openwebui(
    payload: OpenWebUIPasswordChange,
    db: Database,
    user: OpenWebUIUser,
) -> dict[str, str]:
    """Write the user's password only to Keycloak and revoke old sessions."""
    validate_password_minimum(payload.new_password)
    subject = await ensure_keycloak_subject_for_user(user, db)
    await reset_keycloak_password(
        subject, password=payload.new_password, temporary=False,
    )
    await logout_keycloak_user(subject)
    return {'message': 'Keycloak password changed; sessions revoked.'}


@router.post('/users', status_code=201)
async def create_user_from_openwebui(
    payload: OpenWebUIUserCreate,
    db: Database,
    operator: OpenWebUIAdmin,
) -> dict[str, UserRead | OpenWebUIAccessScope | str]:
    """Create through the canonical Visionnaire and Keycloak transaction."""
    canonical_data = payload.model_dump(exclude={'openwebui_role'})
    if canonical_data['group_id'] is None:
        canonical_data['group_id'] = operator.group_id
    canonical_payload = UserCreate.model_validate(canonical_data)
    user = await add_user(payload=canonical_payload, db=db, me=operator)
    stored_user = await db.get(User, user.id)
    subject = await ensure_keycloak_subject_for_user(stored_user, db)
    try:
        await assign_keycloak_client_role(
            subject,
            client_id='open-webui',
            role_name=(
                'openwebui-admin'
                if payload.role == 'admin'
                else 'openwebui-user'
            ),
        )
    except HTTPException:
        await delete_keycloak_user(subject, suppress_errors=True)
        await db.delete(stored_user)
        await db.commit()
        raise
    if payload.openwebui_role == 'pending':
        await update_user_status(
            SetUserStatus(user_id=user.id, status='suspended'), db, operator,
        )
    site_ids = (
        await list_site_ids_for_group(user.group_id, db)
        if user.group_id is not None
        else []
    )
    return {
        'user': user,
        'keycloak_subject': subject,
        'scope': OpenWebUIAccessScope(
            tenant_id=stored_user.tenant_id,
            group_id=user.group_id,
            group_name=user.group_name,
            site_ids=site_ids,
        ),
    }


@router.delete('/users/{user_id}')
async def delete_user_from_openwebui(
    user_id: int,
    db: Database,
    operator: OpenWebUIAdmin,
) -> dict[str, str]:
    """Compensate or perform a canonical cross-application deletion."""
    return await remove_user(
        payload=UserDelete(user_id=user_id),
        db=db,
        me=operator,
    )


async def target_from_subject(subject: str, db: AsyncSession) -> User:
    identity = await db.scalar(
        select(UserIdentity).where(
            UserIdentity.provider == settings.oidc_identity_provider,
            UserIdentity.provider_user_id == subject,
        ),
    )
    if identity is None:
        raise HTTPException(404, 'Visionnaire user is not linked.')
    target = await db.get(User, identity.user_id)
    if target is None:
        raise HTTPException(404, 'Visionnaire user not found.')
    return target


@router.patch('/users/by-sub/{subject}')
async def update_user_from_openwebui(
    subject: str,
    payload: OpenWebUIUserUpdate,
    db: Database,
    operator: OpenWebUIAdmin,
) -> dict[str, str]:
    """Synchronize lifecycle, role, group and password through Visionnaire."""
    target = await target_from_subject(subject, db)
    if payload.role == 'pending':
        await update_user_status(
            SetUserStatus(user_id=target.id, status='suspended'), db, operator,
        )
    elif payload.role in {'user', 'admin'}:
        if target.status != USER_STATUS_ACTIVE:
            await update_user_status(
                SetUserStatus(
                    user_id=target.id, status='active',
                ), db, operator,
            )
        desired = (
            'openwebui-admin' if payload.role == 'admin' else 'openwebui-user'
        )
        previous = (
            'openwebui-admin' if target.role == 'admin' else 'openwebui-user'
        )
        await assign_keycloak_client_role(
            subject,
            client_id='open-webui',
            role_name=desired,
        )
        if desired != previous:
            try:
                await remove_keycloak_client_role(
                    subject, client_id='open-webui', role_name=previous,
                )
                await change_role(
                    UpdateUserRole(user_id=target.id, new_role=payload.role),
                    db,
                    operator,
                )
            except HTTPException:
                await remove_keycloak_client_role(
                    subject, client_id='open-webui', role_name=desired,
                )
                await assign_keycloak_client_role(
                    subject, client_id='open-webui', role_name=previous,
                )
                raise
        else:
            other = (
                'openwebui-user'
                if desired == 'openwebui-admin'
                else 'openwebui-admin'
            )
            await remove_keycloak_client_role(
                subject, client_id='open-webui', role_name=other,
            )
    if payload.group_id is not None:
        await change_group(
            UpdateUserGroup(user_id=target.id, new_group_id=payload.group_id),
            db,
            operator,
        )
    if payload.password:
        await admin_update_pwd_by_id(
            UpdatePasswordById(
                user_id=target.id,
                new_password=payload.password,
            ),
            db,
            operator,
            temporary=False,
        )
    if payload.name is not None or payload.email is not None:
        await update_profile(
            UserProfileUpdate(
                user_id=target.id,
                family_name='' if payload.name is not None else None,
                given_name=payload.name,
                email=payload.email,
            ),
            db,
            operator,
        )
    return {'message': 'Visionnaire user synchronized.'}


@router.delete('/users/by-sub/{subject}')
async def delete_user_by_subject_from_openwebui(
    subject: str,
    db: Database,
    operator: OpenWebUIAdmin,
) -> dict[str, str]:
    target = await target_from_subject(subject, db)
    return await remove_user(UserDelete(user_id=target.id), db, operator)
