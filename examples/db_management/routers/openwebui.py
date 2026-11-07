from __future__ import annotations

from fastapi import APIRouter

from examples.db_management.openwebui_deps import Database
from examples.db_management.openwebui_deps import OpenWebUIAdmin
from examples.db_management.openwebui_deps import OpenWebUIUser
from examples.db_management.schemas.openwebui import OpenWebUIPasswordChange
from examples.db_management.schemas.openwebui import OpenWebUIUserCreate
from examples.db_management.schemas.openwebui import OpenWebUIUserCreated
from examples.db_management.schemas.openwebui import OpenWebUIUserUpdate
from examples.db_management.services import openwebui_services as service

router = APIRouter(prefix='/internal/openwebui', include_in_schema=False)


@router.put('/users/me/password')
async def change_own_password_from_openwebui(
    payload: OpenWebUIPasswordChange,
    db: Database,
    user: OpenWebUIUser,
) -> dict[str, str]:
    """Write the user's password only to Keycloak and revoke old sessions."""
    return await service.change_own_password_from_openwebui(payload, db, user)


@router.post('/users', status_code=201)
async def create_user_from_openwebui(
    payload: OpenWebUIUserCreate,
    db: Database,
    operator: OpenWebUIAdmin,
) -> OpenWebUIUserCreated:
    """Create through the canonical Visionnaire and Keycloak transaction."""
    return await service.create_user_from_openwebui(payload, db, operator)


@router.delete('/users/{user_id}')
async def delete_user_from_openwebui(
    user_id: int,
    db: Database,
    operator: OpenWebUIAdmin,
) -> dict[str, str]:
    """Compensate or perform a canonical cross-application deletion."""
    return await service.delete_user_from_openwebui(user_id, db, operator)


@router.patch('/users/by-sub/{subject}')
async def update_user_from_openwebui(
    subject: str,
    payload: OpenWebUIUserUpdate,
    db: Database,
    operator: OpenWebUIAdmin,
) -> dict[str, str]:
    """Synchronize lifecycle, role, group and password through Visionnaire."""
    return await service.update_user_from_openwebui(
        subject, payload, db, operator,
    )


@router.delete('/users/by-sub/{subject}')
async def delete_user_by_subject_from_openwebui(
    subject: str,
    db: Database,
    operator: OpenWebUIAdmin,
) -> dict[str, str]:
    """Delete a linked identity through the canonical account service."""
    return await service.delete_user_by_subject_from_openwebui(
        subject, db, operator,
    )
