from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.config import Settings
from examples.auth.models import User
from examples.auth.models import UserIdentity
from examples.db_management.schemas.openwebui import OpenWebUIAccessScope
from examples.db_management.schemas.openwebui import OpenWebUIPasswordChange
from examples.db_management.schemas.openwebui import OpenWebUIUserCreate
from examples.db_management.schemas.openwebui import OpenWebUIUserCreated
from examples.db_management.schemas.openwebui import OpenWebUIUserUpdate
from examples.db_management.schemas.user import UserCreate
from examples.db_management.schemas.user import UserDelete
from examples.db_management.services.keycloak_user_management_services import (
    assign_keycloak_client_role,
)
from examples.db_management.services.keycloak_user_management_services import (
    logout_keycloak_user,
)
from examples.db_management.services.keycloak_user_management_services import (
    reset_keycloak_password,
)
from examples.db_management.services.keycloak_user_management_services import (
    set_keycloak_user_enabled,
)
from examples.db_management.services.openwebui_account_update import (
    OpenWebUIAccountUpdate,
)
from examples.db_management.services.password_policy import (
    validate_password_minimum,
)
from examples.db_management.services.site_services import (
    list_site_ids_for_group,
)
from examples.db_management.services.user_action_services import (
    creating_account,
)
from examples.db_management.services.user_action_services import remove_user
from examples.db_management.services.user_management_services import (
    load_user_read,
)
from examples.db_management.services.user_services import (
    ensure_keycloak_subject_for_user,
)
from examples.db_management.services.user_services import get_user_by_id

settings = Settings()


async def change_own_password_from_openwebui(
    payload: OpenWebUIPasswordChange,
    db: AsyncSession,
    user: User,
) -> dict[str, str]:
    """Write the user's password only to Keycloak and revoke old sessions."""
    validate_password_minimum(payload.new_password)
    subject = await ensure_keycloak_subject_for_user(user, db)
    await reset_keycloak_password(
        subject,
        password=payload.new_password,
        temporary=False,
    )
    await logout_keycloak_user(subject)
    return {'message': 'Keycloak password changed; sessions revoked.'}


async def create_user_from_openwebui(
    payload: OpenWebUIUserCreate,
    db: AsyncSession,
    operator: User,
) -> OpenWebUIUserCreated:
    """Create through the canonical Visionnaire and Keycloak transaction."""
    canonical_data = payload.model_dump(exclude={'openwebui_role'})
    if canonical_data['group_id'] is None:
        canonical_data['group_id'] = operator.group_id
    canonical_payload = UserCreate.model_validate(canonical_data)
    async with creating_account(canonical_payload, db, operator) as account:
        await assign_keycloak_client_role(
            account.subject,
            client_id='open-webui',
            role_name=(
                'openwebui-admin'
                if payload.role == 'admin'
                else 'openwebui-user'
            ),
        )
        if payload.openwebui_role == 'pending':
            await set_keycloak_user_enabled(account.subject, enabled=False)
            account.user.status = 'suspended'
        user = await load_user_read(account.user.id, db)
        site_ids = (
            await list_site_ids_for_group(user.group_id, db)
            if user.group_id is not None
            else []
        )
        return {
            'user': user,
            'keycloak_subject': account.subject,
            'scope': OpenWebUIAccessScope(
                tenant_id=account.user.tenant_id,
                group_id=user.group_id,
                group_name=user.group_name,
                site_ids=site_ids,
            ),
        }


async def delete_user_from_openwebui(
    user_id: int,
    db: AsyncSession,
    operator: User,
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
    target = await get_user_by_id(identity.user_id, db)
    if target is None:
        raise HTTPException(404, 'Visionnaire user not found.')
    return target


async def update_user_from_openwebui(
    subject: str,
    payload: OpenWebUIUserUpdate,
    db: AsyncSession,
    operator: User,
) -> dict[str, str]:
    """Synchronize lifecycle, role, group and password through Visionnaire."""
    target = await target_from_subject(subject, db)
    await OpenWebUIAccountUpdate(target, payload, operator, db).execute(
        subject,
    )
    return {'message': 'Visionnaire user synchronized.'}


async def delete_user_by_subject_from_openwebui(
    subject: str,
    db: AsyncSession,
    operator: User,
) -> dict[str, str]:
    target = await target_from_subject(subject, db)
    return await remove_user(UserDelete(user_id=target.id), db, operator)
