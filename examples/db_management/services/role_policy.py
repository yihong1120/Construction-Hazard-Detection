from __future__ import annotations

from fastapi import HTTPException

from examples.db_management.deps import ensure_admin_with_group
from examples.db_management.deps import is_super_admin
from examples.db_management.deps import require_admin
from examples.db_management.deps import SUPER_ADMIN_NAME
from examples.db_management.services.user_management_services import (
    ensure_user_management_scope,
)


def assignable_roles(operator, target=None) -> list[str]:
    require_admin(operator)
    if target is not None:
        if target.tenant_id != operator.tenant_id:
            raise HTTPException(403, detail='Cannot manage another tenant')
        if not is_super_admin(operator):
            ensure_admin_with_group(operator)
            if target.group_id != operator.group_id:
                raise HTTPException(
                    403, detail='Cannot manage users outside your group.',
                )
        # A visible account can legitimately have no assignable roles.
        # Mutations still reject every submitted role through validate_role.
        if target.username.casefold() == SUPER_ADMIN_NAME.casefold():
            return []
        if not is_super_admin(operator) and target.role in {
            'admin',
            'super_admin',
        }:
            return []
        ensure_user_management_scope(target, operator)
    return (
        ['admin', 'user', 'guest']
        if is_super_admin(operator)
        else ['user', 'guest']
    )


def validate_role(operator, role: str, target=None) -> None:
    if role not in assignable_roles(operator, target):
        raise HTTPException(403, detail='Role assignment is not allowed')
