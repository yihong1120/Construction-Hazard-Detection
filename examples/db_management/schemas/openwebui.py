from __future__ import annotations

from typing import Literal
from typing import TypedDict
from uuid import UUID

from pydantic import BaseModel
from pydantic import EmailStr

from examples.db_management.schemas.user import UserCreate
from examples.db_management.schemas.user import UserRead


class OpenWebUIUserUpdate(BaseModel):
    role: Literal['pending', 'user', 'admin'] | None = None
    password: str | None = None
    group_id: int | None = None
    name: str | None = None
    email: EmailStr | None = None


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


class OpenWebUIUserCreated(TypedDict):
    user: UserRead
    keycloak_subject: str
    scope: OpenWebUIAccessScope
