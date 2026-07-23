from __future__ import annotations

from pydantic import BaseModel
from pydantic import Field

from examples.db_management.schemas.auth import UserLogin


class UserSummary(BaseModel):
    id: int
    username: str
    display_name: str
    role: str
    group_id: int | None = None
    status: str


class BffSessionResponse(BaseModel):
    authenticated: bool = True
    user: UserSummary
    feature_names: list[str] = Field(default_factory=list)


class CsrfResponse(BaseModel):
    csrf_token: str


class BffLoginRequest(UserLogin):
    """Web credentials; tokens are never included in the response."""
