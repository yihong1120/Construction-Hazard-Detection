from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel
from pydantic import Field


class MeResponse(BaseModel):
    """Represent the public profile returned by native OAuth ``/me``.

    Attributes:
        id: Database identifier of the authenticated user.
        username: Unique account username.
        tenant_id: Current tenant used by downstream API authorization.
        display_name: User's display name.
        role: Role granted to the user.
        group_id: Optional identifier of the user's group.
        status: Current account lifecycle status.
        feature_names: Features granted to the user.
    """

    id: int
    username: str
    display_name: str
    tenant_id: UUID | None = None
    role: str
    group_id: int | None = None
    status: str
    feature_names: list[str] = Field(default_factory=list)
