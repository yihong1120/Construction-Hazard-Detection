from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import Field


class OAuthTokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: Literal['Bearer'] = 'Bearer'
    expires_in: int


class MeResponse(BaseModel):
    id: int
    username: str
    display_name: str
    role: str
    group_id: int | None = None
    status: str
    feature_names: list[str] = Field(default_factory=list)
