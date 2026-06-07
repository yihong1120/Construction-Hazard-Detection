from __future__ import annotations

from pydantic import BaseModel
from pydantic import Field


class SiteCreate(BaseModel):
    """Schema representing the data required to create a new site."""

    name: str
    group_ids: list[int] = Field(default_factory=list)


class SiteUpdate(BaseModel):
    """Schema for updating the name of an existing site."""

    site_id: int
    new_name: str


class SiteDelete(BaseModel):
    """Schema representing the identifier of a site to be deleted."""

    site_id: int


class SiteUserOp(BaseModel):
    """Operations involving adding or removing a user from a site."""

    site_id: int
    user_id: int


class SiteGroupOp(BaseModel):
    """Operations involving adding or removing a group from a site."""

    site_id: int
    group_id: int


class SiteRead(BaseModel):
    """Details of a site retrieved from the database."""

    id: int
    name: str
    group_ids: list[int] = Field(default_factory=list)
    group_names: list[str] = Field(default_factory=list)
    user_ids: list[int]
