from __future__ import annotations

from pydantic import BaseModel


class GroupCreate(BaseModel):
    """Schema representing the information required to create a new group."""

    name: str
    uniform_number: str


class GroupUpdate(BaseModel):
    """Data for updating an existing group's details."""

    group_id: int
    new_name: str | None = None
    new_uniform_number: str | None = None


class GroupDelete(BaseModel):
    """Schema representing the identifier of a group to be deleted."""

    group_id: int


class GroupRead(BaseModel):
    """Schema representing detailed information about a group."""

    id: int
    name: str
    uniform_number: str

    class Config:
        from_attributes = True


class GroupFeatureRead(BaseModel):
    """Schema representing a group's features retrieved from the database."""

    group_id: int
    group_name: str
    feature_ids: list[int]
