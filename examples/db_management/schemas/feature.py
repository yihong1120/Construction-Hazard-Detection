from __future__ import annotations

from pydantic import BaseModel
from pydantic import ConfigDict


class FeatureCreate(BaseModel):
    """Schema for creating a new feature."""

    feature_name: str
    description: str | None = None


class FeatureUpdate(BaseModel):
    """Schema for updating an existing feature's details."""

    feature_id: int
    new_name: str | None = None
    new_description: str | None = None


class FeatureDelete(BaseModel):
    """Schema for deleting a feature."""

    feature_id: int


class FeatureRead(BaseModel):
    """Schema representing detailed information about a feature."""

    id: int
    feature_name: str
    description: str | None = None
    model_config = ConfigDict(from_attributes=True)


class GroupFeatureUpdate(BaseModel):
    """Schema for updating the set of features associated with a group."""

    group_id: int
    feature_ids: list[int]
