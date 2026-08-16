from __future__ import annotations

from pydantic import BaseModel
from pydantic import ConfigDict


class FeatureCreate(BaseModel):
    """Define input required to create a feature.

    Attributes:
        feature_name: Unique, human-readable feature name.
        description: Optional explanation of the feature's purpose.
    """

    feature_name: str
    description: str | None = None


class FeatureUpdate(BaseModel):
    """Define a partial feature update.

    Attributes:
        feature_id: Identifier of the feature to update.
        new_name: Replacement feature name when it should change.
        new_description: Replacement description when it should change.
    """

    feature_id: int
    new_name: str | None = None
    new_description: str | None = None


class FeatureDelete(BaseModel):
    """Identify a feature that should be deleted.

    Attributes:
        feature_id: Identifier of the feature to delete.
    """

    feature_id: int


class FeatureRead(BaseModel):
    """Represent a feature returned from persistent storage.

    Attributes:
        id: Database identifier of the feature.
        feature_name: Human-readable feature name.
        description: Optional explanation of the feature's purpose.
    """

    id: int
    feature_name: str
    description: str | None = None
    model_config = ConfigDict(from_attributes=True)


class GroupFeatureUpdate(BaseModel):
    """Define the complete feature set assigned to a group.

    Attributes:
        group_id: Identifier of the group whose access is being updated.
        feature_ids: Feature identifiers to assign to the group.
    """

    group_id: int
    feature_ids: list[int]
