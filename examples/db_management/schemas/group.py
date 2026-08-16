from __future__ import annotations

from pydantic import BaseModel
from pydantic import ConfigDict


class GroupCreate(BaseModel):
    """Define input required to create a group.

    Attributes:
        name: Display name for the group.
        uniform_number: Organisation registration number for the group.
    """

    name: str
    uniform_number: str


class GroupUpdate(BaseModel):
    """Define a partial update for an existing group.

    Attributes:
        group_id: Identifier of the group to update.
        new_name: Replacement display name when it should change.
        new_uniform_number: Replacement organisation number when it should
            change.
    """

    group_id: int
    new_name: str | None = None
    new_uniform_number: str | None = None


class GroupDelete(BaseModel):
    """Identify a group that should be deleted.

    Attributes:
        group_id: Identifier of the group to delete.
    """

    group_id: int


class GroupRead(BaseModel):
    """Represent a group returned from persistent storage.

    Attributes:
        id: Database identifier of the group.
        name: Display name for the group.
        uniform_number: Organisation registration number for the group.
    """

    id: int
    name: str
    uniform_number: str
    model_config = ConfigDict(from_attributes=True)


class GroupFeatureRead(BaseModel):
    """Represent feature assignments for one group.

    Attributes:
        group_id: Database identifier of the group.
        group_name: Display name of the group.
        feature_ids: Identifiers of features assigned to the group.
    """

    group_id: int
    group_name: str
    feature_ids: list[int]
