from __future__ import annotations

from pydantic import BaseModel
from pydantic import Field


class SiteCreate(BaseModel):
    """Define input required to create a site.

    Attributes:
        name: Display name of the site.
        group_ids: Initial groups allowed to access the site.
    """

    name: str
    group_ids: list[int] = Field(default_factory=list)


class SiteUpdate(BaseModel):
    """Define a site-name update.

    Attributes:
        site_id: Identifier of the site to update.
        new_name: Replacement display name for the site.
    """

    site_id: int
    new_name: str


class SiteDelete(BaseModel):
    """Identify a site that should be deleted.

    Attributes:
        site_id: Identifier of the site to delete.
    """

    site_id: int


class SiteUserOp(BaseModel):
    """Identify a user and site for an access-membership operation.

    Attributes:
        site_id: Identifier of the site whose membership changes.
        user_id: Identifier of the user to add or remove.
    """

    site_id: int
    user_id: int


class SiteGroupOp(BaseModel):
    """Identify a group and site for an access-membership operation.

    Attributes:
        site_id: Identifier of the site whose membership changes.
        group_id: Identifier of the group to add or remove.
    """

    site_id: int
    group_id: int


class SiteRead(BaseModel):
    """Represent a site returned from persistent storage.

    Attributes:
        id: Database identifier of the site.
        name: Display name of the site.
        group_ids: Identifiers of groups with access to the site.
        group_names: Display names of groups with access to the site.
        user_ids: Identifiers of users with direct access to the site.
    """

    id: int
    name: str
    group_ids: list[int] = Field(default_factory=list)
    group_names: list[str] = Field(default_factory=list)
    user_ids: list[int]
