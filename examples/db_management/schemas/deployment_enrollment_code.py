from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class DeploymentEnrollmentCodeCreate(BaseModel):
    """Create one invite for the authenticated deployment only."""

    model_config = ConfigDict(extra='forbid', strict=True)

    expires_in_minutes: int = Field(ge=1, le=1440)


class DeploymentEnrollmentCodeCreated(BaseModel):
    """The only response shape that exposes a raw code, exactly once."""

    model_config = ConfigDict(extra='forbid')

    id: UUID
    enrollment_code: str
    expires_at: datetime


class DeploymentEnrollmentCodeItem(BaseModel):
    """Safe invitation metadata that never exposes code material."""

    model_config = ConfigDict(extra='forbid')

    id: UUID
    expires_at: datetime
    status: Literal['active', 'redeemed', 'expired', 'revoked']


class DeploymentEnrollmentCodeList(BaseModel):
    """List the authenticated deployment's invitation metadata."""

    model_config = ConfigDict(extra='forbid')

    items: list[DeploymentEnrollmentCodeItem]
