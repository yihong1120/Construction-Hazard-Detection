from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class TenantCreate(BaseModel):
    """Create a tenant without accepting any client-side session identity."""

    model_config = ConfigDict(extra='forbid', strict=True)

    name: str = Field(min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=4000)


class TenantUpdate(BaseModel):
    """Update tenant administration state."""

    model_config = ConfigDict(extra='forbid', strict=True)

    name: str | None = Field(default=None, min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=4000)
    status: str | None = None


class TenantRead(BaseModel):
    """Public management representation of one tenant."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: str | None
    status: str


class DeploymentCreate(BaseModel):
    """Create a canonical API deployment for an existing tenant."""

    model_config = ConfigDict(extra='forbid', strict=True)

    tenant_id: UUID
    api_base_url: str = Field(min_length=8, max_length=2048)


class DeploymentUpdate(BaseModel):
    """Change deployment settings; each change increments revision."""

    model_config = ConfigDict(extra='forbid', strict=True)

    tenant_id: UUID | None = None
    api_base_url: str | None = Field(
        default=None,
        min_length=8,
        max_length=2048,
    )
    status: str | None = None


class DeploymentRead(BaseModel):
    """Privileged management representation of a Registry deployment."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    tenant_id: UUID
    api_base_url: str
    config_revision: int
    status: str
