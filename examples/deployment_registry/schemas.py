from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


# These values define the stable public Registry document contract.
REGISTRY_SCHEMA_VERSION: int = 1
MAX_REGISTRY_TTL_SECONDS: int = 24 * 60 * 60


class DeploymentRegistryDocument(BaseModel):
    """Define the public Registry document consumed by managed clients.

    Attributes:
        schema_version: Version of the signed document contract.
        deployment_id: Identifier of the enrolled deployment.
        tenant_id: Identifier of the deployment tenant.
        api_base_url: Canonical base URL for deployment API calls.
        config_revision: Monotonically increasing deployment configuration.
        issued_at: Unix timestamp at which the document was signed.
        expires_at: Unix timestamp at which the document expires.
        key_id: Identifier for the public key pinned by the client.
        signature: Base64url Ed25519 signature over the unsigned fields.
    """

    # Reject surplus fields so the public document remains predictable.
    model_config = ConfigDict(extra='forbid')

    schema_version: Literal[1]
    deployment_id: UUID
    tenant_id: UUID
    api_base_url: str
    config_revision: int = Field(ge=1)
    issued_at: int = Field(ge=0)
    expires_at: int = Field(ge=0)
    key_id: str = Field(
        min_length=1,
        max_length=256,
        pattern=r'^[A-Za-z0-9][A-Za-z0-9._-]*$',
    )
    # Ed25519 has a 64-byte signature.  Base64url without padding is exactly
    # 86 ASCII characters, which avoids ambiguous signature representations.
    signature: str = Field(pattern=r'^[A-Za-z0-9_-]{86}$')


class EnrollmentExchangeResponse(BaseModel):
    """Define the minimal result of redeeming an enrollment code.

    Attributes:
        deployment_id: Identifier associated with the redeemed code.
    """

    # The exchange response intentionally exposes no user or tenant data.
    model_config = ConfigDict(extra='forbid')

    deployment_id: UUID


class EnrollmentExchangeRequest(BaseModel):
    """Define the anonymous request used to redeem an enrollment code.

    Attributes:
        enrollment_code: Opaque URL-safe one-time enrollment code.
    """

    # Keep codes opaque: do not coerce values or accept unrelated payload keys.
    model_config = ConfigDict(extra='forbid', strict=True)

    enrollment_code: str = Field(
        min_length=16,
        max_length=512,
        pattern=r'^[A-Za-z0-9_-]+$',
    )
