from __future__ import annotations

import ipaddress
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

SiteKey = Literal['main', 'ai', 'wiki']


def canonical_ip(value: str) -> str:
    """Accept a single numeric IP, normalising IPv4-mapped IPv6."""
    if '%' in value:
        raise ValueError('Scoped addresses are not supported')
    ip = ipaddress.ip_address(value)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return str(ip)


class RuleInput(BaseModel):
    """Create or replace a labelled allowed network."""

    model_config = ConfigDict(extra='forbid')
    network: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=120)
    enabled: bool = True
    expected_version: int = Field(ge=0)

    @field_validator('network')
    @classmethod
    def validate_network(cls, value: str) -> str:
        """Reject hostnames, zones, and unrestricted or mapped CIDRs."""
        if '%' in value:
            raise ValueError('Scoped addresses are not supported')
        network = ipaddress.ip_network(value.strip(), strict=False)
        if network.prefixlen == 0:
            raise ValueError('Use unrestricted mode instead of a /0 network')
        if isinstance(network, ipaddress.IPv6Network) and network.overlaps(
            ipaddress.ip_network('::ffff:0:0/96'),
        ):
            raise ValueError('Use an IPv4 CIDR for IPv4-mapped addresses')
        return str(network)

    @field_validator('label')
    @classmethod
    def validate_label(cls, value: str) -> str:
        """Avoid blank labels after whitespace is removed."""
        value = value.strip()
        if not value:
            raise ValueError('A label is required')
        return value


class PolicyInput(BaseModel):
    """Versioned switch between unrestricted and allowlist mode."""

    model_config = ConfigDict(extra='forbid')
    restricted: bool
    expected_version: int = Field(ge=0)


class RuleRead(BaseModel):
    """A stored allowlist entry."""

    model_config = ConfigDict(from_attributes=True)
    id: UUID
    network: str
    label: str
    enabled: bool
    created_at: datetime


class PolicyRead(BaseModel):
    """Saved policy version; gateway application is asynchronous."""

    site_key: SiteKey
    hostname: str
    restricted: bool
    version: int
    updated_at: datetime
    rules: list[RuleRead]


class AuditRead(BaseModel):
    """Bounded audit projection with no authentication material."""

    model_config = ConfigDict(from_attributes=True)
    id: UUID
    site_key: str
    actor_user_id: int | None
    actor_ip: str
    action: str
    before: dict
    after: dict
    created_at: datetime
