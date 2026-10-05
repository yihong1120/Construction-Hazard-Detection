from __future__ import annotations

import os
from pathlib import Path

from fastapi import HTTPException
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator


class InvitationPolicy(BaseModel):
    model_config = ConfigDict(extra='forbid')
    max_minutes: int = Field(default=1440, ge=1, strict=True)
    minutes: list[int] = Field(default_factory=lambda: [30, 60, 120, 1440])
    default_minutes: int | None = 30

    @model_validator(mode='after')
    def valid_minutes(self):
        if any(
            type(n) is not int or n < 1 or n > self.max_minutes
            for n in self.minutes
        ):
            raise ValueError('Invalid invitation duration')
        if len(set(self.minutes)) != len(self.minutes):
            raise ValueError('Duplicate invitation duration')
        if (
            self.default_minutes is not None
            and self.default_minutes not in self.minutes
        ):
            raise ValueError('Invalid default duration')
        return self


def invitation_policy(tenant_id: str) -> InvitationPolicy:
    """Optional tenant policy, with the existing 1440-minute server ceiling."""
    path = os.getenv('INVITATION_POLICY_PATH')
    if not path:
        return InvitationPolicy()
    try:
        import json

        policies = json.loads(Path(path).read_text())
        return InvitationPolicy.model_validate(policies[tenant_id])
    except (KeyError, OSError, ValueError, TypeError) as exc:
        raise HTTPException(
            503, detail={'code': 'INVITATION_POLICY_UNAVAILABLE'},
        ) from exc


def validate_invitation(tenant_id: str, minutes: int) -> None:
    policy = invitation_policy(tenant_id)
    if minutes not in policy.minutes or minutes > policy.max_minutes:
        raise HTTPException(
            422, detail='Invitation duration is not currently allowed',
        )
