from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from examples.YOLO_server_api.model_registry import revision_for
from examples.YOLO_server_api.model_registry import Text


class Option(BaseModel):
    id: Text
    label: Text


class ResourceOptions(BaseModel):
    schema_version: Literal[1] = 1
    revision: Text
    options: list[Option]
    default_id: str | None = None
    policy: dict = Field(default_factory=dict)

    @model_validator(mode='after')
    def valid_default(self):
        ids = [o.id for o in self.options]
        if len(ids) != len(set(ids)) or (
            self.default_id is not None and self.default_id not in ids
        ):
            raise ValueError('Invalid option IDs/default')
        return self


def resource_options(
    user, target, options, default=None, policy=None, source=None,
):
    result = ResourceOptions(
        revision='pending',
        options=options,
        default_id=default,
        policy=policy or {},
    )
    result.revision = revision_for(
        {
            'user': user.id,
            'tenant': str(user.tenant_id),
            'role': user.role,
            'target': target,
            'source': source,
            'result': result.model_dump(exclude={'revision'}),
        },
    )
    return result
