from __future__ import annotations

from pydantic import Field

from src.settings.base import EnvironmentSettings


class ResponseDebugSettings(EnvironmentSettings):
    response_ids: str = Field(
        '',
        validation_alias='VIOLATION_DEBUG_RESPONSE_IDS',
    )

    @property
    def selected(self) -> frozenset[str]:
        return frozenset(
            value.strip()
            for value in self.response_ids.split(',')
            if value.strip()
        )
