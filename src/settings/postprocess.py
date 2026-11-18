from __future__ import annotations

from typing import Literal

from pydantic import Field

from src.settings.base import EnvironmentSettings


class PostprocessSettings(EnvironmentSettings):
    backend: Literal['auto', 'python', 'rust'] = Field(
        'auto', validation_alias='POSTPROCESS_BACKEND',
    )
