"""Host-wide admission budgets shared by participating CHD processes."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic import field_validator

from src.settings.base import EnvironmentSettings


class GpuResourceSettings(EnvironmentSettings):
    model_config = {'env_prefix': 'GPU_RESOURCE_'}
    directory: Path = Path(tempfile.gettempdir()) / f'chd-gpu-{os.getuid()}'
    device_id: str = Field('gpu0', pattern=r'^[A-Za-z0-9_-]{1,100}$')
    inference_slots: int = Field(0, ge=0, le=64)
    memory_limit_mb: int = Field(0, ge=0, le=1048576)
    memory_unit_mb: Literal[64] = 64

    @field_validator('memory_limit_mb')
    @classmethod
    def whole_memory_units(cls, value: int) -> int:
        if value and value < 64:
            raise ValueError('GPU memory budget must be at least 64 MiB')
        return value
