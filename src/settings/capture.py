from __future__ import annotations

from pydantic import Field
from pydantic import model_validator

from src.settings.base import EnvironmentSettings


class CaptureSettings(EnvironmentSettings):
    model_config = {'env_prefix': 'STREAM_CAPTURE_'}
    reopen_delay_seconds: float = Field(5, ge=0, allow_inf_nan=False)
    max_reopen_delay_seconds: float = Field(60, ge=0, allow_inf_nan=False)
    freeze_reconnect_seconds: float = Field(0, ge=0, allow_inf_nan=False)
    freeze_sample_seconds: float = Field(1, ge=0.1, allow_inf_nan=False)
    freeze_frame_delta: float = Field(0.2, ge=0, allow_inf_nan=False)
    timestamp_reconnect_seconds: float = Field(30, ge=0, allow_inf_nan=False)
    timestamp_sample_seconds: float = Field(1, ge=0.1, allow_inf_nan=False)
    open_timeout_ms: int = Field(5000, gt=0)
    read_timeout_ms: int = Field(5000, gt=0)

    @model_validator(mode='after')
    def ordered_delays(self):
        if self.max_reopen_delay_seconds < self.reopen_delay_seconds:
            raise ValueError('Maximum reopen delay must cover initial delay')
        return self
