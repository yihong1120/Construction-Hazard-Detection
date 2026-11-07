from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from src.settings.base import EnvironmentSettings


class OutboxRetention(BaseModel):
    max_events: int = Field(1000, ge=1)
    max_bytes: int = Field(64 * 1024 * 1024, ge=1)
    seconds: float = Field(7 * 86400, gt=0)


class DeliverySettings(EnvironmentSettings):
    model_config = {
        'env_prefix': 'VIOLATION_OUTBOX_',
        'env_nested_delimiter': '__',
    }
    directory_max_bytes: int = Field(2 * 1024**3, ge=1)
    directory_archive_max_bytes: int | None = Field(None, ge=0)
    min_free_bytes: int = Field(128 * 1024**2, ge=0)
    maintenance_batch_size: int = Field(16, ge=1, le=128)
    archive_cleanup_batch_size: int = Field(128, ge=1, le=4096)
    maintenance_budget_seconds: float = Field(
        0.1,
        gt=0,
        le=5,
        allow_inf_nan=False,
    )
    discovery_seconds: float = Field(60, gt=0)
    idle_seconds: float = Field(300, ge=0)
    scheduler_capacity: int = Field(128, ge=1)
    cache_size: int = Field(64, ge=1)
    archive: OutboxRetention = Field(default_factory=OutboxRetention)
    directory: Path = Path('outputs/event-outbox')
    max_events: int = Field(100, ge=1)
    max_bytes: int = Field(256 * 1024 * 1024, ge=1)
    max_retry_seconds: float = Field(60, ge=1)
    max_attempts: int = Field(20, ge=1)
    poll_seconds: float = Field(1, ge=0.01)
    recovery_seconds: float = Field(1, gt=0, allow_inf_nan=False)
    max_recovery_seconds: float = Field(60, ge=1, allow_inf_nan=False)
    concurrency: int = Field(4, ge=1, le=32)
    event_timeout_seconds: float = Field(60, gt=0, le=100)

    @model_validator(mode='after')
    def archive_within_total(self):
        if self.recovery_seconds > self.max_recovery_seconds:
            raise ValueError('Initial recovery delay exceeds maximum')
        if (
            self.directory_archive_max_bytes is not None
            and self.directory_archive_max_bytes > self.directory_max_bytes
        ):
            raise ValueError('Archive budget exceeds directory budget')
        return self

    @property
    def archive_capacity(self) -> int:
        return (
            self.directory_max_bytes // 4
            if self.directory_archive_max_bytes is None
            else self.directory_archive_max_bytes
        )
