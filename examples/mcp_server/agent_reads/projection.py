"""Domain read allowlist; media and credentials stay private."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator


class Projection(BaseModel):
    model_config = ConfigDict(extra='ignore', strict=True)


class SiteRead(Projection):
    id: int = Field(gt=0)
    name: str


class WarningRead(Projection):
    count: int


class ViolationRead(Projection):
    id: int = Field(gt=0)
    site_name: str
    stream_name: str
    detection_time: str
    model_version: str | None = None
    warnings: dict[str, WarningRead] = Field(default_factory=dict)
    warning_text: str | None = None
    is_flagged: bool = False
    review_status: Literal['pending', 'resolved', 'dismissed'] | None = None
    feedback_note: str | None = None

    @field_validator('detection_time')
    @classmethod
    def timestamp(cls, value):
        datetime.fromisoformat(value.replace('Z', '+00:00'))
        return value


class ViolationPage(Projection):
    items: list[ViolationRead]
    next_cursor: str | None
    has_more: bool


class ViolationDetail(Projection):
    id: int = Field(gt=0)
    site_name: str
    stream_name: str
    detection_time: str
    created_at: str
    model_id: str | None = None
    model_version: str | None = None
    warnings: str | None = None
    warning_text: str | None = None
    is_flagged: bool = False
    review_status: Literal['pending', 'resolved', 'dismissed'] | None = None
    review_note: str | None = None
    feedback_note: str | None = None

    @field_validator('detection_time', 'created_at')
    @classmethod
    def timestamp(cls, value):
        datetime.fromisoformat(value.replace('Z', '+00:00'))
        return value
