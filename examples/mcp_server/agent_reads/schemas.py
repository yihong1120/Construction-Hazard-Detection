"""Bounded model inputs for the existing construction-domain read APIs."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class MySites(Strict):
    """List sites visible to the current authenticated user."""


class SearchViolations(Strict):
    site_id: int | None = Field(default=None, gt=0)
    keyword: str | None = Field(default=None, min_length=1, max_length=200)
    start_time: datetime | None = Field(default=None, strict=False)
    end_time: datetime | None = Field(default=None, strict=False)
    limit: int = Field(default=20, ge=1, le=100)
    cursor: str | None = Field(default=None, min_length=1, max_length=2000)

    @field_validator('keyword', 'cursor')
    @classmethod
    def nonempty(cls, value):
        if value is not None and not value.strip():
            raise ValueError('查詢文字及分頁游標不可空白。')
        return value.strip() if value is not None else None

    @model_validator(mode='after')
    def time_range(self):
        if (self.start_time is None) != (self.end_time is None):
            raise ValueError('開始與結束時間須一起提供。')
        if self.start_time is not None:
            if (
                self.start_time.utcoffset() is None
                or self.end_time.utcoffset() is None
            ):
                raise ValueError('查詢時間須包含時區，例如 +08:00。')
            if self.start_time > self.end_time:
                raise ValueError('開始時間不得晚於結束時間。')
        return self


class GetViolation(Strict):
    violation_id: int = Field(gt=0)
