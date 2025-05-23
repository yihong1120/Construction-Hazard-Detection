from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class StreamConfigCreate(BaseModel):
    """Schema representing a new stream configuration."""

    site_id: int
    stream_name: str
    video_url: str
    model_key: str = 'yolo11x'
    detect_with_server: bool = True
    work_start_hour: int = 7
    work_end_hour: int = 18

    detect_no_safety_vest_or_helmet: bool = False
    detect_near_machinery_or_vehicle: bool = False
    detect_in_restricted_area: bool = False
    detect_in_utility_pole_restricted_area: bool = False
    detect_machinery_close_to_pole: bool = False

    store_in_redis: bool = False
    expire_date: datetime | None = None


class StreamConfigUpdate(BaseModel):
    """Schema for updating an existing stream configuration.

    All fields are optional to allow partial updates.
    """

    stream_name: str | None = None
    video_url: str | None = None
    model_key: str | None = None
    detect_with_server: bool | None = None
    work_start_hour: int | None = None
    work_end_hour: int | None = None

    detect_no_safety_vest_or_helmet: bool | None = None
    detect_near_machinery_or_vehicle: bool | None = None
    detect_in_restricted_area: bool | None = None
    detect_in_utility_pole_restricted_area: bool | None = None
    detect_machinery_close_to_pole: bool | None = None

    store_in_redis: bool | None = None
    expire_date: datetime | None = None


class StreamConfigRead(BaseModel):
    """Stream configuration details retrieved from the database."""

    id: int
    stream_name: str
    video_url: str
    model_key: str

    detect_with_server: bool
    store_in_redis: bool
    work_start_hour: int
    work_end_hour: int

    detect_no_safety_vest_or_helmet: bool
    detect_near_machinery_or_vehicle: bool
    detect_in_restricted_area: bool
    detect_in_utility_pole_restricted_area: bool
    detect_machinery_close_to_pole: bool

    expire_date: datetime | None

    total_stream_in_group: int
    max_allowed_streams: int
    updated_at: datetime
