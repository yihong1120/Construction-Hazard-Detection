from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class StreamConfigCreate(BaseModel):
    """Define input required to create a stream configuration.

    Attributes:
        site_id: Identifier of the site that owns the stream.
        stream_name: Unique stream name within the selected site.
        video_url: Source URL used to ingest the stream.
        model_key: Detection-model identifier selected for the stream.
        recognition_enabled: Whether automated recognition is enabled.
        work_start_hour: Start of the daily recognition window in local hours.
        work_end_hour: End of the daily recognition window in local hours.
        detect_no_safety_vest_or_helmet: Whether PPE violations are detected.
        detect_near_machinery_or_vehicle: Whether proximity violations are
            detected.
        detect_in_restricted_area: Whether restricted-area violations are
            detected.
        detect_in_utility_pole_restricted_area: Whether utility-pole area
            violations are detected.
        detect_machinery_close_to_pole: Whether machinery-to-pole proximity is
            detected.
        expire_date: Optional time after which the configuration expires.
        group_id: Optional group assigned to manage the stream.
    """

    site_id: int
    stream_name: str
    video_url: str
    model_key: str = 'yolo26n'
    recognition_enabled: bool = True
    work_start_hour: int = 7
    work_end_hour: int = 18

    detect_no_safety_vest_or_helmet: bool = False
    detect_near_machinery_or_vehicle: bool = False
    detect_in_restricted_area: bool = False
    detect_in_utility_pole_restricted_area: bool = False
    detect_machinery_close_to_pole: bool = False

    expire_date: datetime | None = None
    group_id: int | None = None


class StreamConfigUpdate(BaseModel):
    """Define a partial update for an existing stream configuration.

    Attributes:
        stream_name: Replacement name for the stream.
        video_url: Replacement source URL for the stream.
        model_key: Replacement detection-model identifier.
        recognition_enabled: Whether automated recognition should run.
        work_start_hour: Replacement start of the daily recognition window.
        work_end_hour: Replacement end of the daily recognition window.
        detect_no_safety_vest_or_helmet: Whether PPE violations are detected.
        detect_near_machinery_or_vehicle: Whether proximity violations are
            detected.
        detect_in_restricted_area: Whether restricted-area violations are
            detected.
        detect_in_utility_pole_restricted_area: Whether utility-pole area
            violations are detected.
        detect_machinery_close_to_pole: Whether machinery-to-pole proximity is
            detected.
        expire_date: Replacement optional expiry time.
    """

    stream_name: str | None = None
    video_url: str | None = None
    model_key: str | None = None
    recognition_enabled: bool | None = None
    work_start_hour: int | None = None
    work_end_hour: int | None = None

    detect_no_safety_vest_or_helmet: bool | None = None
    detect_near_machinery_or_vehicle: bool | None = None
    detect_in_restricted_area: bool | None = None
    detect_in_utility_pole_restricted_area: bool | None = None
    detect_machinery_close_to_pole: bool | None = None

    expire_date: datetime | None = None


class SiteStreamConfigItem(BaseModel):
    """Define one stream in a site-scoped configuration replacement.

    Attributes:
        id: Existing stream identifier; omitted for a new stream.
        stream_name: Unique stream name within the site.
        video_url: Source URL used to ingest the stream.
        model_key: Detection-model identifier selected for the stream.
        recognition_enabled: Whether automated recognition is enabled.
        work_start_hour: Start of the daily recognition window in local hours.
        work_end_hour: End of the daily recognition window in local hours.
        detect_no_safety_vest_or_helmet: Whether PPE violations are detected.
        detect_near_machinery_or_vehicle: Whether proximity violations are
            detected.
        detect_in_restricted_area: Whether restricted-area violations are
            detected.
        detect_in_utility_pole_restricted_area: Whether utility-pole area
            violations are detected.
        detect_machinery_close_to_pole: Whether machinery-to-pole proximity is
            detected.
        expire_date: Optional time after which the configuration expires.
    """

    id: int | None = None
    stream_name: str
    video_url: str
    model_key: str = 'yolo26n'
    recognition_enabled: bool = True
    work_start_hour: int = 7
    work_end_hour: int = 18

    detect_no_safety_vest_or_helmet: bool = False
    detect_near_machinery_or_vehicle: bool = False
    detect_in_restricted_area: bool = False
    detect_in_utility_pole_restricted_area: bool = False
    detect_machinery_close_to_pole: bool = False

    expire_date: datetime | None = None


class SiteStreamConfigUpsert(BaseModel):
    """Define a complete site-scoped stream-configuration replacement.

    Attributes:
        streams: Desired stream configurations for the selected site.
    """

    streams: list[SiteStreamConfigItem]


class StreamConfigRead(BaseModel):
    """Represent a stream configuration returned from persistent storage.

    Attributes:
        id: Database identifier of the stream configuration.
        stream_name: Unique stream name within its site.
        video_url: Source URL used to ingest the stream.
        model_key: Detection-model identifier selected for the stream.
        recognition_enabled: Whether automated recognition is enabled.
        work_start_hour: Start of the daily recognition window in local hours.
        work_end_hour: End of the daily recognition window in local hours.
        detect_no_safety_vest_or_helmet: Whether PPE violations are detected.
        detect_near_machinery_or_vehicle: Whether proximity violations are
            detected.
        detect_in_restricted_area: Whether restricted-area violations are
            detected.
        detect_in_utility_pole_restricted_area: Whether utility-pole area
            violations are detected.
        detect_machinery_close_to_pole: Whether machinery-to-pole proximity is
            detected.
        expire_date: Optional time after which the configuration expires.
        total_stream_in_group: Number of streams currently assigned to its
            group.
        max_allowed_streams: Maximum streams permitted for that group.
        updated_at: Time at which the configuration was last changed.
    """

    id: int
    stream_name: str
    video_url: str
    model_key: str

    recognition_enabled: bool
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
