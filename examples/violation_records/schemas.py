from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class SiteOut(BaseModel):
    """
    Schema for returning information about a single site.

    Args:
        id (int): Unique identifier for the site.
        name (str): Name of the site.
        created_at (datetime): Timestamp when the site was created.
        updated_at (datetime): Timestamp when the site was last updated.
    """
    id: int
    name: str
    created_at: datetime
    updated_at: datetime


class ViolationItem(BaseModel):
    """
    Schema for returning a single Violation record, with details such as
    site_name, stream_name, detection_time, etc.

    Args:
        id (int): Unique identifier for the violation record.
        site_name (str): Name of the site where the violation occurred.
        stream_name (str): Name of the stream associated with the violation.
        detection_time (datetime): Timestamp when the violation was detected.
        image_path (str): Path to the image associated with the violation.
        created_at (datetime): Timestamp when the violation record was created.
        detection_items (str | None): Details of detected items, if any.
        warnings (str | None): Any warnings associated with the violation.
        cone_polygons (str | None): Polygons representing cones, if applicable.
        pole_polygons (str | None): Polygons representing poles, if applicable.
    """
    id: int
    site_name: str
    stream_name: str
    detection_time: datetime
    image_path: str
    created_at: datetime
    detection_items: str | None = None
    warnings: str | None = None
    cone_polygons: str | None = None
    pole_polygons: str | None = None


class ViolationList(BaseModel):
    """
    Schema for returning a paginated list of violation records.

    Args:
        total (int): Total number of violation records available.
        items (list[ViolationItem]): List of violation records.
    """
    total: int
    items: list[ViolationItem]


class UploadViolationResponse(BaseModel):
    """
    Schema for the response after uploading a violation record.

    Args:
        message (str):
            Message indicating the result of the upload.
        violation_id (int):
            Unique identifier for the uploaded violation record.
    """
    message: str
    violation_id: int
