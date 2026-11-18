"""Bounding-box operations shared by safety checks and geometry."""
from __future__ import annotations


def normalise_bbox(bbox: list[float]) -> list[float]:
    """Order box coordinates while preserving confidence and tracking data."""
    left_x = min(bbox[0], bbox[2])
    right_x = max(bbox[0], bbox[2])
    top_y = min(bbox[1], bbox[3])
    bottom_y = max(bbox[1], bbox[3])
    return [left_x, top_y, right_x, bottom_y, *bbox[4:]]
