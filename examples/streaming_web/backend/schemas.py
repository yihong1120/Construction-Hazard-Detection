"""Shared schemas and type definitions for the streaming web backend.

This module centralises common data structures used across WebSocket handlers
and Redis services. Using ``TypedDict`` keeps runtime structures lightweight
whilst providing strong editor and type-checking support.

All docstrings use British English.
"""
from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel


class FrameOutData(TypedDict, total=False):
    """A frame record read from Redis and prepared for outbound delivery.

    Keys are optional as upstream sources may omit some. The ``frame_bytes`` is
    the raw encoded image payload. Width and height describe image dimensions.
    """

    key: str
    id: str
    frame_bytes: bytes
    warnings: str
    cone_polygons: str
    pole_polygons: str
    detection_items: str
    width: int
    height: int


class FrameInHeader(TypedDict, total=False):
    """Header fields accompanying inbound frame bytes from clients."""

    label: str
    key: str
    warnings_json: str
    cone_polygons_json: str
    pole_polygons_json: str
    detection_items_json: str
    width: int
    height: int


class LabelListResponse(BaseModel):
    """Response model encapsulating a set of available labels."""

    labels: list[str]


class FramePostResponse(BaseModel):
    """Response model representing the status of a frame upload operation."""

    status: str
    message: str


__all__ = [
    'FrameOutData',
    'FrameInHeader',
    'LabelListResponse',
    'FramePostResponse',
]
