"""Shared overlay contract for detection and historical evidence."""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import FiniteFloat

from examples.shared.class_colors import overlay_style


class NormalizedBBox(BaseModel):
    """Represent an image-relative bounding box using 0..1 co-ordinates.

    Attributes:
        x: Left edge ratio.
        y: Top edge ratio.
        w: Width ratio.
        h: Height ratio.
    """

    x: float
    y: float
    w: float
    h: float

    @field_validator('x', 'y', 'w', 'h')
    @classmethod
    def validate_ratio(cls, value: float) -> float:
        """Validate an image-relative co-ordinate ratio.

        Args:
            value: Candidate ratio.

        Returns:
            Ratio constrained to the inclusive 0..1 range.

        Raises:
            ValueError: If the ratio falls outside the accepted range.
        """
        if not math.isfinite(value) or value < 0 or value > 1:
            raise ValueError('bbox ratio must be between 0 and 1')
        return value


class ViolationOverlayObject(BaseModel):
    """Represent one frontend overlay object for a violation image.

    Attributes:
        object_id: Stable detection identifier.
        label: Optional detected class label.
        confidence: Optional detector confidence.
        bbox: Image-relative bounding box.
        is_flagged: Whether feedback flagged this object.
        flag_reason: Optional reason for the flag.
        flag_note: Optional reviewer or feedback note.
    """

    object_id: str
    show_box: bool = True
    show_label: bool = True
    feedback_show_box: bool = True
    feedback_show_label: bool = True
    display_name: str | None = None
    code: str | None = None
    color: str = '#9E9E9E'
    label: str | None = None
    confidence: float | None = None
    bbox: NormalizedBBox
    is_flagged: bool = False
    flag_reason: str | None = None
    flag_note: str | None = None


class OverlayImageSize(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class OverlayCoordinates(BaseModel):
    space: Literal['normalized'] = 'normalized'
    bbox_format: Literal['xywh'] = 'xywh'
    polygon_format: Literal['xy'] = 'xy'
    origin: Literal['top_left'] = 'top_left'
    reference: Literal['original_image'] = 'original_image'


class OverlayRegion(BaseModel):
    id: str
    kind: Literal['cone', 'pole']
    display_name: str
    points: list[tuple[FiniteFloat, FiniteFloat]] = Field(min_length=3)
    color: str = Field(pattern=r'^#[0-9A-Fa-f]{6}$')
    fill_opacity: float = Field(ge=0, le=1)
    closed: Literal[True] = True


class OverlayPayload(BaseModel):
    """Identical overlay envelope for image detection and violation details."""

    overlay_schema_version: Literal[1] = 1
    image_size: OverlayImageSize | None = None
    overlay_coordinates: OverlayCoordinates = Field(
        default_factory=OverlayCoordinates,
    )
    class_metadata: list[dict] | None = None
    overlay_style: dict = Field(default_factory=overlay_style)
    overlay_objects: list[ViolationOverlayObject] | None = None
    overlay_regions: list[OverlayRegion] = Field(default_factory=list)
    region_status: dict[
        Literal['cone', 'pole'],
        Literal[
            'available',
            'empty',
            'not_computed',
            'not_recorded',
            'invalid',
            'image_unavailable',
        ],
    ] = Field(default_factory=dict)
