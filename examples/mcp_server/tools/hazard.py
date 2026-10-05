from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from examples.mcp_server.schemas import DetectionLikeDict
from examples.mcp_server.schemas import HazardResponse
if TYPE_CHECKING:
    from src.danger_detector import DangerDetector


class HazardTools:
    """Tools for detecting safety violations and generating warning
    polygons."""

    def __init__(self) -> None:
        """Initialise lazy hazard detection resources."""
        self.logger = logging.getLogger(__name__)
        self._detector: DangerDetector | None = None
        self._detection_items: dict[str, bool] | None = None
        self._lock = asyncio.Lock()

    async def detect_violations(
        self,
        detections: list[list[float]] | list[DetectionLikeDict],
        detection_items: dict[str, bool] | None = None,
    ) -> HazardResponse:
        """Analyse detection results for safety violations.

        Args:
            detections: Either raw lists of ``[x1, y1, x2, y2, conf, cls]`` or
                object dictionaries with keys such as ``bbox``/``box``,
                ``confidence``/``conf`` and ``class``/``cls``.
            detection_items: Fine-grained toggles for individual safety checks.

        Returns:
            dict[str, Any]: A mapping with ``warnings``, ``cone_polygons``,
                and ``pole_polygons``.
        """
        try:
            async with self._lock:
                if (
                    self._detector is None
                    or detection_items != self._detection_items
                ):
                    await self._init_detector(detection_items)
                    self._detection_items = (
                        dict(detection_items)
                        if detection_items is not None
                        else None
                    )
                detector = self._detector
                assert detector is not None
                norm_detections = self._normalise_detections(detections)
                result = await asyncio.to_thread(
                    detector.detect_danger, norm_detections,
                )
            warnings, cone_polygons, pole_polygons = result

            return {
                'warnings': warnings,
                'cone_polygons': cone_polygons,
                'pole_polygons': pole_polygons,
            }

        except Exception as e:
            self.logger.error(f"Violation detection failed: {e}")
            raise

    @staticmethod
    def _normalise_detections(
        detections: list[list[float]] | list[DetectionLikeDict],
    ) -> list[list[float]]:
        """Reject malformed rows instead of fabricating class zero
        detections.
        """
        normalized = []
        for detection in detections:
            row: Sequence[float | None]
            if isinstance(detection, dict):
                bbox = detection.get('bbox', detection.get('box'))
                confidence = detection.get('confidence', detection.get('conf'))
                class_id = detection.get('class_', detection.get('cls'))
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    raise ValueError(
                        'Detection bbox must contain four coordinates',
                    )
                row = [*bbox, confidence, class_id]
            else:
                row = detection
            if not isinstance(row, (list, tuple)) or len(row) < 6:
                raise ValueError(
                    'Detection row must contain at least six values',
                )
            try:
                values = []
                for value in row[:6]:
                    if value is None:
                        raise ValueError('Detection values must be numeric')
                    values.append(float(value))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError('Detection values must be numeric') from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError('Detection values must be finite')
            if (
                not 0 <= values[4] <= 1
                or values[5] < 0
                or not values[5].is_integer()
            ):
                raise ValueError('Invalid detection confidence or class ID')
            normalized.append(values)
        return normalized

    async def _init_detector(
        self,
        detection_items: dict[str, bool] | None,
    ) -> None:
        """Initialise the danger detector."""
        # Use provided detection items or sensible defaults
        if detection_items is None:
            detection_items = {
                'detect_no_safety_vest_or_helmet': True,
                'detect_near_machinery_or_vehicle': True,
                'detect_in_restricted_area': True,
                'detect_in_utility_pole_restricted_area': True,
                'detect_machinery_close_to_pole': True,
            }

        from src.danger_detector import DangerDetector

        self._detector = DangerDetector(detection_items)
        self.logger.info('Initialized danger detector')
