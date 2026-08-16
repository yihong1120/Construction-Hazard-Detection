from __future__ import annotations

import logging
from typing import cast

from examples.mcp_server.schemas import DetectionLikeDict
from examples.mcp_server.schemas import HazardResponse
from src.danger_detector import DangerDetector
from src.utils import Utils
from src.warning_types import MutableWarnings
from src.warning_types import Warnings


class HazardTools:
    """Tools for detecting safety violations and generating warning
    polygons.

    """

    def __init__(self) -> None:
        """Initialise lazy hazard detection resources."""
        self.logger = logging.getLogger(__name__)
        self._detector: DangerDetector | None = None

    async def detect_violations(
        self,
        detections: list[list[float]] | list[DetectionLikeDict],
        # Optional contextual parameters (accepted for compatibility)
        image_width: int | None = None,
        image_height: int | None = None,
        working_hour_only: bool | None = None,
        site_config: dict | None = None,
        detection_items: dict[str, bool] | None = None,
    ) -> HazardResponse:
        """Analyse detection results for safety violations.

        Args:
            detections: Either raw lists of ``[x1, y1, x2, y2, conf, cls]`` or
                object dictionaries with keys such as ``bbox``/``box``,
                ``confidence``/``conf`` and ``class``/``cls``.
            image_width: Optional image width used for contextual checks.
            image_height: Optional image height used for contextual checks.
            working_hour_only: When provided, may be used to filter warnings
                to working hours only.
            site_config: Optional site-specific configuration.
            detection_items: Fine-grained toggles for individual safety checks.

        Returns:
            dict[str, Any]: A mapping with ``warnings``, ``cone_polygons``,
            ``pole_polygons`` and a ``meta`` section.
        """
        try:
            # Initialise detector if needed
            if self._detector is None:
                await self._init_detector(detection_items)
            detector = self._detector
            assert detector is not None

            norm_detections = self._normalise_detections(detections)

            # Perform violation detection
            result = detector.detect_danger(
                norm_detections,
            )
            warnings, cone_polygons, pole_polygons = result

            return {
                'warnings': warnings,
                'cone_polygons': cone_polygons,
                'pole_polygons': pole_polygons,
                'meta': {
                    'image_width': image_width,
                    'image_height': image_height,
                    'working_hour_only': working_hour_only,
                    'site_config_provided': bool(site_config),
                },
            }

        except Exception as e:
            self.logger.error(f"Violation detection failed: {e}")
            raise

    @staticmethod
    def _normalise_detections(
        detections: list[list[float]] | list[DetectionLikeDict],
    ) -> list[list[float]]:
        """Convert dictionary detections to the detector's row format."""
        if not detections or not isinstance(detections[0], dict):
            return cast(list[list[float]], detections)
        return [
            normalized
            for detection in cast(list[DetectionLikeDict], detections)
            if (
                normalized := HazardTools._normalise_detection(detection)
            ) is not None
        ]

    @staticmethod
    def _normalise_detection(
        detection: DetectionLikeDict,
    ) -> list[float] | None:
        """Return one normalized detection or skip an incomplete object."""
        bbox = detection['bbox'] if 'bbox' in detection else None
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None
        if 'confidence' not in detection or 'class_' not in detection:
            return None
        x1, y1, x2, y2 = bbox[:4]
        return [
            float(x1),
            float(y1),
            float(x2),
            float(y2),
            HazardTools._coerce_float(detection['confidence']),
            float(HazardTools._coerce_int(detection['class_'])),
        ]

    @staticmethod
    def _coerce_float(value: object) -> float:
        """Convert supported numeric input with the legacy zero fallback."""
        if not isinstance(value, (int, float, str)):
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _coerce_int(value: object) -> int:
        """Convert supported class input with the legacy zero fallback."""
        if not isinstance(value, (int, float, str)):
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

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

        self._detector = DangerDetector(detection_items)
        self.logger.info('Initialized danger detector')

    async def filter_warnings_by_working_hour(
        self,
        warnings: Warnings,
        is_working_hour: bool,
    ) -> MutableWarnings:
        """Filter warnings based on working hours.

        Args:
            warnings: Mapping of warning types and their parameters.
            is_working_hour: Whether the current time is within working hours.

        Returns:
            A filtered warnings mapping.
        """
        return Utils.filter_warnings_by_working_hour(
            warnings,
            is_working_hour,
        )

    async def should_notify(
        self,
        timestamp: int,
        last_notification_time: int,
        cooldown_period: int = 300,
    ) -> bool:
        """Check whether a notification should be sent based on cooldown.

        Args:
            timestamp: Current timestamp.
            last_notification_time: Timestamp of the last notification.
            cooldown_period: Cooldown period in seconds.

        Returns:
            ``True`` if a notification should be sent.
        """
        return Utils.should_notify(
            timestamp,
            last_notification_time,
            cooldown_period,
        )
