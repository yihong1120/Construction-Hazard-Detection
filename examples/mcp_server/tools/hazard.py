from __future__ import annotations

import logging
from typing import cast

from examples.mcp_server.schemas import DetectionLikeDict
from examples.mcp_server.schemas import HazardResponse
from src.danger_detector import DangerDetector
from src.utils import Utils


class HazardTools:
    """Tools for detecting safety violations and generating warning
    polygons.

    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._detector = None

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

            # Normalize detections to expected format if provided as dicts
            norm_detections: list[list[float]]
            if detections and isinstance(detections[0], dict):
                norm: list[list[float]] = []
                det_dicts: list[DetectionLikeDict] = cast(
                    list[DetectionLikeDict], detections,
                )
                for d in det_dicts:
                    # Expected keys:
                    # - bbox [x1,y1,x2,y2]
                    # - confidence/conf
                    # - class_/cls/(class fallback)
                    bbox_any = (
                        d['bbox']
                        if 'bbox' in d
                        else (d['box'] if 'box' in d else None)
                    )
                    if (
                        not isinstance(bbox_any, (list, tuple))
                        or len(bbox_any) < 4
                    ):
                        continue
                    x1, y1, x2, y2 = bbox_any[:4]
                    # confidence (use object-typed intermediary for mypy)
                    conf_any: object | None
                    if 'confidence' in d:
                        conf_any = cast(object, d['confidence'])
                    elif 'conf' in d:
                        conf_any = cast(object, d['conf'])
                    else:
                        conf_any = cast(dict[str, object], d).get(
                            'confidence', 0.0,
                        )  # fallback
                    # convert confidence safely
                    if isinstance(
                        conf_any,
                        (int, float, str),
                    ):
                        try:
                            conf_f = float(conf_any)
                        except Exception:
                            conf_f = 0.0
                    else:
                        conf_f = 0.0
                    # class index (prefer typed keys; allow 'class' fallback)
                    cls_any: object | None
                    if 'class_' in d:
                        cls_any = cast(object, d['class_'])
                    elif 'cls' in d:
                        cls_any = cast(object, d['cls'])
                    else:
                        cls_any = cast(dict[str, object], d).get(
                            'class',
                            0,
                        )
                    if isinstance(cls_any, (int, float, str)):
                        try:
                            cls_idx_i = int(cls_any)
                        except Exception:
                            cls_idx_i = 0
                    else:
                        cls_idx_i = 0
                    norm.append([
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                        conf_f,
                        float(cls_idx_i),
                    ])
                norm_detections = norm
            else:
                norm_detections = cast(list[list[float]], detections)

            # Perform violation detection
            result = self._detector.detect_danger(
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
        warnings: dict[str, dict[str, int]],
        is_working_hour: bool,
    ) -> dict[str, dict[str, int]]:
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
