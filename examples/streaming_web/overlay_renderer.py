from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from functools import lru_cache

import cv2
import numpy as np

from examples.shared.class_colors import normalize_color
from examples.shared.overlay_visibility import object_visibility
from examples.streaming_web.overlay_detection import (
    _parse_tracking_detection as _parse_tracking_detection,
)
from examples.streaming_web.overlay_detection import (
    _warning_classes as _warning_classes,
)
from examples.streaming_web.overlay_detection import (
    _warning_count as _warning_count,
)
from examples.streaming_web.overlay_detection import (
    _warning_targets as _warning_targets,
)
from examples.streaming_web.overlay_detection import has_warning as has_warning
from examples.streaming_web.overlay_labels import (
    _color_for_class as _color_for_class,
)
from examples.streaming_web.overlay_labels import (
    _format_label as _format_label,
)
from examples.streaming_web.overlay_labels import _rgb_to_bgr as _rgb_to_bgr
from examples.streaming_web.overlay_labels import (
    DETECTION_WARNING_KEYS as DETECTION_WARNING_KEYS,
)
from examples.streaming_web.overlay_labels import (
    normalise_label_language as normalise_label_language,
)
from examples.streaming_web.overlay_labels import (
    SUPPORTED_LABEL_LANGUAGES as SUPPORTED_LABEL_LANGUAGES,
)
from examples.streaming_web.overlay_labels import (
    WARNING_LABELS as WARNING_LABELS,
)
from examples.streaming_web.overlay_labels import WARNING_RGB as WARNING_RGB
from examples.streaming_web.overlay_models import DetectionOverlay
from examples.streaming_web.overlay_models import OverlaySnapshot
from examples.streaming_web.overlay_models import PolygonCollection
from examples.streaming_web.overlay_models import PolygonCoordinates
from examples.streaming_web.overlay_models import TrackingDetections
from examples.streaming_web.overlay_models import WarningPayload
from examples.streaming_web.overlay_text import _draw_label_text
from examples.streaming_web.overlay_text import _measure_label_text
from src.settings.media import OverlaySettings


def normalise_overlay_mode(value: str | None) -> str:
    """Normalise a user-supplied overlay mode to a supported value.

    Args:
        value: Optional raw mode from a request or configuration.

    Returns:
        ``backend`` for recognised truthy overlay modes; otherwise ``none``.
    """
    mode = (value or 'none').strip().lower()
    if mode in {'1', 'true', 'yes', 'on', 'backend', 'annotated'}:
        return 'backend'
    return 'none'


def _parse_polygon_collection(value: str) -> PolygonCollection:
    """Decode one detector polygon collection.

    Args:
        value: JSON polygon collection.

    Returns:
        Decoded polygon coordinate lists.
    """
    return json.loads(value)


def _warning_target_overlays(
    warning_targets: set[tuple[int, int, int, int]],
) -> tuple[DetectionOverlay, ...]:
    """Build direct red person boxes for algorithm-marked warning targets.

    Args:
        warning_targets: Clipped person boxes associated with warnings.

    Returns:
        Synthetic warning detections sorted by bounding box.
    """
    return tuple(
        DetectionOverlay(
            class_name='person', confidence=1.0, bbox=bbox, is_warning=True,
        )
        for bbox in sorted(warning_targets)
    )


def _iter_detections_from_data(
    data: TrackingDetections,
    frame_width: int,
    frame_height: int,
    warning_classes: set[str],
    warning_targets: set[tuple[int, int, int, int]],
    min_confidence: float,
    class_metadata: list[dict] | None = None,
) -> Iterator[DetectionOverlay]:
    """Yield valid detections above the configured confidence threshold.

    Args:
        data: Decoded tracked detection rows.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.
        warning_classes: Classes that should receive warning presentation.
        warning_targets: Explicit person boxes associated with warnings.
        min_confidence: Minimum detection confidence to retain.

    Yields:
        Normalised valid detections that meet the confidence threshold.
    """
    for item in data:
        detection = _parse_tracking_detection(
            item,
            frame_width=frame_width,
            frame_height=frame_height,
            warning_classes=warning_classes,
            warning_targets=warning_targets,
            class_metadata=class_metadata,
        )
        if detection is None or detection.confidence < min_confidence:
            continue
        yield detection


def _warning_counts_from_detections(
    detections: tuple[DetectionOverlay, ...],
) -> dict[str, int]:
    """Count normalised warning detections by warning key.

    Args:
        detections: Normalised detections to inspect.

    Returns:
        Count of detections for each mapped warning key.
    """
    counts: dict[str, int] = {}
    for detection in detections:
        _add_detection_warning_count(counts, detection)
    return counts


def _add_detection_warning_count(
    counts: dict[str, int], detection: DetectionOverlay,
) -> None:
    """Increment the warning count represented by one detection.

    Args:
        counts: Mutable inferred warning-count map.
        detection: Normalised detection to map to a warning key.
    """
    key = DETECTION_WARNING_KEYS.get(detection.class_name)
    if not key:
        return
    counts[key] = counts.get(key, 0) + 1


def _format_warning_summary_line(
    key: str, count: int, labels: dict[str, str],
) -> str:
    """Format one translated warning-summary line.

    Args:
        key: Canonical warning key.
        count: Active count for the warning.
        labels: Translation map for the selected language.

    Returns:
        Localised label with a count suffix when greater than one.
    """
    label = labels.get(key) or key
    suffix = f' x{count}' if count > 1 else ''
    return f'{label}{suffix}'


def _draw_detection(
    frame: np.ndarray,
    detection: DetectionOverlay,
    label_language: str,
    box_thickness: int,
    draw_label: bool = True,
) -> None:
    """Draw one detection box and optional label.

    Args:
        frame: Mutable BGR image array to annotate.
        detection: Normalised detection to draw.
        label_language: Canonical label language.
        box_thickness: Requested bounding-box line thickness.
        draw_label: Whether a label badge should be drawn.
    """
    if not object_visibility(detection.class_name)['show_box']:
        return
    x1, y1, x2, y2 = detection.bbox
    rgb = (
        WARNING_RGB
        if detection.is_warning
        else _color_for_class(detection.class_name)
    )
    color = normalize_color(detection.color)
    if color:
        red, green, blue = bytes.fromhex(color[1:])
        rgb = (red, green, blue)
    bgr = _rgb_to_bgr(rgb)
    thickness = max(1, int(box_thickness))
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        bgr,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    if draw_label:
        _draw_label(frame, detection, rgb, label_language)


def _draw_label(
    frame: np.ndarray,
    detection: DetectionOverlay,
    rgb: tuple[int, int, int],
    label_language: str,
) -> None:
    """Draw a filled localised label badge for one detection.

    Args:
        frame: Mutable BGR image array to annotate.
        detection: Normalised detection to label.
        rgb: Badge colour in RGB order.
        label_language: Canonical label language.
    """
    x1, y1, x2, _ = detection.bbox
    label = _format_label(detection, label_language)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = _font_scale(frame)
    thickness = max(1, _line_thickness(frame) - 1)
    padding_x = 6
    padding_y = 3
    text_width, text_height, baseline = _measure_label_text(
        label, frame, font, scale, thickness,
    )
    label_width = text_width + padding_x * 2
    label_height = text_height + baseline + padding_y * 2
    label_x1 = x1
    label_y1 = y1 - label_height
    if label_y1 < 0:
        label_y1 = y1
    label_x2 = min(frame.shape[1] - 1, label_x1 + label_width)
    if label_x2 >= frame.shape[1] - 1:
        label_x1 = max(0, frame.shape[1] - 1 - label_width)
        label_x2 = frame.shape[1] - 1
    label_y2 = min(frame.shape[0] - 1, label_y1 + label_height)
    label_roi = frame[label_y1: label_y2 + 1, label_x1: label_x2 + 1]
    if label_roi.size:
        cv2.rectangle(
            frame,
            (label_x1, label_y1),
            (label_x2, label_y2),
            _rgb_to_bgr(rgb),
            thickness=-1,
        )
    text_color = (255, 255, 255)
    text_origin = (label_x1 + padding_x, label_y1 + padding_y + text_height)
    _draw_label_text(
        frame,
        label,
        text_origin,
        font,
        scale,
        text_color,
        thickness,
        (label_x1, label_y1, label_x2, label_y2),
    )


def _draw_polygon_data(
    frame: np.ndarray,
    polygon_specs: tuple[
        tuple[
            PolygonCollection,
            tuple[int, int, int],
            tuple[int, int, int],
            float,
        ],
        ...,
    ],
) -> None:
    """Draw polygons from already-decoded data.

    Args:
        frame: Mutable BGR image array to annotate.
        polygon_specs: Decoded polygons with fill colour, border colour, and
            alpha.
    """
    for polygon_data, fill_rgb, stroke_rgb, fill_alpha in polygon_specs:
        polygons = _normalised_polygons_from_data(
            polygon_data, frame.shape[1], frame.shape[0],
        )
        if not polygons:
            continue
        _draw_polygon_rois(frame, polygons, fill_rgb, stroke_rgb, fill_alpha)


def _draw_polygon_rois(
    frame: np.ndarray,
    polygons: tuple[np.ndarray, ...],
    fill_rgb: tuple[int, int, int],
    stroke_rgb: tuple[int, int, int],
    fill_alpha: float,
) -> None:
    """Draw polygon fills using bounded ROI copies.

    Args:
        frame: Mutable BGR image array to annotate.
        polygons: Clipped polygon points in pixel coordinates.
        fill_rgb: Polygon fill colour in RGB order.
        stroke_rgb: Polygon outline colour in RGB order.
        fill_alpha: Fill opacity between zero and one.
    """
    fill_bgr = _rgb_to_bgr(fill_rgb)
    stroke_bgr = _rgb_to_bgr(stroke_rgb)
    alpha = max(0.0, min(1.0, fill_alpha))
    for points in polygons:
        _blend_polygon_fill_roi(frame, points, fill_bgr, alpha)
        cv2.polylines(
            frame, [points], isClosed=True, color=stroke_bgr, thickness=3,
        )


def _blend_polygon_fill_roi(
    frame: np.ndarray,
    points: np.ndarray,
    fill_bgr: tuple[int, int, int],
    alpha: float,
) -> None:
    """Blend a polygon fill into the smallest affected frame region.

    Args:
        frame: Mutable BGR image array to annotate.
        points: Clipped polygon points in pixel coordinates.
        fill_bgr: Fill colour in OpenCV BGR order.
        alpha: Fill opacity between zero and one.
    """
    if alpha <= 0:
        return
    x, y, width, height = cv2.boundingRect(points)
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame.shape[1], x + width)
    y2 = min(frame.shape[0], y + height)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    if not roi.size:
        return
    local_points = points.copy()
    local_points[:, 0] -= x1
    local_points[:, 1] -= y1
    fill = np.empty_like(roi)
    fill[:, :] = fill_bgr
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [local_points], 255)
    blended = cv2.addWeighted(fill, alpha, roi, 1 - alpha, 0)
    cv2.copyTo(blended, mask, roi)


def _normalised_polygons_for_overlay(
    polygons_json: str, frame_width: int, frame_height: int,
) -> tuple[np.ndarray, ...]:
    """Return cached pixel polygons parsed from JSON.

    Args:
        polygons_json: JSON polygon collection.
        frame_width: Target frame width in pixels.
        frame_height: Target frame height in pixels.

    Returns:
        Valid clipped polygon arrays cached by geometry and frame size.
    """
    data = _parse_polygon_collection(polygons_json)
    polygons: list[np.ndarray] = []
    for polygon in data:
        points = _normalise_polygon(polygon, frame_width, frame_height)
        if points is not None:
            polygons.append(points)
    return tuple(polygons)


def _normalised_polygons_from_data(
    data: PolygonCollection, frame_width: int, frame_height: int,
) -> tuple[np.ndarray, ...]:
    """Normalise pixel polygons from decoded data.

    Args:
        data: Decoded polygon collection.
        frame_width: Target frame width in pixels.
        frame_height: Target frame height in pixels.

    Returns:
        Valid clipped polygon arrays.
    """
    polygons: list[np.ndarray] = []
    for polygon in data:
        points = _normalise_polygon(polygon, frame_width, frame_height)
        if points is not None:
            polygons.append(points)
    return tuple(polygons)


def _normalise_polygon(
    polygon: PolygonCoordinates, frame_width: int, frame_height: int,
) -> np.ndarray | None:
    """Normalise one polygon to clipped pixel coordinates.

    Args:
        polygon: Polygon coordinates in normalised or pixel form.
        frame_width: Target frame width in pixels.
        frame_height: Target frame height in pixels.

    Returns:
        Integer OpenCV polygon points, or ``None`` when no points are valid.
    """
    points = [(float(point[0]), float(point[1])) for point in polygon]
    if _points_look_normalized(points):
        points = [(x * frame_width, y * frame_height) for x, y in points]
    clipped = [
        (
            max(0, min(int(round(x)), frame_width - 1)),
            max(0, min(int(round(y)), frame_height - 1)),
        )
        for x, y in points
    ]
    return np.array(clipped, dtype=np.int32)


def _points_look_normalized(points: list[tuple[float, float]]) -> bool:
    """Determine whether polygon points appear normalised to zero through one.

    Args:
        points: Polygon points to inspect.

    Returns:
        ``True`` when every point lies in the inclusive unit square.
    """
    return all((0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in points))


def _line_thickness(frame: np.ndarray) -> int:
    """Calculate box line thickness for a frame size.

    Args:
        frame: Target BGR image array.

    Returns:
        Bounded line thickness in pixels.
    """
    min_side = min(frame.shape[:2])
    return max(2, min(4, round(min_side / 360)))


def _font_scale(frame: np.ndarray) -> float:
    """Calculate OpenCV font scale for a frame size.

    Args:
        frame: Target BGR image array.

    Returns:
        Bounded OpenCV font scale.
    """
    min_side = min(frame.shape[:2])
    return max(0.45, min(0.75, min_side / 1000))


__all__ = [
    'OverlayRenderer',
    'normalise_overlay_mode',
    'normalise_label_language',
    'SUPPORTED_LABEL_LANGUAGES',
    'has_warning',
    'PolygonCollection',
    'TrackingDetections',
    'WarningPayload',
]


class OverlayRenderer:

    def __init__(self, settings: OverlaySettings | None = None):
        self.settings = settings or OverlaySettings()
        self.polygons = lru_cache(maxsize=self.settings.parse_cache_size)(
            _normalised_polygons_for_overlay,
        )

    def render(
        self, snapshot: OverlaySnapshot, label_language: str = 'en',
    ) -> np.ndarray:
        """Copy the source once and render it using fixed settings."""
        source = snapshot.frame
        if (
            normalise_overlay_mode(self.settings.mode) != 'backend'
            or source is None
            or source.size == 0
        ):
            return source
        frame = source.copy()
        language = normalise_label_language(label_language)
        _draw_polygon_data(
            frame,
            (
                (snapshot.cone_polys, (255, 64, 129), (233, 30, 99), 0.4),
                (snapshot.pole_polys, (68, 138, 255), (68, 138, 255), 0.4),
            ),
        )
        counts = self._draw_detections_from_data(frame, snapshot, language)
        if self.settings.draw_warning_summary:
            self._draw_warning_summary(
                frame,
                snapshot.warnings,
                language,
                detection_warning_counts=counts,
            )
        return frame

    def _draw_detections_from_data(
        self,
        frame: np.ndarray,
        snapshot: OverlaySnapshot,
        label_language: str,
    ) -> dict[str, int]:
        data = snapshot.track_data or []
        warnings = snapshot.warnings
        class_metadata = snapshot.class_metadata
        frame_height, frame_width = frame.shape[:2]
        min_confidence = self.settings.min_confidence
        warning_targets = _warning_targets(warnings, frame_width, frame_height)
        warning_classes = _warning_classes(warnings)
        warning_counts: dict[str, int] = {}
        label_count = 0
        for detection in _iter_detections_from_data(
            data,
            frame_width=frame_width,
            frame_height=frame_height,
            warning_classes=warning_classes,
            warning_targets=warning_targets,
            min_confidence=min_confidence,
            class_metadata=class_metadata,
        ):
            label_count = self._draw_detection_and_update_counts(
                frame, detection, label_language, label_count, warning_counts,
            )
        for detection in _warning_target_overlays(warning_targets):
            metadata = next(
                (
                    c
                    for c in class_metadata or []
                    if c.get('code') == detection.class_name
                ),
                {},
            )
            detection = replace(
                detection,
                display_name=metadata.get('display_name'),
                color=normalize_color(metadata.get('color')),
            )
            label_count = self._draw_detection_and_update_counts(
                frame, detection, label_language, label_count, warning_counts,
            )
        return warning_counts

    def _draw_detection_and_update_counts(
        self,
        frame: np.ndarray,
        detection: DetectionOverlay,
        label_language: str,
        label_count: int,
        warning_counts: dict[str, int],
    ) -> int:
        box_thickness = self.settings.box_thickness
        draw_label = self._should_draw_label(detection, label_count)
        _draw_detection(
            frame,
            detection,
            label_language,
            box_thickness,
            draw_label=draw_label,
        )
        _add_detection_warning_count(warning_counts, detection)
        return label_count + int(draw_label)

    def _warning_summary_lines(
        self,
        warnings: WarningPayload,
        label_language: str,
        detections: tuple[DetectionOverlay, ...] = (),
        detection_warning_counts: dict[str, int] | None = None,
    ) -> list[str]:
        language = normalise_label_language(label_language)
        labels = WARNING_LABELS.get(language, WARNING_LABELS['en'])
        lines: list[str] = []
        emitted_keys: set[str] = set()
        for key, value in warnings.items():
            count = _warning_count(value)
            if count <= 0:
                continue
            lines.append(_format_warning_summary_line(key, count, labels))
            emitted_keys.add(key)
            if len(lines) >= self.settings.max_warning_summary_items:
                return lines
        inferred_counts = (
            detection_warning_counts
            if detection_warning_counts is not None
            else _warning_counts_from_detections(detections)
        )
        for key, count in inferred_counts.items():
            if key in emitted_keys:
                continue
            lines.append(_format_warning_summary_line(key, count, labels))
            if len(lines) >= self.settings.max_warning_summary_items:
                break
        return lines

    def _draw_warning_summary(
        self,
        frame: np.ndarray,
        warnings: WarningPayload,
        label_language: str,
        detections: tuple[DetectionOverlay, ...] = (),
        detection_warning_counts: dict[str, int] | None = None,
    ) -> None:
        lines = self._warning_summary_lines(
            warnings,
            label_language,
            detections=detections,
            detection_warning_counts=detection_warning_counts,
        )
        has_active_warning = bool(lines)
        if not has_active_warning and (not self.settings.draw_warning_status):
            return
        if has_active_warning:
            lines = ['WARNING'] + lines
        else:
            lines = ['OK', 'No active warning']
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = _font_scale(frame)
        thickness = max(1, _line_thickness(frame) - 1)
        padding_x = 10
        padding_y = 8
        gap = 5
        metrics = [
            _measure_label_text(line, frame, font, scale, thickness)
            for line in lines
        ]
        width = max(metric[0] for metric in metrics) + padding_x * 2
        text_block_height = sum(metric[1] + metric[2] for metric in metrics)
        height = text_block_height + gap * (len(lines) - 1) + padding_y * 2
        x1 = max(0, frame.shape[1] - width - 12)
        y1 = max(0, frame.shape[0] - height - 12)
        x2 = min(frame.shape[1] - 1, x1 + width)
        y2 = min(frame.shape[0] - 1, y1 + height)
        roi = frame[y1: y2 + 1, x1: x2 + 1]
        if not roi.size:
            return
        fill = np.empty_like(roi)
        fill[:, :] = (36, 36, 36)
        cv2.addWeighted(fill, 0.78, roi, 0.22, 0, roi)
        border_color = (0, 0, 255) if has_active_warning else (0, 180, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, thickness=2)
        text_rows: list[tuple[str, int, int, int, int]] = []
        cursor_y = y1 + padding_y
        for line, (text_width, text_height, baseline) in zip(
            lines, metrics, strict=False,
        ):
            text_x = max(x1 + padding_x, x2 - padding_x - text_width)
            baseline_y = cursor_y + text_height
            text_rows.append((line, text_x, baseline_y, text_height, baseline))
            cursor_y += text_height + baseline + gap
        for line, text_x, baseline_y, _, _ in text_rows:
            _draw_label_text(
                frame,
                line,
                (text_x, baseline_y),
                font,
                scale,
                (255, 255, 255),
                max(1, thickness),
                (x1, y1, x2, y2),
            )

    def _should_draw_label(
        self, detection: DetectionOverlay, label_count: int,
    ) -> bool:
        if not object_visibility(detection.class_name)['show_label']:
            return False
        if not self.settings.draw_labels:
            return False
        if self.settings.label_warnings_only and (not detection.is_warning):
            return False
        return (
            self.settings.max_labels <= 0
            or label_count < self.settings.max_labels
        )

    def _draw_polygons(
        self,
        frame: np.ndarray,
        polygon_specs: tuple[
            tuple[str, tuple[int, int, int], tuple[int, int, int], float], ...,
        ],
    ) -> None:
        for polygons_json, fill_rgb, stroke_rgb, fill_alpha in polygon_specs:
            polygons = self.polygons(
                polygons_json, frame.shape[1], frame.shape[0],
            )
            if not polygons:
                continue
            _draw_polygon_rois(
                frame, polygons, fill_rgb, stroke_rgb, fill_alpha,
            )
