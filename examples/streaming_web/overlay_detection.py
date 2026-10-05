"""Normalise detector geometry and identify warning targets."""
from __future__ import annotations

from examples.shared.class_colors import normalize_color
from examples.streaming_web.overlay_models import DetectionOverlay
from examples.streaming_web.overlay_models import TrackingDetection
from examples.streaming_web.overlay_models import WarningBoundingBox
from examples.streaming_web.overlay_models import WarningDetails
from examples.streaming_web.overlay_models import WarningPayload


def _parse_tracking_detection(
    item: TrackingDetection,
    frame_width: int,
    frame_height: int,
    warning_classes: set[str],
    warning_targets: set[tuple[int, int, int, int]],
    class_metadata: list[dict] | None = None,
) -> DetectionOverlay | None:
    """Parse one tracked YOLO row into clipped overlay coordinates.

    Args:
        item: Trusted tracking row with geometry, confidence, class, and ID.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.
        warning_classes: Classes that should receive warning presentation.
        warning_targets: Explicit person boxes associated with warnings.

    Returns:
        Normalised detection, or ``None`` for a degenerate bounding box.
    """
    x1, y1, x2, y2 = (float(item[i]) for i in range(4))
    confidence = float(item[4])
    class_id = int(float(item[5]))
    metadata = next(
        (c for c in (class_metadata or []) if c.get('id') == class_id), {},
    )
    class_name = metadata.get('code', f'class-{class_id}')
    if _looks_normalized([x1, y1, x2, y2]):
        x1 *= frame_width
        x2 *= frame_width
        y1 *= frame_height
        y2 *= frame_height
    bbox = _clip_bbox(
        int(round(x1)),
        int(round(y1)),
        int(round(x2)),
        int(round(y2)),
        frame_width,
        frame_height,
    )
    if bbox is None:
        return None
    track_id = int(item[6])
    return DetectionOverlay(
        class_name=class_name,
        display_name=metadata.get('display_name'),
        color=normalize_color(metadata.get('color')),
        confidence=confidence,
        bbox=bbox,
        track_id=str(track_id) if track_id >= 0 else None,
        is_warning=(
            class_name in warning_classes
            or _is_warning_target(class_name, bbox, warning_targets)
        ),
    )


def _clip_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    """Clip a bounding box to frame bounds.

    Args:
        x1: First horizontal coordinate.
        y1: First vertical coordinate.
        x2: Second horizontal coordinate.
        y2: Second vertical coordinate.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.

    Returns:
        Ordered clipped pixel box, or ``None`` when it has no area.
    """
    left = max(0, min(x1, x2, frame_width - 1))
    top = max(0, min(y1, y2, frame_height - 1))
    right = max(0, min(max(x1, x2), frame_width - 1))
    bottom = max(0, min(max(y1, y2), frame_height - 1))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _looks_normalized(values: list[float]) -> bool:
    """Determine whether all coordinates appear normalised to zero through one.

    Args:
        values: Coordinates to inspect.

    Returns:
        ``True`` when every value is within the inclusive unit interval.
    """
    return all(0.0 <= value <= 1.0 for value in values)


def _warning_targets(
    warnings: WarningPayload,
    frame_width: int,
    frame_height: int,
) -> set[tuple[int, int, int, int]]:
    """Return person boxes that should be highlighted as warning targets.

    Args:
        warnings: Decoded detector-warning payload.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.

    Returns:
        Clipped explicit person boxes for active proximity warnings.
    """
    targets: set[tuple[int, int, int, int]] = set()
    proximity_keys = (
        ('warning_close_to_machinery', 'machinery'),
        ('warning_close_to_vehicle', 'vehicle'),
    )
    for key, _ in proximity_keys:
        warning = warnings.get(key)
        if warning is None or _warning_count(warning) <= 0:
            continue
        targets.update(
            _explicit_person_warning_bboxes(
                warning,
                frame_width,
                frame_height,
            ),
        )

    return targets


def _explicit_person_warning_bboxes(
    warning: WarningDetails,
    frame_width: int,
    frame_height: int,
) -> set[tuple[int, int, int, int]]:
    """Extract explicit person boxes from one warning payload.

    Args:
        warning: Warning details containing optional person boxes.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.

    Returns:
        Set of clipped valid person boxes.
    """
    targets = set()
    for bbox_raw in warning.get('person_bboxes', []):
        bbox = _normalise_warning_bbox(bbox_raw, frame_width, frame_height)
        if bbox is not None:
            targets.add(bbox)
    return targets


def _normalise_warning_bbox(
    bbox_raw: WarningBoundingBox,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    """Normalise a warning box to clipped pixel coordinates.

    Args:
        bbox_raw: Detector box in normalised or pixel coordinates.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.

    Returns:
        Ordered clipped pixel box, or ``None`` when it has no area.
    """
    x1, y1, x2, y2 = (float(bbox_raw[i]) for i in range(4))
    if _looks_normalized([x1, y1, x2, y2]):
        x1 *= frame_width
        x2 *= frame_width
        y1 *= frame_height
        y2 *= frame_height
    return _clip_bbox(
        int(round(x1)),
        int(round(y1)),
        int(round(x2)),
        int(round(y2)),
        frame_width,
        frame_height,
    )


def _is_warning_target(
    class_name: str,
    bbox: tuple[int, int, int, int],
    warning_targets: set[tuple[int, int, int, int]],
) -> bool:
    """Determine whether a detection matches an explicit warning target.

    Args:
        class_name: Canonical detection class name.
        bbox: Clipped pixel bounding box for the detection.
        warning_targets: Explicit warning person boxes.

    Returns:
        ``True`` when a person overlaps a warning target sufficiently.
    """
    if class_name != 'person':
        return False
    return any(_bbox_iou(bbox, target) >= 0.85 for target in warning_targets)


def _bbox_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    """Calculate intersection over union for two pixel boxes.

    Args:
        first: First left, top, right, bottom pixel box.
        second: Second left, top, right, bottom pixel box.

    Returns:
        Intersection-over-union score between zero and one.
    """
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _warning_classes(warnings: WarningPayload) -> set[str]:
    """Return detection classes that should receive warning presentation.

    Args:
        warnings: Decoded detector-warning payload.

    Returns:
        Canonical classes implicated by active warning types.
    """
    classes: set[str] = set()
    if 'warning_no_hardhat' in warnings:
        classes.add('no_helmet')
    if 'warning_no_safety_vest' in warnings:
        classes.add('no_safety_vest')
    if 'warning_people_in_controlled_area' in warnings:
        classes.add('person')
    if 'warning_people_in_utility_pole_controlled_area' in warnings:
        classes.add('person')
    if 'detect_machinery_close_to_pole' in warnings:
        classes.update({'machinery', 'vehicle'})
    return classes


def has_warning(warnings: WarningPayload) -> bool:
    """Determine whether warning metadata contains an active warning.

    Args:
        warnings: Decoded detector-warning payload.

    Returns:
        ``True`` when any warning count is positive.
    """
    return any(_warning_count(value) > 0 for value in warnings.values())


def _warning_count(value: WarningDetails) -> int:
    """Extract the active warning count from warning details.

    Args:
        value: Typed warning details.

    Returns:
        Active warning count.
    """
    return value['count']
