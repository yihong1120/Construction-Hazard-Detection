from __future__ import annotations

from collections.abc import Sequence
from typing import cast


def calculate_polygon_area(polygon_points: Sequence[Sequence[float]]) -> dict:
    """Calculate polygon area using the shoelace formula."""
    if len(polygon_points) < 3:
        area = 0.0
    else:
        total = sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(
                polygon_points,
                (*polygon_points[1:], polygon_points[0]),
                strict=True,
            )
        )
        area = abs(total) / 2.0
    return {
        'success': True,
        'area': area,
        'points_count': len(polygon_points),
        'message': f'Polygon area calculated: {area:.2f} square units',
    }


def point_in_polygon(
    point: Sequence[float], polygon_points: Sequence[Sequence[float]],
) -> dict:
    """Return whether a point lies within a polygon using ray casting."""
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(
        polygon_points,
        (*polygon_points[1:], polygon_points[0]) if polygon_points else (),
        strict=True,
    ):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (
            y2 - y1 + 1e-12
        ) + x1:
            inside = not inside
    is_inside = inside if len(polygon_points) >= 3 else False
    return {
        'success': True,
        'point': point,
        'is_inside': is_inside,
        'polygon_points': polygon_points,
        'message': (
            f"Point {point} is "
            f"{'inside' if is_inside else 'outside'} the polygon"
        ),
    }


def bbox_intersection(
    bbox1: Sequence[float], bbox2: Sequence[float],
) -> dict:
    """Calculate intersection area and IoU for two xyxy boxes."""
    def normalise(box: Sequence[float]) -> tuple[float, float, float, float]:
        """Perform normalise.

        Args:
            box: Value used by this callable.

        Returns:
            The callable result.
        """
        x1, y1, x2, y2 = box
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    ax1, ay1, ax2, ay2 = normalise(bbox1)
    bx1, by1, bx2, by2 = normalise(bbox2)
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection_area = width * height
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection_area
    iou = intersection_area / union if union else 0.0
    return {
        'success': True,
        'bbox1': bbox1,
        'bbox2': bbox2,
        'intersection_area': intersection_area,
        'iou': iou,
        'message': f'Bboxes intersection: {intersection_area:.2f} area, {iou:.3f} IoU',
    }


def _is_number(value: object) -> bool:
    """Perform is number.

    Args:
        value: Value used by this callable.

    Returns:
        The callable result.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validation_errors(
    detection: object, index: int, image_width: int, image_height: int,
) -> list[str]:
    """Perform validation errors.

    Args:
        detection: Value used by this callable.
        index: Value used by this callable.
        image_width: Value used by this callable.
        image_height: Value used by this callable.

    Returns:
        The callable result.
    """
    if not isinstance(detection, dict):
        return [f'[{index}] detection must be an object/dict']
    bbox = detection.get('bbox', detection.get('box'))
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return [f"[{index}] 'bbox' must be a list of 4 numbers [x1,y1,x2,y2]"]
    if not all(_is_number(value) for value in bbox):
        return [f"[{index}] 'bbox' values must be numbers"]
    x1, y1, x2, y2 = (float(cast(float | int, value)) for value in bbox)
    errors: list[str] = []
    if not all(0 <= value <= 1 for value in (x1, y1, x2, y2)):
        for name, value, maximum in (
            ('x1', x1, image_width), ('x2', x2, image_width),
            ('y1', y1, image_height), ('y2', y2, image_height),
        ):
            if not 0 <= value <= maximum:
                errors.append(f'[{index}] {name} out of range [0,{maximum}]')
    if x2 <= x1 or y2 <= y1:
        errors.append(f'[{index}] bbox has non-positive size: {bbox}')
    for field in ('confidence', 'conf'):
        if field in detection and not _is_number(detection[field]):
            errors.append(f"[{index}] '{field}' must be a number")
    for field in ('class', 'cls'):
        if field in detection and not isinstance(detection[field], int):
            errors.append(f"[{index}] '{field}' must be an integer")
    return errors


def validate_detection_data(
    detections: Sequence[object], image_width: int, image_height: int,
) -> dict:
    """Validate detection bounding boxes and optional metadata."""
    errors = [
        error
        for index, detection in enumerate(detections)
        for error in _validation_errors(
            detection, index, image_width, image_height,
        )
    ]
    return {
        'success': True,
        'is_valid': not errors,
        'detections_count': len(detections),
        'validation_errors': errors,
        'image_size': [image_width, image_height],
        'message': (
            f"Validation {'passed' if not errors else 'failed'}: "
            f'{len(errors)} errors found'
        ),
    }
