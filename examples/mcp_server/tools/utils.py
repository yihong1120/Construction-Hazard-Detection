from __future__ import annotations

import logging
from collections.abc import Sequence
from math import sqrt
from typing import cast

from src.utils import Utils


def _clip_coordinate(value: float, minimum: float, maximum: float) -> float:
    """Clamp a coordinate to an inclusive range."""
    return max(minimum, min(maximum, value))


def _is_number(value: object) -> bool:
    """Return whether a value is a non-boolean number."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _in_range(value: float, minimum: float, maximum: float) -> bool:
    """Return whether a value is inside an inclusive range."""
    return minimum <= value <= maximum


class UtilsTools:
    """Tools for geometry operations and general utilities."""

    def __init__(self) -> None:
        """Initialise lazy utility resources."""
        self.logger = logging.getLogger(__name__)
        self._utils: Utils | None = None

    async def calculate_polygon_area(
        self,
        polygon_points: Sequence[Sequence[float]],
    ) -> dict:
        """Calculate the area of a polygon.

        Args:
            polygon_points: List of ``[x, y]`` coordinate pairs.

        Returns:
            dict[str, Any]: A mapping with the computed area and metadata.
        """
        try:
            # Calculate area using the shoelace formula directly
            if not polygon_points or len(polygon_points) < 3:
                area = 0.0
            else:
                n = len(polygon_points)
                s = 0.0
                for i in range(n):
                    x1, y1 = polygon_points[i]
                    x2, y2 = polygon_points[(i + 1) % n]
                    s += x1 * y2 - x2 * y1
                area = abs(s) / 2.0

            return {
                'success': True,
                'area': area,
                'points_count': len(polygon_points),
                'message': f"Polygon area calculated: {area:.2f} square units",
            }

        except Exception as e:
            self.logger.error(f"Failed to calculate polygon area: {e}")
            raise

    async def point_in_polygon(
        self,
        point: Sequence[float],
        polygon_points: Sequence[Sequence[float]],
    ) -> dict:
        """Check whether a point is inside a polygon.

        Args:
            point: ``[x, y]`` coordinates of the point.
            polygon_points: List of ``[x, y]`` coordinate pairs defining the
                polygon.

        Returns:
            dict[str, Any]: A mapping with the result and contextual data.
        """
        try:
            # Ray casting algorithm for point-in-polygon
            x, y = point
            inside = False
            n = len(polygon_points)
            if n >= 3:
                for i in range(n):
                    x1, y1 = polygon_points[i]
                    x2, y2 = polygon_points[(i + 1) % n]
                    # Check if edge crosses the horizontal ray
                    # to the right of the point
                    intersects = ((y1 > y) != (y2 > y)) and (
                        x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
                    )
                    if intersects:
                        inside = not inside
            is_inside = inside

            return {
                'success': True,
                'point': point,
                'is_inside': is_inside,
                'polygon_points': polygon_points,
                'message': (
                    f"Point {point} is {'inside' if is_inside else 'outside'} "
                    'the polygon'
                ),
            }

        except Exception as e:
            self.logger.error(f"Failed to test point in polygon: {e}")
            raise

    async def bbox_intersection(
        self,
        bbox1: Sequence[float],
        bbox2: Sequence[float],
    ) -> dict:
        """Calculate the intersection of two bounding boxes.

        Args:
            bbox1: ``[x1, y1, x2, y2]`` coordinates of the first bounding box.
            bbox2: ``[x1, y1, x2, y2]`` coordinates of the second bounding box.

        Returns:
            dict[str, Any]: Intersection area and IoU
                (intersection-over-union).
        """
        try:
            # Ensure bbox order [x1, y1, x2, y2]
            def _norm(b: Sequence[float]) -> tuple[float, float, float, float]:
                """Return a bounding box in left-top-right-bottom order."""
                x1, y1, x2, y2 = b
                return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

            a1x1, a1y1, a1x2, a1y2 = _norm(bbox1)
            b1x1, b1y1, b1x2, b1y2 = _norm(bbox2)

            inter_x1 = max(a1x1, b1x1)
            inter_y1 = max(a1y1, b1y1)
            inter_x2 = min(a1x2, b1x2)
            inter_y2 = min(a1y2, b1y2)

            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)
            intersection_area = inter_w * inter_h

            area_a = max(0.0, (a1x2 - a1x1)) * max(0.0, (a1y2 - a1y1))
            area_b = max(0.0, (b1x2 - b1x1)) * max(0.0, (b1y2 - b1y1))
            union = area_a + area_b - intersection_area
            iou = (intersection_area / union) if union > 0 else 0.0

            return {
                'success': True,
                'bbox1': bbox1,
                'bbox2': bbox2,
                'intersection_area': intersection_area,
                'iou': iou,
                'message': (
                    f"Bboxes intersection: {intersection_area:.2f} area, "
                    f"{iou:.3f} IoU"
                ),
            }

        except Exception as e:
            self.logger.error(f"Failed to calculate bbox intersection: {e}")
            raise

    async def distance_between_points(
        self,
        point1: Sequence[float],
        point2: Sequence[float],
        metric: str = 'euclidean',
    ) -> dict:
        """Calculate the distance between two points.

        Args:
            point1: ``[x, y]`` coordinates of the first point.
            point2: ``[x, y]`` coordinates of the second point.
            metric: Distance metric ("euclidean", "manhattan", "chebyshev").

        Returns:
            dict[str, Any]: A mapping with the numeric distance and details.
        """
        try:
            # Calculate classic distances inline
            x1, y1 = point1
            x2, y2 = point2
            if metric == 'euclidean':
                distance = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            elif metric == 'manhattan':
                distance = abs(x1 - x2) + abs(y1 - y2)
            elif metric == 'chebyshev':
                distance = max(abs(x1 - x2), abs(y1 - y2))
            else:
                raise ValueError(
                    "Unsupported metric. Use 'euclidean', 'manhattan', or "
                    "'chebyshev'",
                )

            return {
                'success': True,
                'point1': point1,
                'point2': point2,
                'distance': distance,
                'metric': metric,
                'message': f"{metric.capitalize()} distance: {distance:.2f}",
            }

        except Exception as e:
            self.logger.error(f"Failed to calculate distance: {e}")
            raise

    async def create_safety_zone(
        self,
        center_point: list[float],
        radius: float,
        shape: str = 'circle',
    ) -> dict:
        """Create a safety zone around a point.

        Args:
            center_point: ``[x, y]`` coordinates of the centre.
            radius: Radius of the safety zone.
            shape: Shape of the zone ("circle", "square").

        Returns:
            dict[str, Any]: A mapping with the generated zone points.
        """
        try:
            # Provide a simple zone generator without external deps
            cx, cy = center_point
            if shape.lower() == 'circle':
                # Approximate circle with 32-gon (no numpy dependency)
                steps = 32
                zone_points = []
                for i in range(steps):
                    theta = 2.0 * 3.141592653589793 * i / steps
                    zone_points.append([
                        cx + radius * float(__import__('math').cos(theta)),
                        cy + radius * float(__import__('math').sin(theta)),
                    ])
            elif shape.lower() == 'square':
                zone_points = [
                    [cx - radius, cy - radius],
                    [cx + radius, cy - radius],
                    [cx + radius, cy + radius],
                    [cx - radius, cy + radius],
                ]
            else:
                raise ValueError(
                    "Unsupported shape. Use 'circle' or 'square'.",
                )

            return {
                'success': True,
                'center_point': center_point,
                'radius': radius,
                'shape': shape,
                'zone_points': zone_points,
                'message': f"Created {shape} safety zone with radius {radius}",
            }

        except Exception as e:
            self.logger.error(f"Failed to create safety zone: {e}")
            raise

    async def normalize_coordinates(
        self,
        coordinates: list[list[float]],
        image_width: int,
        image_height: int,
        target_format: str = 'yolo',
    ) -> dict:
        """Normalise coordinates to different formats.

        Args:
            coordinates: List of coordinate pairs.
            image_width: Width of the image.
            image_height: Height of the image.
            target_format: Target format ("yolo", "coco", "normalized").

        Returns:
            dict[str, Any]: A mapping with normalised coordinates and meta.
        """
        try:
            if target_format == 'normalized':
                norm = self._normalise_point_coordinates(
                    coordinates,
                    image_width,
                    image_height,
                )
            elif target_format == 'yolo':
                norm = self._normalise_yolo_boxes(
                    coordinates,
                    image_width,
                    image_height,
                )
            elif target_format == 'coco':
                norm = self._normalise_coco_boxes(
                    coordinates,
                    image_width,
                    image_height,
                )
            else:
                raise ValueError(
                    "Unsupported target_format. Use 'yolo', 'coco', or "
                    "'normalized'.",
                )

            return {
                'success': True,
                'original_coordinates': coordinates,
                'normalized_coordinates': norm,
                'image_size': [image_width, image_height],
                'target_format': target_format,
                'message': f"Coordinates normalized to {target_format} format",
            }

        except Exception as e:
            self.logger.error(f"Failed to normalize coordinates: {e}")
            raise

    @staticmethod
    def _normalise_point_coordinates(
        coordinates: list[list[float]],
        image_width: int,
        image_height: int,
    ) -> list[list[float]]:
        """Convert absolute point coordinates to the normalized range."""
        return [
            [
                _clip_coordinate(x / image_width, 0.0, 1.0),
                _clip_coordinate(y / image_height, 0.0, 1.0),
            ]
            for x, y in coordinates
        ]

    @staticmethod
    def _normalise_yolo_boxes(
        coordinates: list[list[float]],
        image_width: int,
        image_height: int,
    ) -> list[list[float]]:
        """Convert xyxy boxes to normalized YOLO centre-width-height boxes."""
        normalized: list[list[float]] = []
        for bbox in coordinates:
            if len(bbox) != 4:
                raise ValueError('YOLO expects [x1,y1,x2,y2] per item')
            x1, y1, x2, y2 = bbox
            normalized.append([
                _clip_coordinate(((x1 + x2) / 2.0) / image_width, 0.0, 1.0),
                _clip_coordinate(((y1 + y2) / 2.0) / image_height, 0.0, 1.0),
                _clip_coordinate(abs(x2 - x1) / image_width, 0.0, 1.0),
                _clip_coordinate(abs(y2 - y1) / image_height, 0.0, 1.0),
            ])
        return normalized

    @staticmethod
    def _normalise_coco_boxes(
        coordinates: list[list[float]],
        image_width: int,
        image_height: int,
    ) -> list[list[float]]:
        """Convert xyxy boxes to clipped COCO left-top-width-height boxes."""
        normalized: list[list[float]] = []
        for bbox in coordinates:
            if len(bbox) != 4:
                raise ValueError('COCO expects [x1,y1,x2,y2] per item')
            x1, y1, x2, y2 = bbox
            normalized.append([
                _clip_coordinate(min(x1, x2), 0.0, float(image_width)),
                _clip_coordinate(min(y1, y2), 0.0, float(image_height)),
                _clip_coordinate(abs(x2 - x1), 0.0, float(image_width)),
                _clip_coordinate(abs(y2 - y1), 0.0, float(image_height)),
            ])
        return normalized

    async def convert_image_format(
        self,
        image_base64: str,
        target_format: str = 'JPEG',
        quality: int = 95,
    ) -> dict:
        """Convert image format and quality.

        Args:
            image_base64: Base64-encoded image.
            target_format: Target format ("JPEG", "PNG", "WEBP").
            quality: Image quality (1–100, for formats that support it).

        Returns:
            dict[str, Any]: A mapping with the converted image and metrics.
        """
        try:
            # Lightweight in-place convert using PIL if available,
            # else passthrough
            converted_base64 = image_base64
            original_size = len(image_base64.encode('utf-8'))
            new_size = original_size
            try:
                from io import BytesIO
                import base64
                from PIL import Image

                img_bytes = base64.b64decode(image_base64)
                with BytesIO(img_bytes) as bio:
                    with Image.open(bio) as img:
                        out = BytesIO()
                        save_kwargs = {}
                        if target_format.upper() == 'JPEG':
                            save_kwargs['quality'] = int(quality)
                            save_kwargs['optimize'] = True
                        img.convert('RGB').save(
                            out, format=target_format.upper(), **save_kwargs,
                        )
                        new_b64 = base64.b64encode(
                            out.getvalue(),
                        ).decode('utf-8')
                        converted_base64 = new_b64
                        new_size = len(out.getvalue())
            except Exception as pil_e:
                # If PIL not available or fails, keep original and log
                self.logger.warning(
                    'PIL conversion failed or unavailable, returning '
                    f'original image: {pil_e}',
                )

            return {
                'success': True,
                'converted_image': converted_base64,
                'original_size': original_size,
                'new_size': new_size,
                'compression_ratio': (
                    original_size / new_size if new_size > 0 else 1.0
                ),
                'target_format': target_format,
                'quality': quality,
                'message': (
                    f"Image converted to {target_format}, size: "
                    f"{original_size} → {new_size} bytes"
                ),
            }

        except Exception as e:
            self.logger.error(f"Failed to convert image format: {e}")
            raise

    async def validate_detection_data(
        self,
        detections: Sequence[object],
        image_width: int,
        image_height: int,
    ) -> dict:
        """Validate detection data format and coordinates.

        Args:
            detections: List of detection objects.
            image_width: Width of the image.
            image_height: Height of the image.

        Returns:
            dict[str, Any]: A mapping with the validation outcome and details.
        """
        try:
            errors: list[str] = []
            for idx, det in enumerate(detections):
                errors.extend(
                    self._detection_validation_errors(
                        det,
                        idx,
                        image_width,
                        image_height,
                    ),
                )

            is_valid = len(errors) == 0
            validation_errors = errors

            return {
                'success': True,
                'is_valid': is_valid,
                'detections_count': len(detections),
                'validation_errors': validation_errors,
                'image_size': [image_width, image_height],
                'message': (
                    f"Validation {'passed' if is_valid else 'failed'}: "
                    f"{len(validation_errors)} errors found"
                ),
            }

        except Exception as e:
            self.logger.error(f"Failed to validate detection data: {e}")
            raise

    @staticmethod
    def _detection_validation_errors(
        detection: object,
        index: int,
        image_width: int,
        image_height: int,
    ) -> list[str]:
        """Return validation messages for one detection object."""
        if not isinstance(detection, dict):
            return [f"[{index}] detection must be an object/dict"]
        bbox = detection.get(
            'bbox',
        ) if 'bbox' in detection else detection.get('box')
        bbox_error = UtilsTools._bbox_validation_error(bbox, index)
        if bbox_error is not None:
            return [bbox_error]
        assert isinstance(bbox, (list, tuple))
        return (
            UtilsTools._bbox_geometry_errors(
                bbox,
                index,
                image_width,
                image_height,
            )
            + UtilsTools._optional_field_errors(detection, index)
        )

    @staticmethod
    def _bbox_validation_error(bbox: object, index: int) -> str | None:
        """Return the first shape or type error for a detection bbox."""
        if bbox is None:
            return f"[{index}] missing 'bbox'/'box'"
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return (
                f"[{index}] 'bbox' must be a list of 4 numbers "
                '[x1,y1,x2,y2]'
            )
        if not all(_is_number(value) for value in bbox):
            return f"[{index}] 'bbox' values must be numbers"
        return None

    @staticmethod
    def _bbox_geometry_errors(
        bbox: Sequence[object],
        index: int,
        image_width: int,
        image_height: int,
    ) -> list[str]:
        """Validate bbox bounds and positive geometry after type checking."""
        x1, y1, x2, y2 = (
            float(cast(float | int, value)) for value in bbox
        )
        errors: list[str] = []
        if not all(0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)):
            bounds = (
                ('x1', x1, image_width),
                ('x2', x2, image_width),
                ('y1', y1, image_height),
                ('y2', y2, image_height),
            )
            errors.extend(
                f"[{index}] {name} out of range [0,{maximum}]"
                for name, value, maximum in bounds
                if not _in_range(value, 0.0, float(maximum))
            )
        if x2 <= x1 or y2 <= y1:
            errors.append(f"[{index}] bbox has non-positive size: {bbox}")
        return errors

    @staticmethod
    def _optional_field_errors(
        detection: dict[object, object],
        index: int,
    ) -> list[str]:
        """Validate optional confidence and class fields."""
        errors = [
            f"[{index}] '{field}' must be a number"
            for field in ('confidence', 'conf')
            if field in detection and not _is_number(detection[field])
        ]
        errors.extend(
            f"[{index}] '{field}' must be an integer"
            for field in ('class', 'cls')
            if field in detection and not isinstance(detection[field], int)
        )
        return errors

    async def _ensure_utils(self) -> Utils:
        """Ensure the utils module is initialised and return it."""
        if self._utils is None:
            self._utils = Utils()
            self.logger.info('Initialised utils module')
        return self._utils
