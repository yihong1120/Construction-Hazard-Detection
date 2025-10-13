from __future__ import annotations

import logging
from math import sqrt

from src.utils import Utils


class UtilsTools:
    """Tools for geometry operations and general utilities."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._utils = None

    async def calculate_polygon_area(
        self,
        polygon_points: list[list[float]],
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
        point: list[float],
        polygon_points: list[list[float]],
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
        bbox1: list[float],
        bbox2: list[float],
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
            def _norm(b: list[float]) -> tuple[float, float, float, float]:
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
        point1: list[float],
        point2: list[float],
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
            # Basic normalisation utilities
            def _clip(v: float, lo: float, hi: float) -> float:
                return max(lo, min(hi, v))

            norm = []
            if target_format == 'normalized':
                # Convert absolute coords to [0,1]
                for x, y in coordinates:
                    nx = _clip(x / image_width, 0.0, 1.0)
                    ny = _clip(y / image_height, 0.0, 1.0)
                    norm.append([nx, ny])
            elif target_format == 'yolo':
                # Expect coordinates are bbox corners [x1,y1,x2,y2] for each
                # item
                for bbox in coordinates:
                    if len(bbox) != 4:
                        raise ValueError('YOLO expects [x1,y1,x2,y2] per item')
                    x1, y1, x2, y2 = bbox
                    cx = _clip(((x1 + x2) / 2.0) / image_width, 0.0, 1.0)
                    cy = _clip(((y1 + y2) / 2.0) / image_height, 0.0, 1.0)
                    w = _clip((abs(x2 - x1)) / image_width, 0.0, 1.0)
                    h = _clip((abs(y2 - y1)) / image_height, 0.0, 1.0)
                    norm.append([cx, cy, w, h])
            elif target_format == 'coco':
                # Convert [x1,y1,x2,y2] -> [x,y,w,h]
                for bbox in coordinates:
                    if len(bbox) != 4:
                        raise ValueError('COCO expects [x1,y1,x2,y2] per item')
                    x1, y1, x2, y2 = bbox
                    x = _clip(min(x1, x2), 0.0, float(image_width))
                    y = _clip(min(y1, y2), 0.0, float(image_height))
                    w = _clip(abs(x2 - x1), 0.0, float(image_width))
                    h = _clip(abs(y2 - y1), 0.0, float(image_height))
                    norm.append([x, y, w, h])
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
        detections: list[dict],
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
            # Validate detections inline
            errors: list[str] = []

            def _is_number(v) -> bool:
                return isinstance(v, (int, float)) and not isinstance(v, bool)

            def _in_range(v: float, lo: float, hi: float) -> bool:
                return lo <= v <= hi

            for idx, det in enumerate(detections):
                if not isinstance(det, dict):
                    errors.append(f"[{idx}] detection must be an object/dict")
                    continue

                # Accept 'bbox' or 'box'
                bbox = det.get('bbox') if 'bbox' in det else det.get('box')
                if bbox is None:
                    errors.append(f"[{idx}] missing 'bbox'/'box'")
                    continue
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    errors.append(
                        (
                            f"[{idx}] 'bbox' must be a list of 4 numbers "
                            '[x1,y1,x2,y2]'
                        ),
                    )
                    continue
                if not all(_is_number(v) for v in bbox):
                    errors.append(f"[{idx}] 'bbox' values must be numbers")
                    continue

                x1, y1, x2, y2 = bbox
                # Normalized coords support: if all in [0,1],
                # scale check loosely
                if all(0.0 <= float(v) <= 1.0 for v in [x1, y1, x2, y2]):
                    # allow normalized, but ensure ordering and non-zero area
                    pass
                else:
                    if not _in_range(float(x1), 0.0, float(image_width)):
                        errors.append(
                            f"[{idx}] x1 out of range [0,{image_width}]",
                        )
                    if not _in_range(float(x2), 0.0, float(image_width)):
                        errors.append(
                            f"[{idx}] x2 out of range [0,{image_width}]",
                        )
                    if not _in_range(float(y1), 0.0, float(image_height)):
                        errors.append(
                            f"[{idx}] y1 out of range [0,{image_height}]",
                        )
                    if not _in_range(float(y2), 0.0, float(image_height)):
                        errors.append(
                            f"[{idx}] y2 out of range [0,{image_height}]",
                        )

                # Check geometry validity (after potential normalization)
                if float(x2) <= float(x1) or float(y2) <= float(y1):
                    errors.append(
                        f"[{idx}] bbox has non-positive size: {bbox}",
                    )

                # Optional fields validation
                if 'confidence' in det and not _is_number(det['confidence']):
                    errors.append(f"[{idx}] 'confidence' must be a number")
                if 'conf' in det and not _is_number(det['conf']):
                    errors.append(f"[{idx}] 'conf' must be a number")
                if 'class' in det and not isinstance(det['class'], int):
                    errors.append(f"[{idx}] 'class' must be an integer")
                if 'cls' in det and not isinstance(det['cls'], int):
                    errors.append(f"[{idx}] 'cls' must be an integer")

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

    async def _ensure_utils(self) -> None:
        """Ensure the utils module is initialised."""
        if self._utils is None:
            self._utils = Utils()
            self.logger.info('Initialised utils module')
