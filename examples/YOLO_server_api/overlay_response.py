"""Build image detection overlays without storing violation records."""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

from examples.shared.class_colors import overlay_style
from examples.shared.overlay_geometry import stored_regions
from examples.shared.overlay_visibility import object_visibility
from examples.YOLO_server_api.detection import convert_to_image
from examples.YOLO_server_api.schemas import DetectionOverlayResponse


def calculate_regions(rows, classes):
    from sklearn.cluster import HDBSCAN
    from src.geometry import (
        detect_polygon_from_cones,
        build_utility_pole_union,
        polygons_to_coords,
    )

    # Map model classes to the geometry functions’ internal semantic IDs.
    semantic_ids = {'safety_cone': 6, 'utility_pole': 9}
    id_map = {
        c.id: semantic_ids[c.code] for c in classes if c.code in semantic_ids
    }
    inputs = [[*r[:5], id_map[r[5]]] for r in rows if r[5] in id_map]
    clusterer = HDBSCAN(min_samples=3, min_cluster_size=2, copy=True)
    cones = polygons_to_coords(detect_polygon_from_cones(inputs, clusterer))
    poles = polygons_to_coords([build_utility_pole_union(inputs, clusterer)])
    return cones, poles


def build_overlay_response(rows, entry, image_bytes, compute_regions, view):
    image = convert_to_image(image_bytes)
    height, width = image.shape[:2]
    metadata = [c.model_dump() for c in entry.classes]
    classes = {c.id: c for c in entry.classes}
    style = overlay_style(metadata)
    objects, valid_rows = [], []
    for index, row in enumerate(rows):
        if len(row) < 6 or not all(math.isfinite(v) for v in row[:6]):
            continue
        x1, y1, x2, y2, confidence, class_id = row[:6]
        x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
        y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        c = classes.get(class_id)
        code = c.code if c else None
        label = code or f'class-{class_id:g}'
        visibility = object_visibility(code)
        if view == 'objects':
            visibility.update(show_box=True, show_label=True)
        objects.append(
            {
                'object_id': f'det_{index}',
                'code': code,
                'label': label,
                'display_name': c.display_name if c else label,
                'color': c.color if c else style['unknown_class_color'],
                'confidence': confidence,
                'bbox': {
                    'x': x1 / width,
                    'y': y1 / height,
                    'w': (x2 - x1) / width,
                    'h': (y2 - y1) / height,
                },
                **visibility,
            },
        )
        valid_rows.append([x1, y1, x2, y2, confidence, class_id])
    regions = []
    status = {'cone': 'not_computed', 'pole': 'not_computed'}
    if compute_regions:
        cones, poles = calculate_regions(valid_rows, entry.classes)
        regions, status = stored_regions(
            SimpleNamespace(
                cone_polygon_json=json.dumps(cones),
                pole_polygon_json=json.dumps(poles),
            ),
            (width, height),
            style,
        )
    return DetectionOverlayResponse(
        model_id=entry.id,
        model_version=entry.version,
        image_size={'width': width, 'height': height},
        class_metadata=metadata,
        overlay_style=style,
        overlay_objects=objects,
        overlay_regions=regions,
        region_status=status,
        view=view,
    )
