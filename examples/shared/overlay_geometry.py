"""Present persisted cluster boundaries; never run clustering on reads."""
from __future__ import annotations

import json
import math


def decode_polygons(raw):
    """Decode pixel-space polygon rings, preserving their original points."""
    if raw is None:
        return None
    value = json.loads(raw)
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError('Expected polygon array')
    for ring in value:
        if not isinstance(ring, list) or len(ring) < 3:
            raise ValueError('Polygon requires at least three points')
        for point in ring:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError('Expected [x, y] point')
            if any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                for v in point
            ):
                raise ValueError('Polygon coordinates must be finite numbers')
        area = sum(
            a[0] * b[1] - b[0] * a[1]
            for a, b in zip(ring, ring[1:] + ring[:1])
        )
        if not math.isfinite(area) or abs(area) < 1e-9:
            raise ValueError('Polygon must have nonzero area')
    return value


def stored_regions(violation, image_size, style):
    regions, status = [], {}
    for kind, raw, name, color_key in (
        (
            'cone',
            getattr(violation, 'cone_polygon_json', None),
            '安全錐管制區域',
            'cone_polygon_color',
        ),
        (
            'pole',
            getattr(violation, 'pole_polygon_json', None),
            '電桿管制區域',
            'pole_polygon_color',
        ),
    ):
        try:
            rings = decode_polygons(raw)
        except ValueError, TypeError, OverflowError:
            status[kind] = 'invalid'
            continue
        if rings is None:
            status[kind] = 'not_recorded'
        elif not rings:
            status[kind] = 'empty'
        elif not image_size or min(image_size) <= 0:
            status[kind] = 'image_unavailable'
        else:
            status[kind] = 'available'
            width, height = image_size
            for index, ring in enumerate(rings):
                regions.append(
                    {
                        'id': f'{kind}_{index}',
                        'kind': kind,
                        'display_name': name,
                        'points': [[x / width, y / height] for x, y in ring],
                        'color': style[color_key],
                        'fill_opacity': 0.4,
                        'closed': True,
                    },
                )
    return regions, status
