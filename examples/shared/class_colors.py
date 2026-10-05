"""Server-owned class colors and overlay presentation policy."""
from __future__ import annotations

import hashlib
import re

COLORS = {
    'helmet': '#4CAF50',
    'mask': '#26A69A',
    'no_helmet': '#F44336',
    'no_mask': '#EF5350',
    'no_safety_vest': '#E53935',
    'person': '#FF9800',
    'safety_cone': '#FF5722',
    'safety_vest': '#8BC34A',
    'machinery': '#FFC107',
    'utility_pole': '#03A9F4',
    'vehicle': '#FFEB3B',
    'forklift': '#AB47BC',
}


def normalize_color(value):
    if isinstance(value, str) and re.fullmatch(r'#[0-9a-fA-F]{6}', value):
        return value.upper()
    return None


def published_color(code):
    return COLORS.get(code) or (
        '#' + hashlib.sha256(code.encode('utf-8')).hexdigest()[:6].upper()
    )


def overlay_style(class_metadata=None):
    """Return an independent policy object for each response."""
    colors = dict(COLORS)
    for item in class_metadata or []:
        color = normalize_color(item.get('color'))
        if color:
            colors[item['code']] = color
    return {
        'schema_version': 1,
        'class_colors': colors,
        'unknown_class_color': '#9E9E9E',
        'cone_polygon_color': '#FF4081',
        'pole_polygon_color': '#448AFF',
        'warning_color': '#F44336',
    }
