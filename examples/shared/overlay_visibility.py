"""Display policy; does not filter detections or grant feedback access."""
from __future__ import annotations


def object_visibility(code: str | None) -> dict[str, bool]:
    show = code != 'safety_cone'
    return {
        'show_box': show,
        'show_label': show,
        'feedback_show_box': True,
        'feedback_show_label': True,
    }
