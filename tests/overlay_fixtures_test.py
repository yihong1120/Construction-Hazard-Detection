from __future__ import annotations

from examples.streaming_web.overlay_renderer import OverlayRenderer
from src.settings.media import MediaPublishSettings
from src.settings.media import OverlaySettings


def make_overlay_renderer():
    settings = MediaPublishSettings()
    return OverlayRenderer(
        OverlaySettings(
            min_confidence=settings.overlay_min_confidence,
            box_thickness=settings.box_thickness,
        ),
    )
