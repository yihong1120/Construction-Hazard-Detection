from __future__ import annotations

import cv2
import numpy as np

from examples.streaming_web.overlay_renderer import OverlayRenderer
from src.settings.media import MediaPublishSettings
from src.settings.overlay_language import get_overlay_language_policy
from src.stream_runtime_state import OverlaySnapshot as _OverlaySnapshot


class OverlayFrameCache:
    def __init__(self, renderer: OverlayRenderer):
        self.renderer = renderer
        self.frames: dict[str, tuple[tuple[int, int], np.ndarray]] = {}

    def prune(self, languages: set[str]) -> None:
        for language in self.frames.keys() - languages:
            del self.frames[language]

    def render(
        self,
        snapshot: _OverlaySnapshot,
        languages: set[str],
    ) -> dict[str, np.ndarray]:
        self.prune(languages)
        if snapshot.track_data is None:
            return {language: snapshot.frame for language in languages}
        result = {}
        for language in languages:
            cached = self.frames.get(language)
            if cached is None or cached[0] != snapshot.sequence:
                cached = (
                    snapshot.sequence,
                    self.renderer.render(snapshot, language),
                )
                self.frames[language] = cached
            result[language] = cached[1]
        return result


def _build_media_startup_frame(
    site: str,
    stream_name: str,
    size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return a startup frame for the annotated media path."""
    if size is None:
        settings = MediaPublishSettings()
        size = (settings.startup_width, settings.startup_height)
    width, height = size
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    title = f'{site} / {stream_name}'
    subtitle = 'Starting live analysis...'
    cv2.putText(
        frame,
        title[:80],
        (48, max(80, height // 2 - 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        subtitle,
        (48, max(130, height // 2 + 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (160, 200, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def _allowed_overlay_languages() -> tuple[str, ...]:
    """Return enabled overlay languages supported by the renderer."""
    return get_overlay_language_policy().allowed


def _mark_frame_readonly(frame: np.ndarray) -> None:
    """Mark a captured frame immutable so it can be shared without copies."""
    try:
        frame.setflags(write=False)
    except ValueError:
        pass
