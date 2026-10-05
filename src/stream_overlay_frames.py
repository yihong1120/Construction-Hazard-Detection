"""Build immutable media frames and cache per-language overlay renditions."""
from __future__ import annotations

import os

import cv2
import numpy as np

from examples.streaming_web.overlay_labels import normalise_label_language
from examples.streaming_web.overlay_labels import SUPPORTED_LABEL_LANGUAGES
from examples.streaming_web.overlay_models import PolygonCollection
from examples.streaming_web.overlay_models import TrackingDetections
from examples.streaming_web.overlay_models import WarningPayload
from examples.streaming_web.overlay_renderer import render_overlay_array
from src.stream_runtime_state import OverlaySnapshot as _OverlaySnapshot


def _overlay_publish_frames(
    snapshot: _OverlaySnapshot,
    requested_languages: set[str],
    rendered_overlay_cache: dict[
        str,
        tuple[tuple[int, int], np.ndarray],
    ],
) -> dict[str, np.ndarray]:
    """Build at most one immutable overlay frame per requested language."""
    if snapshot.track_data is None:
        return {language: snapshot.frame for language in requested_languages}
    publish_frames: dict[str, np.ndarray] = {}
    for language in requested_languages:
        cached = rendered_overlay_cache.get(language)
        if cached is not None and cached[0] == snapshot.sequence:
            publish_frames[language] = cached[1]
            continue
        publish_frame = _build_media_publish_frame(
            frame=snapshot.frame,
            warnings=snapshot.warnings,
            cone_polys=snapshot.cone_polys,
            pole_polys=snapshot.pole_polys,
            track_data=snapshot.track_data,
            label_language=language,
            class_metadata=snapshot.class_metadata,
        )
        rendered_overlay_cache[language] = (snapshot.sequence, publish_frame)
        publish_frames[language] = publish_frame
    return publish_frames


def _build_media_publish_frame(
    frame: np.ndarray,
    warnings: WarningPayload,
    cone_polys: PolygonCollection,
    pole_polys: PolygonCollection,
    track_data: TrackingDetections,
    label_language: str = 'en',
    class_metadata: list[dict] | None = None,
) -> np.ndarray:
    """Return the annotated frame published to MediaMTX."""
    # Overlay rendering draws onto the frame, so copy exactly once here while
    # the rest of the live pipeline can pass frame references around.
    return render_overlay_array(
        frame.copy(),
        detection_items=track_data,
        class_metadata=class_metadata,
        warnings=warnings,
        cone_polygons=cone_polys,
        pole_polygons=pole_polys,
        overlay_mode='backend',
        label_language=label_language,
        min_confidence=float(
            os.getenv(
                'MEDIA_PUBLISH_OVERLAY_MIN_CONFIDENCE',
                '0.25',
            ),
        ),
        box_thickness=max(
            1,
            int(float(os.getenv('MEDIA_PUBLISH_BOX_THICKNESS', '2'))),
        ),
    )


def _build_media_startup_frame(site: str, stream_name: str) -> np.ndarray:
    """Return a startup frame for the annotated media path."""
    width = int(os.getenv('MEDIA_PUBLISH_STARTUP_WIDTH', '1280'))
    height = int(os.getenv('MEDIA_PUBLISH_STARTUP_HEIGHT', '720'))
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


def _csv_env(name: str, default: str) -> list[str]:
    """Read a comma-separated environment setting.

    Args:
        name: Environment variable name.
        default: Default comma-separated value.

    Returns:
        Normalised non-empty entries.
    """
    value = os.getenv(name, default)
    return [item.strip() for item in value.split(',') if item.strip()]


def _allowed_overlay_languages() -> tuple[str, ...]:
    """Return enabled overlay languages supported by the renderer."""
    configured = _csv_env(
        'MEDIA_OVERLAY_ALLOWED_LANGUAGES',
        ','.join(SUPPORTED_LABEL_LANGUAGES),
    )
    allowed = []
    for language in configured:
        normalised = normalise_label_language(language)
        if (
            normalised in SUPPORTED_LABEL_LANGUAGES
            and normalised not in allowed
        ):
            allowed.append(normalised)
    return tuple(allowed or ('en',))


def _mark_frame_readonly(frame: np.ndarray) -> None:
    """Mark a captured frame immutable so it can be shared without copies."""
    try:
        frame.setflags(write=False)
    except ValueError:
        pass
