"""Optional native backend, selected once for each camera process."""
from __future__ import annotations

import importlib
from typing import cast
from typing import Literal
from typing import Protocol

from src.danger_detector import DangerDetector
from src.yolo_detector import YoloDetector


class NativeTracker(Protocol):
    def track(
        self,
        detections: list[list[float]],
        frame_count: int,
        movement_thr_sq: float,
        max_id_keep: int,
    ) -> list[list[float]]: ...


class NativeExtension(Protocol):
    ABI_VERSION: int

    def CentroidTracker(self) -> NativeTracker: ...

    def safety_warnings(
        self,
        datas: list[list[float]],
        detect_safety: bool,
        detect_proximity: bool,
    ) -> dict[str, dict[str, object]]: ...


def load_postprocess_backend(
    backend: Literal['auto', 'python', 'rust'],
) -> NativeExtension | None:
    """Absence is allowed in auto mode; broken installations remain errors."""
    if backend == 'python':
        return None
    try:
        extension = importlib.import_module('construction_postprocess')
    except ModuleNotFoundError as error:
        if backend == 'auto' and error.name == 'construction_postprocess':
            return None
        raise RuntimeError(
            'Rust postprocessing is unavailable; install the '
            'native/postprocess wheel or select POSTPROCESS_BACKEND=python',
        ) from error
    if getattr(extension, 'ABI_VERSION', None) != 1:
        raise RuntimeError('Unsupported construction_postprocess ABI version')
    if not all(
        callable(getattr(extension, name, None))
        for name in ('CentroidTracker', 'safety_warnings')
    ):
        raise RuntimeError('Incomplete construction_postprocess extension')
    return cast(NativeExtension, extension)


class RustYoloDetector(YoloDetector):
    """Shared-worker detection with camera-local native centroid state."""

    def __init__(self, *, native_extension: NativeExtension, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.remote_tracker != 'centroid':
            raise ValueError('Rust tracking supports the centroid tracker')
        self._native_tracker = native_extension.CentroidTracker()

    def _track_remote(self, dets: list[list[float]]) -> list[list[float]]:
        if self.remote_tracker != 'centroid':
            raise RuntimeError('Tracker algorithm cannot change mid-stream')
        return self._native_tracker.track(
            dets, self.frame_count, self.movement_thr_sq, self.max_id_keep,
        )


class RustDangerDetector(DangerDetector):
    """Native PPE/proximity rules with the existing polygon pipeline."""

    def __init__(
        self,
        detection_items: dict[str, bool] | None = None,
        *,
        native_extension: NativeExtension,
    ) -> None:
        super().__init__(detection_items)
        self._native_extension = native_extension

    def _check_person_safety(
        self,
        datas: list[list[float]],
        warnings: dict[str, dict[str, object]],
    ) -> None:
        warnings.update(
            self._native_extension.safety_warnings(
                datas,
                self._feature_enabled('detect_no_safety_vest_or_helmet'),
                self._feature_enabled('detect_near_machinery_or_vehicle'),
            ),
        )
