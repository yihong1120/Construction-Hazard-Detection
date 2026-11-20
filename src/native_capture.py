"""OpenCV-shaped adapter for the explicitly selected native capture backend."""
from __future__ import annotations

from typing import Any
from typing import cast

import cv2
import numpy as np

from src.native_postprocess import load_postprocess_backend
from src.settings.capture import CaptureSettings


class RustCapture:
    def __init__(self, url: str, settings: CaptureSettings) -> None:
        extension = cast(Any, load_postprocess_backend('rust'))
        if getattr(extension, 'CAPTURE_API_VERSION', None) != 1:
            raise RuntimeError(
                'Rebuild bindings/python/postprocess for Rust capture',
            )
        self.timeout = settings.read_timeout_ms / 1000
        self.capture = extension.LatestCapture(
            url, ffmpeg=settings.ffmpeg_path, ffprobe=settings.ffprobe_path,
            timeout=settings.open_timeout_ms / 1000,
            cuda=settings.decoder == 'cuda',
        )
        self.closed = False

    def read(self):
        if self.closed:
            return False, None
        owner = self.capture.read(self.timeout)
        if owner is None:
            return False, None
        frame = np.frombuffer(owner, dtype=np.uint8).reshape(
            self.capture.height, self.capture.width, 3,
        )
        return True, frame

    def isOpened(self) -> bool:
        return not self.closed

    def get(self, key: int) -> float:
        return {
            cv2.CAP_PROP_FPS: self.capture.fps,
            cv2.CAP_PROP_FRAME_WIDTH: self.capture.width,
            cv2.CAP_PROP_FRAME_HEIGHT: self.capture.height,
        }.get(key, 0.0)

    def release(self) -> None:
        if not self.closed:
            self.capture.close()
            self.closed = True
