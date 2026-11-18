"""Opt-in device-frame experiment; deliberately separate from StreamCapture.

The calibrated point-chroma profile has not passed the legacy color gate.
Codec I/O uses PyAV/FFmpeg, while image storage, preprocessing, composition,
inference and NVENC input ownership are implemented by the native Rust module.
"""
from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
from av.codec.hwaccel import HWAccel

from src.gpu_color_lookup import build_nv12_lookup
from src.native_inference import create_rust_model
from src.native_inference import load_inference_extension
from src.nvenc_session import release_nvenc_session
from src.nvenc_session import take_nvenc_reservation
from src.nvenc_session import try_acquire_nvenc_session
from src.settings.inference import WorkerSettings


class ExperimentalGpuMedia:
    def __init__(self, engine: Path, *, device: int = 0, fps: int = 15):
        self._closed = True
        settings = WorkerSettings(backend='rust', native_async=True)
        self.native = load_inference_extension(settings)
        if getattr(self.native, 'GPU_FRAME_API_VERSION', None) != 1:
            raise RuntimeError('Rebuild the GPU-frame inference wheel')
        if not getattr(self.native, 'HAS_GPU_ENCODER', False):
            raise RuntimeError('Rebuild the inference wheel with gpu-media')
        self.device = device
        self.fps = fps
        self.factories: dict[tuple[int, int], Any] = {}
        self.encoder = None
        self.reservation = None
        self.model = create_rust_model(
            engine, settings, device=f'cuda:{device}', rect=False,
        )
        self.input_frames = 0
        self.encoded_bytes = 0
        self.glyph_upload_bytes = 0
        self._closed = False

    def open_source(self, url: str):
        # Binding an already active CUDA context avoids primary-context flags
        # conflicts with TensorRT. Use this API on the model's owning thread.
        return av.open(
            url, timeout=(5.0, 5.0),
            options={'rtsp_transport': 'tcp'},
            hwaccel=HWAccel(
                'cuda', device=str(self.device), allow_software_fallback=False,
                options={'primary_ctx': '0'}, flags=2, is_hw_owned=True,
            ),
        )

    def import_frame(self, decoded):
        if decoded.format.name != 'cuda' or len(decoded.planes) != 2:
            raise ValueError('Device media requires CUDA NV12 frames')
        if decoded.color_range not in (
                0, 1, 2,
        ) or decoded.colorspace not in (
                1, 2, 5, 6,
        ):
            raise ValueError(
                'Color metadata is outside the experimental profile',
            )
        profile = (decoded.colorspace, decoded.color_range)
        factory = self.factories.get(profile)
        if factory is None:
            if len(self.factories) >= 2:
                raise RuntimeError('GPU color-profile cache is full')
            factory = self.native.GpuFrameFactory(
                build_nv12_lookup(*profile), self.device,
            )
            self.factories[profile] = factory
        frame = factory.import_nv12(
            decoded.planes[0],
            decoded.planes[1],
            decoded.width,
            decoded.height,
        )
        self.input_frames += 1
        return frame

    def predict(self, frames):
        return self.model.predict_device_detections(frames)

    def _blit(self, output, patch, x, y):
        payload = patch.tobytes()
        self.glyph_upload_bytes += len(payload)
        output.blend_bgra(payload, patch.shape[1], patch.shape[0], x, y)

    def annotate(self, frame, detections):
        output = frame.copy()
        try:
            for row in detections:
                x1, y1, x2, y2 = (int(value) for value in row[:4])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.width - 1, x2), min(frame.height - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                width, height = x2 - x1, y2 - y1
                horizontal = np.zeros((5, width + 5, 4), dtype=np.uint8)
                vertical = np.zeros((height + 5, 5, 4), dtype=np.uint8)
                cv2.line(
                    horizontal, (2, 2), (width + 2, 2), (0, 255, 0, 255), 2,
                )
                cv2.line(
                    vertical, (2, 2), (2, height + 2), (0, 255, 0, 255), 2,
                )
                for patch, x, y in (
                    (horizontal, x1 - 2, y1 - 2),
                    (horizontal, x1 - 2, y2 - 2),
                    (vertical, x1 - 2, y1 - 2),
                    (vertical, x2 - 2, y1 - 2),
                ):
                    self._blit(output, patch, x, y)
                label = f'{int(row[5])} {row[4]:.2f}'
                (width, height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, .4, 1,
                )
                glyph = np.zeros(
                    (height + baseline + 4, width + 4, 4), dtype=np.uint8,
                )
                cv2.putText(
                    glyph, label, (2, height + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255, 255), 1,
                )
                self._blit(output, glyph, x1, max(0, y1 - glyph.shape[0]))
            return output
        except BaseException:
            output.close()
            raise

    def _open_encoder(self, frame):
        if not try_acquire_nvenc_session():
            raise RuntimeError('Host-wide NVENC budget exhausted')
        self.reservation = take_nvenc_reservation()
        try:
            self.encoder = self.native.GpuEncoder(
                frame.width, frame.height, self.fps, self.device,
                lease=self.reservation,
            )
        except BaseException:
            release_nvenc_session(self.reservation)
            self.reservation = None
            raise

    def encode(self, frame):
        if self.encoder is None:
            self._open_encoder(frame)
        data = self.encoder.encode(frame)
        self.encoded_bytes += len(data)
        return data

    def flush(self):
        if self.encoder is None:
            return b''
        data = self.encoder.encode()
        self.encoded_bytes += len(data)
        return data

    def close(self):
        if self._closed:
            return
        if self.encoder is not None:
            self.encoder.close()
            self.encoder = None
        if self.reservation is not None:
            release_nvenc_session(self.reservation)
            self.reservation = None
        self.factories.clear()
        self.model.close()
        self._closed = True

    def __del__(self):
        if not getattr(self, '_closed', True):
            with suppress(Exception):
                self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
