from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from src.stream_capture import _redact_stream_url


class GpuDecodeOpenError(RuntimeError):
    """Raised when TorchCodec cannot open an RTSP source for NVDEC."""


@dataclass(frozen=True)
class GpuLetterbox:
    """Geometry needed to map YOLO boxes back to the decoded frame."""

    source_height: int
    source_width: int
    scale: float
    pad_left: int
    pad_top: int

    def restore_rows(self, rows: list[list[float]]) -> list[list[float]]:
        """Map first-four-coordinate YOLO rows to source-frame pixels."""
        restored: list[list[float]] = []
        for row in rows:
            copied = list(row)
            copied[0] = self._restore_x(copied[0])
            copied[1] = self._restore_y(copied[1])
            copied[2] = self._restore_x(copied[2])
            copied[3] = self._restore_y(copied[3])
            restored.append(copied)
        return restored

    def _restore_x(self, value: float) -> float:
        """Map one model-space x coordinate back to the source frame."""
        return min(
            float(self.source_width),
            max(0.0, (float(value) - self.pad_left) / self.scale),
        )

    def _restore_y(self, value: float) -> float:
        """Map one model-space y coordinate back to the source frame."""
        return min(
            float(self.source_height),
            max(0.0, (float(value) - self.pad_top) / self.scale),
        )


@dataclass
class GpuFrame:
    """One RGB CUDA frame retained on GPU until a CPU consumer needs it."""

    tensor: torch.Tensor
    timestamp: float
    _bgr_frame: np.ndarray | None = field(default=None, init=False, repr=False)

    def prepare_for_yolo(
        self,
        image_size: int,
        use_half: bool,
    ) -> tuple[torch.Tensor, GpuLetterbox]:
        """Letterbox and normalise this RGB CUDA frame without CPU copies."""
        if self.tensor.ndim != 3 or self.tensor.shape[0] != 3:
            raise ValueError(
                'GPU decoder must return an RGB tensor shaped '
                '[3, height, width]',
            )

        _, source_height, source_width = self.tensor.shape
        scale = min(image_size / source_height, image_size / source_width)
        resized_height = max(1, round(source_height * scale))
        resized_width = max(1, round(source_width * scale))
        pad_width = image_size - resized_width
        pad_height = image_size - resized_height
        pad_left = pad_width // 2
        pad_top = pad_height // 2

        resized = functional.interpolate(
            self.tensor.unsqueeze(0).to(dtype=torch.float32),
            size=(resized_height, resized_width),
            mode='bilinear',
            align_corners=False,
        )
        prepared = functional.pad(
            resized,
            (
                pad_left,
                pad_width - pad_left,
                pad_top,
                pad_height - pad_top,
            ),
            value=114.0,
        ).div_(255.0)
        if use_half:
            prepared = prepared.to(dtype=torch.float16)
        return prepared, GpuLetterbox(
            source_height=source_height,
            source_width=source_width,
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
        )

    def to_bgr(self) -> np.ndarray:
        """Download this RGB frame once for overlay rendering or a snapshot."""
        if self._bgr_frame is None:
            rgb = self.tensor.permute(1, 2, 0).to(device='cpu')
            self._bgr_frame = np.ascontiguousarray(rgb.numpy()[..., ::-1])
        return self._bgr_frame


class GpuStreamCapture:
    """Read live RTSP frames with NVDEC and return CUDA-backed RGB tensors."""

    def __init__(
        self,
        stream_url: str,
        capture_interval: float = 15.0,
    ) -> None:
        """Initialise a GPU decoder for one live stream."""
        self.stream_url = stream_url
        self.capture_interval = capture_interval
        self.reopen_delay = float(
            os.getenv('STREAM_CAPTURE_REOPEN_DELAY_SECONDS', '5.0'),
        )
        self.max_reopen_delay = float(
            os.getenv(
                'STREAM_CAPTURE_MAX_REOPEN_DELAY_SECONDS',
                '60.0',
            ),
        )
        self.device = os.getenv('GPU_DECODE_DEVICE', 'cuda:0')
        self._ops: Any | None = None
        self._decoder: torch.Tensor | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix='gpu-stream-decoder',
        )

    @staticmethod
    def is_available() -> bool:
        """Return whether this process can create CUDA-backed decoders."""
        if not torch.cuda.is_available():
            return False
        try:
            from torchcodec._core import ops  # type: ignore[import-not-found]
        except ImportError:
            return False
        return all(
            hasattr(ops, name)
            for name in (
                'create_from_file',
                'add_video_stream',
                'get_next_frame',
            )
        )

    def update_capture_interval(self, capture_interval: float) -> None:
        """Set the minimum delay between frames yielded to downstream work."""
        self.capture_interval = capture_interval

    async def initialise_stream(self, stream_url: str) -> None:
        """Open the source and attach TorchCodec's NVDEC video stream."""
        if not self.is_available():
            raise RuntimeError(
                'GPU decode requires CUDA and torchcodec with NVDEC support',
            )
        self.stream_url = stream_url
        try:
            await asyncio.get_running_loop().run_in_executor(
                self._executor,
                self._initialise_decoder,
            )
        except Exception as exc:
            await self.release_resources()
            raise GpuDecodeOpenError(
                'TorchCodec could not open the RTSP source for NVDEC',
            ) from exc

    def _initialise_decoder(self) -> None:
        """Create the TorchCodec decoder on its dedicated capture thread."""
        from torchcodec._core import ops  # type: ignore[import-not-found]

        decoder = ops.create_from_file(self.stream_url, 'approximate')
        ops.add_video_stream(
            decoder,
            num_threads=1,
            dimension_order='NCHW',
            device=self.device,
            device_variant='default',
            output_dtype='uint8',
        )
        self._ops = ops
        self._decoder = decoder

    async def release_resources(self) -> None:
        """Release the live decoder and its dedicated capture thread."""
        self._decoder = None
        self._ops = None
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def execute_capture(self):
        """Yield the latest CUDA frame at the configured processing cadence."""
        await self.initialise_stream(self.stream_url)
        last_process_time = time.monotonic() - self.capture_interval
        backoff_seconds = self.reopen_delay
        fail_count = 0

        while True:
            try:
                frame = await asyncio.get_running_loop().run_in_executor(
                    self._executor,
                    self._read_next_frame,
                )
            except Exception as exc:
                fail_count += 1
                print(
                    'Failed to GPU-decode frame, reinitialising stream. '
                    f'Fail count: {fail_count}, '
                    f'source={_redact_stream_url(self.stream_url)}, '
                    f'error={type(exc).__name__}',
                    flush=True,
                )
                self._decoder = None
                self._ops = None
                await asyncio.sleep(backoff_seconds)
                backoff_seconds = min(
                    self.max_reopen_delay,
                    max(self.reopen_delay, backoff_seconds * 1.5),
                )
                await self.initialise_stream(self.stream_url)
                continue

            fail_count = 0
            backoff_seconds = self.reopen_delay
            if time.monotonic() - last_process_time < self.capture_interval:
                continue
            last_process_time = time.monotonic()
            yield frame, frame.timestamp

    def _read_next_frame(self) -> GpuFrame:
        """Decode the next frame and assert that it remains CUDA-backed."""
        if self._ops is None or self._decoder is None:
            raise RuntimeError('GPU stream decoder is not initialised')
        tensor, _pts, _duration = self._ops.get_next_frame(self._decoder)
        if tensor.device.type != 'cuda':
            raise RuntimeError('TorchCodec fell back to CPU decoding')
        return GpuFrame(tensor=tensor, timestamp=time.time())
