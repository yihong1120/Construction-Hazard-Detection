from __future__ import annotations

import asyncio
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.gpu_stream_capture import GpuDecodeOpenError
from src.gpu_stream_capture import GpuFrame
from src.gpu_stream_capture import GpuStreamCapture


def test_gpu_frame_letterbox_preserves_source_coordinates() -> None:
    """GPU preprocessing returns model input and reversible box geometry."""
    frame = GpuFrame(
        tensor=torch.zeros((3, 100, 200), dtype=torch.uint8),
        timestamp=1.0,
    )

    prepared, letterbox = frame.prepare_for_yolo(640, use_half=True)

    assert prepared.shape == (1, 3, 640, 640)
    assert prepared.dtype is torch.float16
    assert torch.all(prepared[:, :, :160] == 114 / 255)
    assert letterbox.restore_rows(
        [[32.0, 192.0, 96.0, 320.0, 0.9, 2.0]],
    ) == [[10.0, 10.0, 30.0, 50.0, 0.9, 2.0]]


def test_gpu_frame_downloads_bgr_once() -> None:
    """Multiple CPU consumers share one cached BGR download."""
    rgb = torch.tensor(
        [
            [[10]],
            [[20]],
            [[30]],
        ],
        dtype=torch.uint8,
    )
    frame = GpuFrame(tensor=rgb, timestamp=1.0)

    first = frame.to_bgr()
    second = frame.to_bgr()

    assert first is second
    assert np.array_equal(first, np.array([[[30, 20, 10]]], dtype=np.uint8))


def test_gpu_frame_rejects_non_rgb_tensor() -> None:
    """The GPU decoder contract rejects an invalid channel count."""
    frame = GpuFrame(
        tensor=torch.zeros((1, 8, 8), dtype=torch.uint8),
        timestamp=1.0,
    )

    with pytest.raises(ValueError, match='RGB tensor'):
        frame.prepare_for_yolo(640, use_half=False)


def test_gpu_stream_open_error_is_safe_for_cpu_fallback() -> None:
    """TorchCodec opening failures become a fallback-specific exception."""

    async def run() -> None:
        capture = GpuStreamCapture('rtsp://example.invalid/stream')
        with (
            patch.object(GpuStreamCapture, 'is_available', return_value=True),
            patch.object(
                capture,
                '_initialise_decoder',
                side_effect=RuntimeError('source URL details'),
            ),
            pytest.raises(GpuDecodeOpenError, match='TorchCodec could not'),
        ):
            await capture.initialise_stream(capture.stream_url)

    asyncio.run(run())
