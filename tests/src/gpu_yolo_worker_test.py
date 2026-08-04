from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest
import torch

from src.gpu_stream_capture import GpuFrame
from src.gpu_yolo_worker import GpuYoloBatcher


class _FakeBoxes:
    """Minimal Ultralytics-style boxes for a deterministic batch test."""

    def __init__(self, rows: list[list[float]]) -> None:
        self.data = torch.tensor(rows, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data)


class _FakeResult:
    """Minimal Ultralytics-style result object."""

    def __init__(self, rows: list[list[float]]) -> None:
        self.boxes = _FakeBoxes(rows)


class _FakeModel:
    """Record predict calls while returning one box for every batch entry."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def predict(self, **kwargs: Any) -> list[_FakeResult]:
        self.calls.append(kwargs)
        batch = kwargs['source']
        return [
            _FakeResult([[32.0, 192.0, 96.0, 320.0, 0.9, 2.0]])
            for _ in range(batch.shape[0])
        ]


def test_gpu_yolo_batcher_reuses_one_model_for_two_cameras() -> None:
    """Same-model CUDA frames become one model call with batch size two."""

    async def run() -> None:
        batcher = GpuYoloBatcher(
            device='cpu',
            batch_size=8,
            batch_wait_seconds=0.05,
        )
        model = _FakeModel()
        batcher.model_cache['yolo26n'] = model
        first = GpuFrame(
            tensor=torch.zeros((3, 100, 200), dtype=torch.uint8),
            timestamp=1.0,
        )
        second = GpuFrame(
            tensor=torch.zeros((3, 100, 200), dtype=torch.uint8),
            timestamp=2.0,
        )
        try:
            results = await asyncio.gather(
                batcher.client('site|camera-a').detect(first, 'yolo26n'),
                batcher.client('site|camera-b').detect(second, 'yolo26n'),
            )
        finally:
            await batcher.close()

        assert len(model.calls) == 1
        assert model.calls[0]['source'].shape == (2, 3, 640, 640)
        assert results[0][0][:4] == [10.0, 10.0, 30.0, 50.0]
        assert results[1][0][:4] == [10.0, 10.0, 30.0, 50.0]
        assert results[0][0][4] == pytest.approx(0.9)
        assert results[1][0][4] == pytest.approx(0.9)
        assert [results[0][0][5], results[1][0][5]] == [2, 2]

    asyncio.run(run())


def test_gpu_yolo_batcher_keeps_different_model_keys_separate() -> None:
    """Each model key gets a separate batch and retains its cache entry."""

    async def run() -> None:
        batcher = GpuYoloBatcher(
            device='cpu',
            batch_size=8,
            batch_wait_seconds=0.01,
        )
        first_model = _FakeModel()
        second_model = _FakeModel()
        batcher.model_cache.update(
            {'yolo26n': first_model, 'yolo26s': second_model},
        )
        frame = GpuFrame(
            tensor=torch.zeros((3, 100, 200), dtype=torch.uint8),
            timestamp=1.0,
        )
        try:
            await asyncio.gather(
                batcher.client('site|camera-a').detect(frame, 'yolo26n'),
                batcher.client('site|camera-b').detect(frame, 'yolo26s'),
            )
        finally:
            await batcher.close()

        assert len(first_model.calls) == 1
        assert len(second_model.calls) == 1
        assert first_model.calls[0]['source'].shape[0] == 1
        assert second_model.calls[0]['source'].shape[0] == 1

    asyncio.run(run())


def test_gpu_yolo_batcher_uploads_cpu_fallback_frame() -> None:
    """A TCP-decoded BGR frame still uses the shared model cache."""

    async def run() -> None:
        batcher = GpuYoloBatcher(
            device='cpu',
            batch_size=1,
            batch_wait_seconds=0,
        )
        model = _FakeModel()
        batcher.model_cache['yolo26n'] = model
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame.setflags(write=False)
        try:
            result = await batcher.client('site|camera-a').detect(
                frame,
                'yolo26n',
            )
        finally:
            await batcher.close()

        assert len(model.calls) == 1
        assert model.calls[0]['source'].shape == (1, 3, 640, 640)
        assert result[0][:4] == [10.0, 10.0, 30.0, 50.0]

    asyncio.run(run())


def test_gpu_yolo_batcher_keeps_only_latest_pending_camera_frame() -> None:
    """A newer frame supersedes a queued frame from the same camera."""

    async def run() -> None:
        batcher = GpuYoloBatcher(
            device='cpu',
            batch_size=2,
            batch_wait_seconds=0.05,
        )
        model = _FakeModel()
        batcher.model_cache['yolo26n'] = model
        first = GpuFrame(
            tensor=torch.zeros((3, 100, 200), dtype=torch.uint8),
            timestamp=1.0,
        )
        second = GpuFrame(
            tensor=torch.ones((3, 100, 200), dtype=torch.uint8),
            timestamp=2.0,
        )
        try:
            first_result = asyncio.create_task(
                batcher.client('site|camera-a').detect(first, 'yolo26n'),
            )
            await asyncio.sleep(0)
            second_result = asyncio.create_task(
                batcher.client('site|camera-a').detect(second, 'yolo26n'),
            )
            results = await asyncio.gather(first_result, second_result)
        finally:
            await batcher.close()

        assert results[0] == []
        assert results[1][0][:4] == [10.0, 10.0, 30.0, 50.0]
        assert len(model.calls) == 1

    asyncio.run(run())
