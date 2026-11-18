"""Async adapter ordering, shape barriers and failure cleanup without CUDA."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.native_inference import create_rust_model
from src.native_inference import RustYoloModel
from src.settings.gpu_resources import GpuResourceSettings
from src.settings.inference import WorkerSettings


class AsyncDouble:
    def __init__(self, *args, **kwargs):
        self.max_batch = 2
        self.pending = {}
        self.sequence = 0
        self.events = []
        self.fail_submit = None
        self.short_output = False

    def submit(self, frames):
        self.sequence += 1
        if self.fail_submit == self.sequence:
            raise RuntimeError('injected submit failure')
        assert len(self.pending) < 2
        self.pending[self.sequence] = [
            [[0, 0, w, h, .9, 5]] for h, w, _ in frames
        ]
        self.events.append(('submit', self.sequence))
        return self.sequence

    def collect(self, ticket):
        self.events.append(('collect', ticket))
        result = self.pending.pop(ticket)
        return [] if self.short_output else result

    def predict_device_frames(self, frames):
        return [] if self.short_output else [[] for _ in frames]

    def close(self):
        self.pending.clear()


def make_model(rect=False, asynchronous=True):
    return RustYoloModel(
        SimpleNamespace(ASYNC_API_VERSION=1, TensorRTEngine=AsyncDouble),
        Path('fixture.engine'), WorkerSettings(
            backend='rust', native_async=asynchronous,
            native_micro_batch_size=2,
        ), device='cuda:0', rect=rect,
    )


def test_two_slots_preserve_order_and_flush_partial_shape():
    frames = [np.zeros((4, 6, 3), dtype=np.uint8)] * 7
    with_model = make_model()
    try:
        result = with_model.predict_detections(frames)
        assert result == [[[0, 0, 6, 4, .9, 5]]] * 7
        assert with_model.engine.events == [
            ('submit', 1), ('submit', 2), ('collect', 1), ('submit', 3),
            ('collect', 2), ('collect', 3), ('submit', 4), ('collect', 4),
        ]
        assert not with_model.engine.pending
    finally:
        with_model.close()


def test_rect_shape_barrier_and_failed_submit_drain_pending():
    model = make_model(rect=True)
    try:
        frames = [np.zeros((h, 6, 3), dtype=np.uint8) for h in [4, 4, 8, 8]]
        model.predict_detections(frames)
        assert model.engine.events == [
            ('submit', 1), ('collect', 1), ('submit', 2), ('collect', 2),
        ]
        model.engine.fail_submit = 4
        with pytest.raises(RuntimeError, match='injected'):
            model.predict_detections(frames[:2] * 3)
        assert not model.engine.pending
        model.engine.short_output = True
        with pytest.raises(RuntimeError, match='incomplete batch'):
            model.predict_detections(frames[:1])
    finally:
        model.close()


def test_device_frame_adapter_checks_mode_and_batch_completeness():
    model = make_model()
    try:
        assert model.predict_device_detections([object()] * 5) == [[]] * 5
        model.engine.short_output = True
        with pytest.raises(RuntimeError, match='incomplete batch'):
            model.predict_device_detections([object()])
    finally:
        model.close()
    model = make_model(asynchronous=False)
    try:
        with pytest.raises(RuntimeError, match='asynchronous'):
            model.predict_device_detections([object()])
    finally:
        model.close()


def test_admission_cannot_silently_fall_back_to_python(monkeypatch):
    monkeypatch.setenv('GPU_RESOURCE_INFERENCE_SLOTS', '1')
    with pytest.raises(ValueError, match='explicit Rust'):
        create_rust_model(
            Path('fixture.pt'), WorkerSettings(backend='auto'),
            device='cuda:0', rect=False,
        )
    with pytest.raises(ValueError, match='at least 64'):
        GpuResourceSettings(memory_limit_mb=1)


def test_async_requires_supported_native_api():
    with pytest.raises(RuntimeError, match='Rebuild'):
        RustYoloModel(
            SimpleNamespace(), Path('fixture.engine'),
            WorkerSettings(backend='rust', native_async=True),
            device='cuda:0', rect=False,
        )
