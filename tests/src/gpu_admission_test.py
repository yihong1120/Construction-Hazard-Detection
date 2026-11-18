from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.gpu_resources import try_acquire_resource
from src.native_inference import RustYoloModel
from src.settings.inference import WorkerSettings


class EngineDouble:
    def __init__(self, *args, **kwargs):
        self.max_batch = 8
        self.closed = 0

    def predict(self, frames):
        return [[] for _ in frames]

    def close(self):
        self.closed += 1


def model(settings=None, engine=EngineDouble):
    return RustYoloModel(
        SimpleNamespace(
            TensorRTEngine=engine,
        ),
        Path('fixture.engine'),
        settings or WorkerSettings(
            backend='rust',
        ),
        device='cuda:0',
        rect=False,
    )


def test_memory_admission_recovers_on_close_and_failed_constructor(
        monkeypatch,
        tmp_path,
):
    monkeypatch.setenv('GPU_RESOURCE_DIRECTORY', str(tmp_path))
    monkeypatch.setenv('GPU_RESOURCE_MEMORY_LIMIT_MB', '128')
    settings = WorkerSettings(backend='rust', gpu_memory_reservation_mb=128)
    first = model(settings)
    with pytest.raises(RuntimeError, match='budget exhausted'):
        model(settings)
    first.close()
    first.close()
    with pytest.raises(RuntimeError, match='load failed'):
        model(settings, MagicMock(side_effect=RuntimeError('load failed')))
    recovered = try_acquire_resource('memory', 2, 2)
    assert recovered is not None
    recovered.close()
    with pytest.raises(ValueError, match='explicit Rust'):
        model(WorkerSettings(backend='auto'))


def test_admission_timeout_does_not_leak_permits(monkeypatch, tmp_path):
    monkeypatch.setenv('GPU_RESOURCE_DIRECTORY', str(tmp_path))
    monkeypatch.setenv('GPU_RESOURCE_INFERENCE_SLOTS', '1')
    lease = try_acquire_resource('inference', 1)
    adapter = model(
        WorkerSettings(
            backend='rust',
            gpu_admission_timeout_seconds=.01,
        ),
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    try:
        with pytest.raises(TimeoutError, match='admission'):
            adapter.predict_detections([frame])
        lease.close()
        assert adapter.predict_detections([frame]) == [[]]
        recovered = try_acquire_resource('inference', 1)
        assert recovered is not None
        recovered.close()
    finally:
        lease.close()
        adapter.close()
