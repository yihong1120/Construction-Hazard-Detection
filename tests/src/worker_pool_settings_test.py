from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.settings.inference import WorkerPoolSettings


@pytest.mark.parametrize(
    'field',
    [
        'cameras_per_engine',
        'queue_size',
        'result_queue_size',
    ],
)
@pytest.mark.parametrize('value', [0, -1])
def test_zero_or_negative_pool_capacity_is_rejected(field, value):
    with pytest.raises(ValidationError):
        WorkerPoolSettings(**{field: value})


def test_csv_settings_preserve_explicit_per_model_limits(monkeypatch):
    monkeypatch.setenv('YOLO_WORKER_DEVICES', 'cpu, cuda:1')
    monkeypatch.setenv(
        'YOLO_WORKER_CAMERAS_PER_ENGINE_BY_MODEL',
        'YOLO26N=3,YOLO26M=1',
    )
    settings = WorkerPoolSettings()
    assert settings.devices == ('cpu', 'cuda:1')
    assert settings.cameras_per_engine_by_model == {'yolo26n': 3, 'yolo26m': 1}
    monkeypatch.setenv('YOLO_WORKER_CAMERAS_PER_ENGINE_BY_MODEL', 'yolo26n=0')
    with pytest.raises(ValidationError):
        WorkerPoolSettings()
