from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def bff_encryption_secret(monkeypatch):
    monkeypatch.setenv(
        'BFF_TOKEN_ENCRYPTION_KEY',
        'test-only-bff-encryption-secret',
    )


@pytest.fixture(autouse=True)
def isolated_gpu_budget(monkeypatch, tmp_path):
    monkeypatch.setenv('GPU_RESOURCE_DIRECTORY', str(tmp_path / 'gpu-budget'))
