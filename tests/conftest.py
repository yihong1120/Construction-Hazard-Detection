"""Provide an isolated encryption secret for BFF storage tests."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def bff_encryption_secret(monkeypatch):
    monkeypatch.setenv(
        'BFF_TOKEN_ENCRYPTION_KEY',
        'test-only-bff-encryption-secret',
    )
