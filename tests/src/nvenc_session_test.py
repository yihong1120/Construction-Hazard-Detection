from __future__ import annotations

from typing import Any

from src import nvenc_session


def test_nvenc_session_budget_limits_concurrent_publishers(
        monkeypatch: Any,
) -> None:
    """The local budget prevents ffmpeg from exceeding NVENC sessions."""
    monkeypatch.setenv('MEDIA_NVENC_MAX_SESSIONS', '2')
    previous_sessions = nvenc_session._active_sessions
    nvenc_session._active_sessions = 0
    try:
        assert nvenc_session.try_acquire_nvenc_session() is True
        assert nvenc_session.try_acquire_nvenc_session() is True
        assert nvenc_session.try_acquire_nvenc_session() is False
    finally:
        nvenc_session._active_sessions = previous_sessions
