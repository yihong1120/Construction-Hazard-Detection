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


def test_release_nvenc_session_never_makes_active_count_negative() -> None:
    """Releasing an unreserved session keeps the shared counter safe."""
    previous_sessions = nvenc_session._active_sessions
    nvenc_session._active_sessions = 0
    try:
        nvenc_session.release_nvenc_session()
        assert nvenc_session._active_sessions == 0

        nvenc_session._active_sessions = 1
        nvenc_session.release_nvenc_session()
        assert nvenc_session._active_sessions == 0
    finally:
        nvenc_session._active_sessions = previous_sessions


def test_nvenc_session_invalid_environment_uses_default(
        monkeypatch: Any,
) -> None:
    """An invalid process budget does not break media publishing."""
    monkeypatch.setenv('MEDIA_NVENC_MAX_SESSIONS', 'not-a-number')

    assert nvenc_session._max_sessions() == nvenc_session._default_max_sessions
