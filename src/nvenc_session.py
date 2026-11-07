from __future__ import annotations

from threading import Lock

from src.settings.media import NvencSettings

_lock = Lock()
_active_sessions = 0
_session_budget = 0


def try_acquire_nvenc_session() -> bool:
    """Reserve one local NVENC session without exceeding the safe budget."""
    global _active_sessions, _session_budget
    with _lock:
        if _active_sessions == 0:
            _session_budget = _max_sessions()
        if _active_sessions >= _session_budget:
            return False
        _active_sessions += 1
        return True


def release_nvenc_session() -> None:
    """Release one locally reserved NVENC session."""
    global _active_sessions
    with _lock:
        _active_sessions = max(0, _active_sessions - 1)


def _max_sessions() -> int:
    """Return a conservative per-process NVENC session budget."""
    return NvencSettings().max_sessions
