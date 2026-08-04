from __future__ import annotations

import os
from threading import Lock


_default_max_sessions = 6
_lock = Lock()
_active_sessions = 0


def try_acquire_nvenc_session() -> bool:
    """Reserve one local NVENC session without exceeding the safe budget."""
    global _active_sessions
    with _lock:
        if _active_sessions >= _max_sessions():
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
    try:
        return max(
            1,
            int(
                os.getenv(
                    'MEDIA_NVENC_MAX_SESSIONS',
                    str(_default_max_sessions),
                ),
            ),
        )
    except ValueError:
        return _default_max_sessions
