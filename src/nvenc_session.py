from __future__ import annotations

import os
from threading import local
from threading import Lock

from src.gpu_resources import GpuLease
from src.gpu_resources import try_acquire_resource
from src.settings.media import NvencSettings

_lock = Lock()
_active_sessions = 0
_session_budget = 0
_reservations: list[GpuLease] = []
_pending = local()


def _after_fork():
    global _active_sessions, _lock
    _lock = Lock()
    _active_sessions = 0
    _reservations.clear()
    _pending.value = None


os.register_at_fork(after_in_child=_after_fork)


def try_acquire_nvenc_session() -> bool:
    """Reserve one host-wide session; descriptor may pass to encoder child."""
    global _active_sessions, _session_budget
    with _lock:
        _pending.value = None
        if _active_sessions == 0:
            for lease in _reservations:
                lease.close()
            _reservations.clear()
            _session_budget = _max_sessions()
        lease = try_acquire_resource('nvenc', _session_budget)
        if lease is None:
            return False
        _reservations.append(lease)
        _pending.value = lease
        _active_sessions += 1
        return True


def take_nvenc_reservation():
    lease = getattr(_pending, 'value', None)
    _pending.value = None
    return lease


def release_nvenc_session(reservation=None) -> None:
    global _active_sessions
    with _lock:
        if reservation is None:
            reservation = _reservations[-1] if _reservations else None
        if reservation is not None:
            if reservation not in _reservations:
                return
            _reservations.remove(reservation)
            reservation.close()
        _active_sessions = max(0, _active_sessions - 1)


def _max_sessions() -> int:
    return NvencSettings().max_sessions
