"""Crash-recoverable file-lock leases; Rust and Python share one protocol."""
from __future__ import annotations

import fcntl
import os
import re
from pathlib import Path
from typing import BinaryIO
from weakref import WeakSet

from src.native_kernels import optional_kernel
from src.settings.gpu_resources import GpuResourceSettings

_live_leases: WeakSet = WeakSet()


class GpuLease:
    def __init__(self, inner):
        self.inner = inner
        _live_leases.add(self)

    def filenos(self):
        return self.inner.filenos() if self.inner is not None else []

    def close(self):
        if self.inner is not None:
            self.inner.close()
            self.inner = None

    def __del__(self):
        self.close()


def _forget_inherited_leases():
    for lease in list(_live_leases):
        lease.close()


os.register_at_fork(after_in_child=_forget_inherited_leases)


class FileLease:
    def __init__(self, files):
        self.files = files

    def filenos(self):
        return [file.fileno() for file in self.files]

    def close(self):
        # Closing inherited descriptors must not unlock a parent's flock.
        for file in self.files:
            file.close()
        self.files.clear()

    def __del__(self):
        self.close()


def _open_lock(path: Path):
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW
    return os.fdopen(os.open(path, flags, 0o600), 'r+b', buffering=0)


def _try_lock(file):
    try:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def _python_acquire(directory: str, key: str, capacity: int, units: int):
    root = Path(directory)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    held: list[BinaryIO] = []
    with _open_lock(root / f'{key}.control') as control:
        fcntl.flock(control, fcntl.LOCK_EX)
        stored = control.read(32).strip()
        old_capacity = int(stored) if stored else capacity
        if not 1 <= old_capacity <= 16384:
            raise RuntimeError('Corrupt GPU resource capacity')
        if old_capacity != capacity:
            for index in range(old_capacity):
                file = _open_lock(root / f'{key}.{index}')
                if not _try_lock(file):
                    file.close()
                    FileLease(held).close()
                    raise RuntimeError(
                        'GPU budget differs from active processes',
                    )
                held.append(file)
            FileLease(held).close()
        control.seek(0)
        control.write(str(capacity).encode())
        control.truncate()
        for index in range(capacity):
            file = _open_lock(root / f'{key}.{index}')
            if _try_lock(file):
                held.append(file)
                if len(held) == units:
                    return FileLease(held)
            else:
                file.close()
    FileLease(held).close()
    return None


def try_acquire_resource(kind: str, capacity: int, units: int = 1):
    if not re.fullmatch(r'[a-z][a-z0-9_-]{0,40}', kind):
        raise ValueError('Invalid GPU resource kind')
    if not 1 <= units <= capacity <= 16384:
        raise ValueError('Invalid GPU resource capacity or units')
    settings = GpuResourceSettings()
    key = f'{settings.device_id}.{kind}'
    native = optional_kernel('acquire_resource', 'RESOURCE_API_VERSION')
    acquire = native if native is not None else _python_acquire
    lease = acquire(str(settings.directory), key, capacity, units)
    return GpuLease(lease) if lease is not None else None
