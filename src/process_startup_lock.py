from __future__ import annotations

import fcntl
import os
import tempfile
from pathlib import Path


class ProcessStartupLock:
    """Serialize loading with a lock the OS releases after process death."""

    def __init__(self):
        descriptor, name = tempfile.mkstemp(
            prefix='construction-model-startup-',
        )
        os.close(descriptor)
        self.path = Path(name)
        self._file = None

    def __enter__(self):
        self._file = self.path.open('rb')
        try:
            fcntl.flock(self._file, fcntl.LOCK_EX)
        except BaseException:
            self._file.close()
            self._file = None
            raise
        return self

    def __exit__(self, *_):
        if self._file is not None:
            self._file.close()
            self._file = None

    def close(self):
        self.path.unlink(missing_ok=True)
