from __future__ import annotations

import re
import unicodedata

_filename_strip_re = re.compile(r'[^A-Za-z0-9_.-]')
_windows_device_files = {
    'CON',
    'PRN',
    'AUX',
    'NUL',
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def sanitize_filename(value: str) -> str:
    """Return an ASCII-only filename safe for local filesystem joins."""
    filename = unicodedata.normalize('NFKD', value)
    filename = filename.encode('ascii', 'ignore').decode('ascii')
    filename = filename.replace('/', ' ').replace('\\', ' ')
    filename = '_'.join(filename.split())
    filename = _filename_strip_re.sub('', filename).strip('._')
    if filename.split('.', 1)[0].upper() in _windows_device_files:
        filename = f"_{filename}"
    return filename
