from __future__ import annotations

import hashlib
from pathlib import Path


def model_file_checksum(model_path: Path) -> str:
    """Calculate a streaming SHA-256 checksum without loading the model."""
    digest = hashlib.sha256()
    with model_path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
