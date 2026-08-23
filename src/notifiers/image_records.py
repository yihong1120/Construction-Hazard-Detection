from __future__ import annotations

import json
import logging
import os
from pathlib import Path


class ImageRecordStore:
    """Read and write image-upload timestamps without leaking I/O details."""

    def __init__(self, path: str, logger: logging.Logger) -> None:
        """Perform init.

        Args:
            path: Value used by this callable.
            logger: Value used by this callable.
        """
        self.path = path
        self.logger = logger

    def load(self) -> dict[str, str]:
        """Return stored image records, falling back safely to an empty map."""
        try:
            if os.path.exists(self.path):
                with open(self.path, encoding='utf-8') as file:
                    records = json.load(file)
                    if isinstance(records, dict):
                        return records
                self.logger.warning('Image record file is not a JSON object')
        except Exception as exc:  # Record persistence must never break alerts.
            self.logger.error('Failed to load image records: %s', exc)
        return {}

    def save(self, records: dict[str, str]) -> None:
        """Atomically persist image records and log recoverable failures."""
        try:
            destination = Path(self.path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(f'{destination.suffix}.tmp')
            with temporary.open('w', encoding='utf-8') as file:
                json.dump(records, file)
            temporary.replace(destination)
        except Exception as exc:  # Record persistence must never break alerts.
            self.logger.error('Failed to save image records: %s', exc)
