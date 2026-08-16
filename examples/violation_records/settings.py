from __future__ import annotations

from pathlib import Path

# A single source prevents upload, image, and thumbnail paths from drifting.
# Deployments may replace this relative path with a mounted absolute volume.
STATIC_DIR: Path = Path('static')
