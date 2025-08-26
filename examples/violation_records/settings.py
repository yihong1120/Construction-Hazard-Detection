from __future__ import annotations

from pathlib import Path

# Centralised static directory configuration for violation records and assets.
# Use a single source of truth across the app instead of hardcoding 'static'
# in multiple files. This path is relative to the project root when running
# from the repo, but you can also set it to an absolute path if needed.

# You can change this to point elsewhere (e.g., an NFS mount or volume) and
# all dependent modules will follow.
STATIC_DIR: Path = Path('static')
