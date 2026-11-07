from __future__ import annotations

from collections.abc import Mapping


def stream_identity(config: Mapping[str, object]) -> str:
    """Use the database identity across capture, worker IPC and cleanup."""
    stream_id = config.get('stream_id')
    if stream_id is not None:
        return f'id:{stream_id}'
    # Legacy programmatic callers do not yet supply database IDs.
    return f"{config['site']}|{config['stream_name']}"
