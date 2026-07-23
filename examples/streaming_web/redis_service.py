from __future__ import annotations

import base64
import re
from typing import Any

from examples.streaming_web.utils import Utils

_stream_scan_count = 500
_metadata_key_pattern = re.compile(
    r'stream_metadata:([A-Za-z0-9\-_]+=*)\|([A-Za-z0-9\-_]+=*)',
)
_stream_name_cache_max_items = 512
_stream_name_cache: dict[str, str] = {}


def build_metadata_key(site: str, stream_name: str) -> str:
    """Build the Redis key used for compact live-view metadata."""
    return f"stream_metadata:{Utils.encode(site)}|{Utils.encode(stream_name)}"


def _extract_stream_id(redis_key: str) -> str:
    """Return the encoded stream-name segment from a metadata key."""
    match = _metadata_key_pattern.match(redis_key)
    if match:
        return match.group(2)
    splitted = redis_key.split('|')
    return splitted[-1] if len(splitted) >= 2 else ''


def _decode_stream_name(redis_key: str) -> str:
    """Decode the stream-name portion of a metadata key."""
    splitted = redis_key.split('|')
    if len(splitted) < 2:
        return 'Unknown'

    encoded_name = splitted[-1]
    cached = _stream_name_cache.get(encoded_name)
    if cached is not None:
        return cached

    try:
        decoded = base64.urlsafe_b64decode(encoded_name).decode('utf-8')
    except Exception:
        return 'Unknown'
    if len(_stream_name_cache) >= _stream_name_cache_max_items:
        _stream_name_cache.clear()
    _stream_name_cache[encoded_name] = decoded
    return decoded


def _decode_bytes(value: bytes | None) -> str:
    """Decode a Redis byte value into text, defaulting to an empty string."""
    return value.decode('utf-8') if value else ''


def _build_metadata_record(
    redis_key: str,
    message_id: bytes | str,
    data: dict[bytes, bytes],
) -> dict[str, Any]:
    """Build a compact metadata record from a Redis stream message."""
    message_id_str = (
        message_id.decode('utf-8')
        if isinstance(message_id, bytes)
        else message_id
    )
    return {
        'id': message_id_str,
        'key': _decode_stream_name(redis_key),
        'stream_id': _extract_stream_id(redis_key),
        'redis_key': redis_key,
        'has_warning': _decode_bytes(data.get(b'has_warning')),
    }


async def get_metadata_keys_for_label(rds: Any, label: str) -> list[str]:
    """Retrieve all compact live metadata keys in Redis for a given label."""
    encoded_label = base64.urlsafe_b64encode(
        label.encode('utf-8'),
    ).decode('utf-8')
    cursor: int = 0
    matching_keys: list[str] = []

    while True:
        cursor, keys = await rds.scan(
            cursor=cursor,
            match=f"stream_metadata:{encoded_label}|*",
            count=_stream_scan_count,
        )

        for key in keys:
            decoded_key: str = key.decode('utf-8', errors='ignore')
            if _metadata_key_pattern.match(decoded_key):
                matching_keys.append(decoded_key)

        if cursor == 0:
            break

    return sorted(matching_keys)


async def fetch_latest_metadata_for_key(
    rds: Any,
    redis_key: str,
    last_id: str,
    block_ms: int = 2000,
) -> dict[str, Any] | None:
    """Wait for the next compact metadata message for one Redis key."""
    messages = await rds.xread(
        {redis_key: last_id},
        count=1,
        block=block_ms,
    )
    if not messages:
        return None

    _stream_key, stream_messages = messages[0]
    if not stream_messages:
        return None

    message_id, data = stream_messages[0]
    message_id_str = (
        message_id.decode('utf-8')
        if isinstance(message_id, bytes)
        else message_id
    )
    if message_id_str == last_id:
        return None

    return _build_metadata_record(
        redis_key=redis_key,
        message_id=message_id,
        data=data,
    )
