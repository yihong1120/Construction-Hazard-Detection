from __future__ import annotations

import redis.asyncio as redis

from examples.streaming_web.metadata_keys import metadata_key_stream_id
from examples.streaming_web.metadata_keys import metadata_key_stream_name
from examples.streaming_web.schemas import FrameOutData


def _extract_stream_id(redis_key: str) -> str:
    """Extract the encoded stream identifier from a metadata key.

    Args:
        redis_key: Canonical Redis metadata key.

    Returns:
        Encoded identifier for the configured stream.
    """
    return metadata_key_stream_id(redis_key)


def _decode_stream_name(redis_key: str) -> str:
    """Decode the configured stream name from a metadata key.

    Args:
        redis_key: Canonical Redis metadata key.

    Returns:
        Original configured stream name.
    """
    return metadata_key_stream_name(redis_key)


def _build_metadata_record(
    redis_key: str,
    message_id: bytes,
    data: dict[bytes, bytes],
) -> FrameOutData:
    """Build the public metadata record from one Redis Stream message.

    Args:
        redis_key: Canonical key from which the message was read.
        message_id: Redis-assigned identifier for the message.
        data: Binary field map stored by the frame producer.

    Returns:
        Decoded frame metadata suitable for client delivery.
    """
    return {
        'id': message_id.decode('utf-8'),
        'key': _decode_stream_name(redis_key),
        'stream_id': _extract_stream_id(redis_key),
        'redis_key': redis_key,
        'has_warning': data[b'has_warning'] == b'1',
    }


async def fetch_latest_metadata_for_key(
    rds: redis.Redis,
    redis_key: str,
    last_id: str,
    block_ms: int = 2000,
) -> FrameOutData | None:
    """Wait for the next compact metadata message for one Redis key.

    Args:
        rds: Redis connection used for the blocking stream read.
        redis_key: Canonical stream key to consume.
        last_id: Most recently delivered Redis message identifier.
        block_ms: Maximum wait time in milliseconds.

    Returns:
        The next decoded frame record, or ``None`` when no new record arrives.
    """
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
    message_id_str = message_id.decode('utf-8')
    if message_id_str == last_id:
        return None

    return _build_metadata_record(
        redis_key=redis_key,
        message_id=message_id,
        data=data,
    )
