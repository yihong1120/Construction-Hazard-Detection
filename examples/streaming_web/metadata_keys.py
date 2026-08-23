from __future__ import annotations

import base64
import re
from typing import Final
from typing import Protocol


class MetadataRedis(Protocol):
    """Minimal Redis operations used by metadata-generation keys."""

    async def get(self, key: str) -> bytes | str | None:
        """Perform get.

        Args:
            key: Value used by this callable.

        Returns:
            The callable result.
        """
        ...

    async def incr(self, key: str) -> int:
        """Perform incr.

        Args:
            key: Value used by this callable.

        Returns:
            The callable result.
        """
        ...


# Use a full-match pattern so scans cannot mistake unrelated legacy keys for
# metadata records with the trusted encoded-site and encoded-stream contract.
_metadata_key_pattern: Final[re.Pattern[str]] = re.compile(
    r'stream_metadata:([A-Za-z0-9\-_]+=*)(?::g[1-9][0-9]*)?\|'
    r'([A-Za-z0-9\-_]+=*)',
)


def encode_stream_id(value: str) -> str:
    """Encode a site or stream name for the metadata-key contract.

    Args:
        value: UTF-8 logical name to encode.

    Returns:
        Padded URL-safe Base64 identifier.
    """
    return base64.urlsafe_b64encode(value.encode('utf-8')).decode('ascii')


def decode_stream_id(value: str) -> str:
    """Decode an identifier produced by :func:`encode_stream_id`.

    Args:
        value: Padded URL-safe Base64 identifier.

    Returns:
        Original UTF-8 logical name.
    """
    return base64.urlsafe_b64decode(value).decode('utf-8')


def metadata_site_generation_key(site: str) -> str:
    """Build the Redis generation counter key for one site."""
    return f"stream_metadata_generation:{encode_stream_id(site)}"


async def get_metadata_site_generation(
    rds: MetadataRedis,
    site: str,
) -> int:
    """Return the current metadata generation for a site.

    Missing counters map to generation zero, preserving the original key
    namespace until the site is first deleted.
    """
    raw_generation = await rds.get(metadata_site_generation_key(site))
    if raw_generation is None:
        return 0
    if isinstance(raw_generation, bytes):
        raw_generation = raw_generation.decode('ascii')
    try:
        return max(0, int(raw_generation))
    except (TypeError, ValueError):
        return 0


async def increment_metadata_site_generation(
    rds: MetadataRedis,
    site: str,
) -> int:
    """Invalidate all metadata for one site with one atomic increment."""
    return int(await rds.incr(metadata_site_generation_key(site)))


def build_metadata_key(
    site: str,
    stream_name: str,
    generation: int = 0,
) -> str:
    """Build the Redis stream key used for live-view metadata.

    Args:
        site: Site label that owns the stream.
        stream_name: Configured stream name.

    Returns:
        Canonical encoded Redis metadata key.
    """
    namespace = f"stream_metadata:{encode_stream_id(site)}"
    if generation:
        namespace = f"{namespace}:g{generation}"
    return f"{namespace}|{encode_stream_id(stream_name)}"


def build_metadata_key_from_stream_id(
    site: str,
    stream_id: str,
    generation: int = 0,
) -> str:
    """Build a metadata key from an already encoded stream identifier.

    Args:
        site: Site label that owns the stream.
        stream_id: Canonical encoded stream identifier.

    Returns:
        Canonical Redis metadata key.
    """
    namespace = f"stream_metadata:{encode_stream_id(site)}"
    if generation:
        namespace = f"{namespace}:g{generation}"
    return f"{namespace}|{stream_id}"


def metadata_key_stream_id(redis_key: str) -> str:
    """Extract the encoded stream identifier from a canonical key.

    Args:
        redis_key: Candidate Redis metadata key.

    Returns:
        Encoded stream identifier from the key.

    Raises:
        ValueError: If the key does not use the metadata-key contract.
    """
    match = _metadata_key_pattern.fullmatch(redis_key)
    if match is None:
        raise ValueError('invalid_metadata_key')
    return match.group(2)


def metadata_key_stream_name(redis_key: str) -> str:
    """Extract the decoded stream name from a canonical key.

    Args:
        redis_key: Canonical Redis metadata key.

    Returns:
        Decoded configured stream name.
    """
    return decode_stream_id(metadata_key_stream_id(redis_key))


def is_metadata_key(redis_key: str) -> bool:
    """Determine whether a Redis key uses the metadata-key contract.

    Args:
        redis_key: Candidate Redis key.

    Returns:
        ``True`` only for a fully canonical metadata key.
    """
    return _metadata_key_pattern.fullmatch(redis_key) is not None
