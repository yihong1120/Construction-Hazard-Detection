from __future__ import annotations

import base64
import re
from typing import Final


# Use a full-match pattern so scans cannot mistake unrelated legacy keys for
# metadata records with the trusted encoded-site and encoded-stream contract.
_metadata_key_pattern: Final[re.Pattern[str]] = re.compile(
    r'stream_metadata:([A-Za-z0-9\-_]+=*)\|([A-Za-z0-9\-_]+=*)',
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


def build_metadata_key(site: str, stream_name: str) -> str:
    """Build the Redis stream key used for live-view metadata.

    Args:
        site: Site label that owns the stream.
        stream_name: Configured stream name.

    Returns:
        Canonical encoded Redis metadata key.
    """
    return f'stream_metadata:{encode_stream_id(site)}|{encode_stream_id(stream_name)}'


def build_metadata_key_from_stream_id(site: str, stream_id: str) -> str:
    """Build a metadata key from an already encoded stream identifier.

    Args:
        site: Site label that owns the stream.
        stream_id: Canonical encoded stream identifier.

    Returns:
        Canonical Redis metadata key.
    """
    return f'stream_metadata:{encode_stream_id(site)}|{stream_id}'


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
