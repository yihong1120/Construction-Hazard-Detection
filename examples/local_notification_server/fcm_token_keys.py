from __future__ import annotations


def _token_index_key(user_id: int) -> str:
    """Build the Redis set key for a user's token hashes.

    Args:
        user_id: Token owner identifier.

    Returns:
        Redis key indexing metadata hashes for the user's tokens.
    """
    return f"fcm_token_index:{user_id}"


def _token_meta_key(user_id: int, token_hash: str) -> str:
    """Build the Redis hash key for one token's non-sensitive metadata.

    Args:
        user_id: Token owner identifier.
        token_hash: SHA-256 digest of the raw FCM token.

    Returns:
        Redis key containing metadata for the token registration.
    """
    return f"fcm_token_meta:{user_id}:{token_hash}"


def _token_cache_ready_key(user_id: int) -> str:
    """Build the marker that distinguishes a cold token cache from no
    tokens."""
    return f"fcm_tokens_ready:{user_id}"
