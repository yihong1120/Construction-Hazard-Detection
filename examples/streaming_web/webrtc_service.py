from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time


def _build_turn_rest_credential(
    shared_secret: str,
    user_id: str,
    ttl_seconds: int,
) -> tuple[str, str]:
    """Build a coturn REST API short-lived username and credential.

    Args:
        shared_secret: TURN REST shared secret configured for coturn.
        user_id: Authenticated user identifier embedded in the username.
        ttl_seconds: Requested credential lifetime in seconds.

    Returns:
        Coturn username and HMAC-SHA1 credential pair.
    """
    expires_at = int(time.time()) + max(60, ttl_seconds)
    username = f"{expires_at}:{user_id}"
    digest = hmac.new(
        shared_secret.encode('utf-8'),
        # coturn TURN REST credentials require this keyed HMAC-SHA1 format.
        username.encode('utf-8'),
        hashlib.sha1,
    ).digest()
    credential = base64.b64encode(digest).decode('ascii')
    return username, credential


def get_public_ice_servers(
    user_id: str,
) -> list[dict[str, object]]:
    """Return configured ICE servers for an authenticated user.

    Args:
        user_id: Authenticated user identifier for TURN credentials.

    Returns:
        STUN configuration and, when configured, TURN configuration.

    Raises:
        KeyError: If a required ICE environment variable is absent.
        ValueError: If configured ICE values are empty or inconsistent.
    """
    stun_urls = [
        item.strip()
        for item in os.environ['STREAMING_WEBRTC_STUN_URLS'].split(',')
        if item.strip()
    ]
    if not stun_urls:
        raise ValueError('STREAMING_WEBRTC_STUN_URLS must not be empty')

    ice_servers: list[dict[str, object]] = [{'urls': stun_urls}]
    # TURN remains optional; STUN-only deployments do not need a shared secret.
    raw_turn_urls = os.environ.get('STREAMING_WEBRTC_TURN_URLS')
    if raw_turn_urls is None:
        return ice_servers

    turn_urls = [
        item.strip() for item in raw_turn_urls.split(',') if item.strip()
    ]
    if not turn_urls:
        raise ValueError('STREAMING_WEBRTC_TURN_URLS must not be empty')
    if not user_id:
        raise ValueError('user_id is required when TURN is configured')

    shared_secret = os.environ['STREAMING_WEBRTC_TURN_SHARED_SECRET']
    if not shared_secret:
        raise ValueError(
            'STREAMING_WEBRTC_TURN_SHARED_SECRET must not be empty',
        )
    username, credential = _build_turn_rest_credential(
        shared_secret=shared_secret,
        user_id=user_id,
        ttl_seconds=int(os.environ['STREAMING_WEBRTC_TURN_TTL_SECONDS']),
    )
    ice_servers.append(
        {
            'urls': turn_urls,
            'username': username,
            'credential': credential,
        },
    )
    return ice_servers
