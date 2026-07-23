from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time

_turn_default_ttl_seconds = 600
_default_stun_servers = ('stun:stun.l.google.com:19302',)


def _get_env_int(name: str, default: int) -> int:
    """Read an integer environment setting."""
    try:
        return int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


def _get_env_csv(name: str) -> list[str]:
    """Read a comma-separated environment variable into non-empty strings."""
    value = os.getenv(name, '')
    return [item.strip() for item in value.split(',') if item.strip()]


def _build_turn_rest_credential(
    shared_secret: str,
    user_id: str,
    ttl_seconds: int,
) -> tuple[str, str]:
    """Build a coturn REST API short-lived username and credential."""
    expires_at = int(time.time()) + max(60, ttl_seconds)
    username = f'{expires_at}:{user_id}'
    digest = hmac.new(
        shared_secret.encode('utf-8'),
        # coturn TURN REST credentials require this keyed HMAC-SHA1 format.
        # codeql[py/weak-sensitive-data-hashing]
        username.encode('utf-8'),
        hashlib.sha1,
    ).digest()
    credential = base64.b64encode(digest).decode('ascii')
    return username, credential


def get_public_ice_servers(
    user_id: str | None = None,
) -> list[dict[str, object]]:
    """Return ICE servers that Flutter/web clients should use."""
    stun_urls = _get_env_csv('STREAMING_WEBRTC_STUN_URLS')
    if not stun_urls:
        stun_urls = list(_default_stun_servers)

    ice_servers: list[dict[str, object]] = [{'urls': stun_urls}]
    turn_urls = _get_env_csv('STREAMING_WEBRTC_TURN_URLS')
    if turn_urls:
        turn_server: dict[str, object] = {'urls': turn_urls}
        shared_secret = os.getenv('STREAMING_WEBRTC_TURN_SHARED_SECRET', '')
        if shared_secret:
            username, credential = _build_turn_rest_credential(
                shared_secret=shared_secret,
                user_id=user_id or 'webrtc-viewer',
                ttl_seconds=_get_env_int(
                    'STREAMING_WEBRTC_TURN_TTL_SECONDS',
                    _turn_default_ttl_seconds,
                ),
            )
        else:
            username = os.getenv('STREAMING_WEBRTC_TURN_USERNAME', '')
            credential = os.getenv('STREAMING_WEBRTC_TURN_CREDENTIAL', '')
        if username:
            turn_server['username'] = username
        if credential:
            turn_server['credential'] = credential
        ice_servers.append(turn_server)
    return ice_servers
