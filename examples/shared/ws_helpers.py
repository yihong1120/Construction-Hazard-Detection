from __future__ import annotations

import json
import os
import time

from fastapi import WebSocket
from redis import asyncio as redis_async

from examples.shared.ws_auth import authenticate_websocket
from examples.shared.ws_auth import SettingsLike
from examples.shared.ws_auth import WS_MAX_SESSION_SECONDS
from examples.shared.ws_utils import _safe_websocket_send_json
from examples.shared.ws_utils import _safe_websocket_send_text


def get_auto_register_jti(env_var: str = 'WS_AUTO_REGISTER_JTI') -> bool:
    """
    Read the flag controlling automatic JTI registration from the env.

    Args:
        env_var: Name of the environment variable to read.

    Returns:
        Whether missing JTIs should be auto-registered for WebSocket tokens.
    """
    return os.getenv(env_var, 'false').lower() == 'true'


def start_session_timer() -> float:
    """Start a monotonic timer for measuring WebSocket session lifetime."""
    return time.monotonic()


def session_timeout_payload() -> dict[str, str]:
    """
    Build a standard JSON payload used when a session times out.

    Returns:
        A JSON-serialisable dictionary describing the session timeout.
    """
    return {
        'status': 'closing',
        'reason': 'session_timeout',
        'message': 'WebSocket session reached 5-minute limit.',
    }


async def check_and_maybe_close_on_timeout(
    websocket: WebSocket,
    session_start: float,
    client_label: str,
    *,
    use_text: bool = False,
) -> bool:
    """
    Check a session's elapsed time and close the connection if exceeded.

    Args:
        websocket: Active WebSocket connection to check.
        session_start: Monotonic timestamp of the session start (post-auth).
        client_label: Human-readable identifier for logs
            (e.g., ``"IP (user)"``).
        use_text: If ``True``, send a text frame rather than a JSON frame.

    Returns:
        ``True`` if the connection was closed due to a timeout; ``False``
        otherwise.
    """
    if time.monotonic() - session_start < WS_MAX_SESSION_SECONDS:
        return False

    print(
        (
            f"[{client_label}] Session timeout reached "
            f"(>{int(WS_MAX_SESSION_SECONDS)}s), closing"
        ),
    )

    payload = session_timeout_payload()
    if use_text:
        await _safe_websocket_send_text(
            websocket,
            json.dumps(payload),
            client_label,
        )
    else:
        await _safe_websocket_send_json(
            websocket,
            payload,
            client_label,
        )

    try:
        await websocket.close(code=1000, reason='Session timeout (5 minutes)')
    finally:
        return True


def log_every_n(
    prefix: str,
    count: int,
    unit: str = 'frames',
    n: int = 100,
) -> None:
    """
    Log a simple progress message every ``n`` items processed.

    Args:
        prefix: Text prefix to contextualise the message.
        count: Current number of processed items.
        unit: Unit label printed after the count (defaults to ``"frames"``).
        n: Print frequency; set to ``0`` or a negative number to disable.
    """
    if n > 0 and count % n == 0:
        print(f"{prefix}: Processed {count} {unit}")


async def authenticate_ws_or_none(
    websocket: WebSocket,
    rds: redis_async.Redis,
    settings: SettingsLike,
    *,
    auto_register_jti: bool,
    client_tag: str,
) -> tuple[str, dict[str, object]] | tuple[None, None]:
    """
    Authenticate a WebSocket connection and return the user information.

    Args:
        websocket: Active WebSocket connection to authenticate.
        rds: Redis connection used by the authentication cache.
        settings: Configuration required to verify JWTs.
        auto_register_jti: Whether to auto-register unknown JTIs.
        client_tag: Label used in log lines for correlation.

    Returns:
        A 2-tuple of ``(username, payload)`` on success, or ``(None, None)``
        when authentication fails and the connection has been closed.
    """
    try:
        username, jti, payload = await authenticate_websocket(
            websocket,
            rds,
            settings,
            auto_register_jti=auto_register_jti,
            client_tag=client_tag,
        )
        return username, payload
    except SystemExit:
        return None, None
