from __future__ import annotations

import logging
import os
from collections.abc import Awaitable
from collections.abc import Mapping
from typing import NoReturn
from typing import Protocol

from fastapi import HTTPException
from fastapi import Request
from redis.asyncio import Redis
from starlette.types import Scope

from examples.auth.jwt_config import jwt_access


class WebSocketLike(Protocol):
    """A minimal protocol describing the WebSocket operations we use."""

    @property
    def scope(self) -> Scope:
        """Return the ASGI connection scope."""

    @property
    def headers(self) -> Mapping[str, str]:
        """Return request headers."""

    @property
    def query_params(self) -> Mapping[str, str]:
        """Return request query parameters."""

    def close(self, code: int, reason: str) -> Awaitable[None]:
        """Close the WebSocket connection with a code and textual reason."""


# Shared defaults/configuration
WS_MAX_SESSION_SECONDS: float = float(
    os.getenv('WS_MAX_SESSION_SECONDS', '1800'),
)
logger = logging.getLogger(__name__)


def extract_token_from_ws(websocket: WebSocketLike) -> str | None:
    """Extract a JWT from a WebSocket request.

    Args:
        websocket: The WebSocket-like object containing headers and query
            parameters.

    Returns:
        The raw JWT string if found; otherwise ``None``.
    """
    # Header first
    auth = websocket.headers.get('authorization')
    if auth and auth.lower().startswith('bearer '):
        return auth.split(' ', 1)[1]
    # Then query param
    return websocket.query_params.get('token')


async def _fail_ws(
    websocket: WebSocketLike,
    code: int,
    reason: str,
    tag: str,
    log_msg: str,
    exit_reason: str,
) -> NoReturn:
    """Log, close the websocket, and raise SystemExit with a short code.

    Args:
        websocket: The WebSocket-like request, used to close the connection.
        code: The WebSocket close code (e.g. ``1008`` for Policy Violation).
        reason: The textual reason sent with the close frame.
        tag: Optional label used in log messages for easier tracing.
        log_msg: The message to log before closing.
        exit_reason: The short string used as the SystemExit reason.

    Raises:
        SystemExit: Always raised after closing the WebSocket.

    Notes:
        - The WebSocket is closed before raising SystemExit, so the caller
          does not need to do so.

    Returns:
        None. This function does not return; it always raises SystemExit.
    """
    logger.warning(
        'WebSocket authentication failed tag=%s reason=%s',
        tag,
        log_msg,
    )
    await websocket.close(code=code, reason=reason)
    raise SystemExit(exit_reason)


def get_model_key_from_ws(websocket: WebSocketLike) -> str | None:
    """Extract the model key for YOLO WebSocket endpoints.

    Args:
        websocket: The WebSocket-like request.

    Returns:
        The model key if present; otherwise ``None``.
    """
    mk = websocket.headers.get('x-model-key')
    if mk:
        return mk
    return websocket.query_params.get('model')


async def authenticate_websocket(
    websocket: WebSocketLike,
    rds: Redis,
    client_tag: str | None = None,
) -> tuple[str, str, dict[str, object]]:
    """Authenticate a WebSocket client using a JWT.

    Args:
        websocket: The WebSocket connection.
        rds: Redis-like connection used by the user cache helpers.
        client_tag: Optional label used in log messages for easier tracing.

    Returns:
        A tuple ``(username, jti, payload)``.

    Raises:
        SystemExit: If the request is unauthenticated or the token is invalid.

    Notes:
        - On error, the WebSocket is closed with code ``1008`` (Policy
          Violation) and a descriptive reason. This function does not return
          in such cases.
        - Uses the same verifier, deployment binding, identity mapping and
          revocation checks as HTTP APIs.
    """
    tag = client_tag or '[WebSocket]'

    # Extract token
    token = extract_token_from_ws(websocket)
    if token is None or token == '':
        await _fail_ws(
            websocket,
            code=1008,
            reason='Missing authentication token',
            tag=tag,
            log_msg='No token found in header or query parameter',
            exit_reason='missing_token',
        )
    token_str: str = token

    # Reuse the HTTP policy. Only translate transport metadata; deployment
    # selection still uses the server-resolved host and external scheme.
    scope = websocket.scope.copy()
    scope['type'] = 'http'
    scope['method'] = 'GET'
    scope['scheme'] = {'ws': 'http', 'wss': 'https'}.get(
        scope.get('scheme', 'ws'), scope.get('scheme', 'ws'),
    )
    try:
        credentials = await jwt_access.authenticate_token(
            Request(scope), token_str, rds,
        )
    except HTTPException as exc:
        unavailable = exc.status_code >= 500
        await _fail_ws(
            websocket,
            code=1011 if unavailable else 1008,
            reason=(
                'Authentication service unavailable' if unavailable
                else 'Invalid token or deployment'
            ),
            tag=tag,
            log_msg=f'Authentication rejected (HTTP {exc.status_code})',
            exit_reason='authentication_unavailable' if unavailable
            else 'invalid_token',
        )
    payload = dict(credentials.payload)
    payload['subject'] = credentials.subject
    return (
        credentials.subject['username'],
        credentials.subject['jti'],
        payload,
    )
