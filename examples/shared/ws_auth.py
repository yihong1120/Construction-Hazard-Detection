from __future__ import annotations

import logging
import os
from collections.abc import Awaitable
from collections.abc import Mapping
from typing import cast
from typing import NoReturn
from typing import Protocol

import jwt
from redis.asyncio import Redis

from examples.auth.cache import rate_limiter_service
from examples.auth.jwt_config import access_token_subject_from_payload
from examples.auth.token_cleanup import prune_user_cache
from examples.db_management.schemas.auth import AccessTokenSubject


class WebSocketLike(Protocol):
    """
    A minimal protocol describing the WebSocket operations we use.
    """

    @property
    def headers(self) -> Mapping[str, str]:
        """Return request headers."""

    @property
    def query_params(self) -> Mapping[str, str]:
        """Return request query parameters."""

    def close(self, code: int, reason: str) -> Awaitable[None]:
        """Close the WebSocket connection with a code and textual reason."""


class SettingsLike(Protocol):
    """Configuration needed to verify JWTs.

    Attributes:
        authjwt_secret_key: Secret used to verify signatures.
        ALGORITHM: JWT algorithm name (e.g. ``HS256``).
    """

    authjwt_secret_key: str
    ALGORITHM: str


# Shared defaults/configuration
WS_MAX_SESSION_SECONDS: float = float(
    os.getenv('WS_MAX_SESSION_SECONDS', '1800'),
)
AUTO_REGISTER_JTI: bool = os.getenv(
    'WS_AUTO_REGISTER_JTI', 'false',
).lower() == 'true'
logger = logging.getLogger(__name__)


def extract_token_from_ws(websocket: WebSocketLike) -> str | None:
    """
    Extract a JWT from a WebSocket request.

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
    """
    Log, close the websocket, and raise SystemExit with a short code.

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
    logger.warning('WebSocket authentication failed tag=%s reason=%s', tag, log_msg)
    await websocket.close(code=code, reason=reason)
    raise SystemExit(exit_reason)


async def _decode_or_fail(
    token: str,
    settings: SettingsLike,
    websocket: WebSocketLike,
    tag: str,
) -> dict[str, object]:
    """
    Decode a JWT or close the WebSocket on failure.

    Args:
        token: The raw JWT string.
        settings: Object providing ``authjwt_secret_key`` and ``ALGORITHM``.
        websocket: The WebSocket-like request, used to close on failure.
        tag: Optional label used in log messages for easier tracing.

    Returns:
        The decoded JWT payload as a dictionary.
    """
    try:
        # WebSocket services can be reached through a separate WSS upstream,
        # so their transport origin is not necessarily the API issuer. Read
        # the signed deployment issuer/audience first, then verify them with
        # the same values. HTTP endpoints additionally compare them with the
        # currently registered API origin before granting access.
        unsigned = jwt.decode(
            token,
            options={
                'verify_signature': False,
                'verify_exp': False,
                'verify_aud': False,
                'verify_iss': False,
            },
        )
        issuer = unsigned.get('iss')
        audience = unsigned.get('aud')
        decode_kwargs: dict[str, object] = {
            'algorithms': [settings.ALGORITHM],
        }
        if isinstance(issuer, str) and isinstance(audience, str):
            decode_kwargs['issuer'] = issuer
            decode_kwargs['audience'] = audience
        else:
            decode_kwargs['options'] = {
                'verify_aud': False,
                'verify_iss': False,
            }
        payload = cast(
            dict[str, object],
            jwt.decode(
                token,
                settings.authjwt_secret_key,
                **decode_kwargs,
            ),
        )
        payload['subject'] = access_token_subject_from_payload(payload)
        if isinstance(issuer, str) and issuer != 'docformify':
            subject = cast(AccessTokenSubject, payload['subject'])
            if payload.get('token_use') != 'access' or (
                not isinstance(subject.get('tenant_id'), str)
                or not isinstance(subject.get('deployment_id'), str)
                or not isinstance(subject.get('config_revision'), int)
            ):
                raise jwt.InvalidTokenError('Missing deployment binding')
        return payload
    except Exception as e:  # noqa: BLE001 - deliberate broad catch to close WS
        await _fail_ws(
            websocket,
            code=1008,
            reason='Invalid token',
            tag=tag,
            log_msg=f"Invalid JWT token: {e}",
            exit_reason='invalid_token',
        )


def _extract_identity(
    payload: dict[str, object],
) -> tuple[str, str, AccessTokenSubject]:
    """
    Return the canonical username, JTI, and subject from an access token.

    Args:
        payload: The decoded JWT payload.

    Returns:
        A tuple of (username, jti, subject_data).
    """
    subject_data = access_token_subject_from_payload(payload)
    return subject_data['username'], subject_data['jti'], subject_data


def _build_autoreg_cache(
    user_data: dict[str, object] | None,
    username: str,
    jti: str,
    payload: dict[str, object],
    subject_data: AccessTokenSubject,
) -> dict[str, object]:
    """
    Build a user cache dictionary that adds a missing JTI to the active list.

    Args:
        user_data: Existing user cache data, if any.
        username: The username to use if ``user_data`` is ``None``.
        jti: The JTI to add to the active list.
        payload: The full JWT payload.
        subject_data: The ``subject`` dictionary from the payload.

    Returns:
        A new user cache dictionary with the JTI added to the active list.
    """
    # Start from the canonical cache shape when the user has no prior session.
    cache: dict[str, object] = dict(user_data) if user_data is not None else {
        'db_user': {
            'id': cast(
                object,
                subject_data['user_id'],
            ),
            'username': username,
            'role': cast(
                object,
                subject_data['role'],
            ),
            'group_id': None,
            'status': 'active',
        },
        'jti_list': [],
        'jti_meta': {},
        'refresh_tokens': [],
        'refresh_token_hashes': [],
        'refresh_token_families': {},
        'feature_names': [],
    }
    jtis = list(cast(list[str], cache['jti_list']))
    if jti not in jtis:
        jtis.append(jti)
    cache['jti_list'] = jtis

    jti_meta = cast(dict[str, int], cache['jti_meta'])
    jti_meta[jti] = cast(int, payload['exp'])
    return cache


def get_model_key_from_ws(websocket: WebSocketLike) -> str | None:
    """
    Extract the model key for YOLO WebSocket endpoints.

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
    settings: SettingsLike,
    auto_register_jti: bool = AUTO_REGISTER_JTI,
    client_tag: str | None = None,
) -> tuple[str, str, dict[str, object]]:
    """
    Authenticate a WebSocket client using a JWT.

    Args:
        websocket: The WebSocket connection.
        rds: Redis-like connection used by the user cache helpers.
        settings: Object providing ``authjwt_secret_key`` and ``ALGORITHM``.
        auto_register_jti: When ``True`` (default), add an unknown JTI to the
            user's active list in the cache.
        client_tag: Optional label used in log messages for easier tracing.

    Returns:
        A tuple ``(username, jti, payload)``.

    Raises:
        SystemExit: If the request is unauthenticated or the token is invalid.

    Notes:
        - On error, the WebSocket is closed with code ``1008`` (Policy
          Violation) and a descriptive reason. This function does not return
          in such cases.
        - The cache layout is maintained by helper functions in
          ``examples.auth``; this function merely orchestrates the flow.
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

    # Verify JWT signature and structure
    payload = await _decode_or_fail(token_str, settings, websocket, tag)
    if not payload:
        await _fail_ws(
            websocket,
            code=1008,
            reason='Empty token payload',
            tag=tag,
            log_msg='Empty JWT payload',
            exit_reason='empty_payload',
        )

    # Read user identity from payload
    username_str, jti_str, subject_data = _extract_identity(payload)

    # Prune and validate JTI in cache for the user
    await prune_user_cache(rds, username_str)
    user_data: dict[str, object] | None = await rate_limiter_service.get_user_data(
        rds, username_str,
    )

    # A WebSocket server may sit behind a dedicated WSS upstream, but the
    # deployment-bound token must still agree with the cached account tenant.
    # This prevents a valid token for one tenant from reusing another tenant's
    # account cache entry.
    cached_user = (
        cast(dict[str, object], user_data.get('db_user'))
        if user_data and isinstance(user_data.get('db_user'), dict)
        else None
    )
    token_tenant_id = subject_data.get('tenant_id')
    cached_tenant_id = cached_user.get('tenant_id') if cached_user else None
    if (
        isinstance(token_tenant_id, str)
        and isinstance(cached_tenant_id, str)
        and token_tenant_id != cached_tenant_id
    ):
        await _fail_ws(
            websocket,
            code=1008,
            reason='Deployment configuration changed',
            tag=tag,
            log_msg='Token tenant does not match cached account tenant',
            exit_reason='deployment_configuration_changed',
        )

    # Validate JTI against cache list
    jti_is_active = (
        user_data is not None
        and jti_str in cast(list[str], user_data['jti_list'])
    )
    if not jti_is_active:
        if auto_register_jti:
            logger.info(
                'WebSocket auto-registering missing JTI tag=%s username=%s',
                tag,
                username_str,
            )
            new_cache = _build_autoreg_cache(
                user_data,
                username=username_str,
                jti=jti_str,
                payload=payload,
                subject_data=subject_data,
            )
            await rate_limiter_service.set_user_data(rds, username_str, new_cache)
        else:
            await _fail_ws(
                websocket,
                code=1008,
                reason='Token not active',
                tag=tag,
                log_msg=(
                    f"JTI not found in user active tokens for {username_str}"
                ),
                exit_reason='jti_not_active',
            )

    # Successful authentication (callers may log connection context)
    return username_str, jti_str, payload
