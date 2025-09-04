from __future__ import annotations

import os
from collections.abc import Awaitable
from collections.abc import Iterable
from collections.abc import Mapping
from typing import cast
from typing import NoReturn
from typing import Protocol

import jwt

from examples.auth.cache import get_user_data
from examples.auth.cache import set_user_data
from examples.auth.token_cleanup import prune_user_cache


class WebSocketLike(Protocol):
    """
    A minimal protocol describing the WebSocket operations we use.
    """

    headers: Mapping[str, str]
    # FastAPI provides a mapping-like object; we only convert it to ``dict``.
    query_params: object

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


def _to_str_dict(obj: object) -> dict[str, str]:
    """
    Coerce a mapping-like or iterable of pairs into a string-keyed dictionary.

    Args:
        obj: A mapping-like object or an iterable of ``(key, value)`` pairs.

    Returns:
        A new dictionary with stringified keys and values. Returns an empty
        dictionary when coercion is not possible.
    """
    if isinstance(obj, Mapping):
        m = cast(Mapping[object, object], obj)
        return {str(k): str(v) for k, v in m.items()}
    if isinstance(obj, Iterable):
        try:
            it = cast(Iterable[tuple[object, object]], obj)
            return {str(k): str(v) for k, v in it}
        except Exception:
            return {}
    return {}


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
    try:
        query_params = _to_str_dict(websocket.query_params)
        tok = query_params.get('token')
        if tok:
            return tok
    except Exception:
        pass
    return None


async def _fail_ws(
    websocket: WebSocketLike,
    *,
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
    print(f"{tag}: {log_msg}")
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
        return cast(
            dict[str, object],
            jwt.decode(
                token,
                settings.authjwt_secret_key,
                algorithms=[settings.ALGORITHM],
            ),
        )
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
) -> tuple[str | None, str | None, dict[str, object]]:
    """
    Return (username, jti, subject_data) from payload if present.

    Args:
        payload: The decoded JWT payload.

    Returns:
        A tuple of (username, jti, subject_data). ``username`` and ``j
    """
    subject_obj = payload.get('subject')
    subject_data: dict[str, object]
    if isinstance(subject_obj, dict):
        subject_data = cast(dict[str, object], subject_obj)
    else:
        subject_data = {}
    username = cast(
        str | None,
        subject_data.get('username'),
    ) or cast(str | None, payload.get('username'))
    jti = cast(
        str | None,
        subject_data.get('jti'),
    ) or cast(str | None, payload.get('jti'))
    return username, jti, subject_data


def _build_autoreg_cache(
    user_data: dict[str, object] | None,
    *,
    username: str,
    jti: str,
    payload: dict[str, object],
    subject_data: dict[str, object],
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
    # Start from existing cache or construct a minimal skeleton
    cache: dict[str, object] = user_data or {
        'db_user': {
            'id': cast(
                object,
                subject_data.get('user_id') or payload.get('user_id'),
            ),
            'username': username,
            'role': cast(
                object,
                subject_data.get('role') or payload.get('role'),
            ),
            'group_id': None,
            'is_active': True,
        },
        'jti_list': [],
        'refresh_tokens': [],
    }
    # Ensure jti_list is a list
    jtis_obj = cache.get('jti_list')
    if isinstance(jtis_obj, list):
        jtis = list(cast(list[str], jtis_obj))
    else:
        jtis = []
    if jti not in jtis:
        jtis.append(jti)
    cache['jti_list'] = jtis

    # Record exp if present and parseable
    try:
        exp_val = payload.get('exp')
        exp_ts: int | None = None
        if isinstance(exp_val, (int, float, str)):
            exp_ts = int(exp_val)
        if exp_ts is not None:
            jti_meta_obj = cache.get('jti_meta')
            if isinstance(jti_meta_obj, dict):
                jti_meta = cast(dict[str, int], jti_meta_obj)
            else:
                jti_meta = {}
            jti_meta[jti] = exp_ts
            cache['jti_meta'] = jti_meta
    except Exception:
        # Ignore malformed exp
        pass
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
    try:
        query_params = _to_str_dict(websocket.query_params)
        return query_params.get('model')
    except Exception:
        return None


async def authenticate_websocket(
    websocket: WebSocketLike,
    rds: object,
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
    username, jti, subject_data = _extract_identity(payload)
    if username is None or jti is None:
        await _fail_ws(
            websocket,
            code=1008,
            reason='Invalid token data',
            tag=tag,
            log_msg='Missing username or JTI in token',
            exit_reason='missing_user_or_jti',
        )
    username_str: str = username
    jti_str: str = jti

    # Prune and validate JTI in cache for the user
    await prune_user_cache(rds, username_str)
    user_data: dict[str, object] | None = await get_user_data(
        rds, username_str,
    )

    # Validate JTI against cache list
    jti_is_active = False
    if user_data is not None:
        jl_obj = user_data.get('jti_list')
        if isinstance(jl_obj, list):
            jti_is_active = jti_str in jl_obj
    if not jti_is_active:
        if auto_register_jti:
            print(
                f"{tag}: Auto-registering missing JTI for {username_str}",
            )
            new_cache = _build_autoreg_cache(
                user_data,
                username=username_str,
                jti=jti_str,
                payload=payload,
                subject_data=subject_data,
            )
            await set_user_data(rds, username_str, new_cache)
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

    # Successful authentication
    print(f"{tag}: Authenticated as {username_str}")
    return username_str, jti_str, payload
