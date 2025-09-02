from __future__ import annotations

import os
from collections.abc import Awaitable
from collections.abc import Iterable
from collections.abc import Mapping
from typing import cast
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
    if not token:
        print(
            f"{tag}: No token found in header or query parameter",
        )
        await websocket.close(
            code=1008,
            reason='Missing authentication token',
        )
        raise SystemExit('missing_token')

    # Verify JWT signature and structure
    try:
        payload: dict[str, object] = jwt.decode(
            token,
            settings.authjwt_secret_key,
            algorithms=[settings.ALGORITHM],
        )
    except Exception as e:
        print(
            f"{tag}: Invalid JWT token: {e}",
        )
        await websocket.close(
            code=1008,
            reason='Invalid token',
        )
        raise SystemExit('invalid_token')

    if not payload:
        print(f"{tag}: Empty JWT payload")
        await websocket.close(
            code=1008,
            reason='Empty token payload',
        )
        raise SystemExit('empty_payload')

    # Read user identity from payload, preferring a nested ``subject`` field
    # whilst remaining backwards compatible with flat payloads.
    subject_obj = payload.get('subject')
    subject_data: dict[str, object]
    if isinstance(subject_obj, dict):
        subject_data = cast(dict[str, object], subject_obj)
    else:
        subject_data = {}
    username: str | None = cast(
        str | None,
        subject_data.get('username'),
    ) or cast(str | None, payload.get('username'))
    jti: str | None = cast(
        str | None,
        subject_data.get('jti'),
    ) or cast(str | None, payload.get('jti'))

    if not username or not jti:
        print(f"{tag}: Missing username or JTI in token")
        await websocket.close(
            code=1008,
            reason='Invalid token data',
        )
        raise SystemExit('missing_user_or_jti')

    # Prune and validate JTI in cache for the user
    await prune_user_cache(rds, username)
    user_data: dict[str, object] | None = await get_user_data(rds, username)

    # Validate JTI against cache list
    jti_is_active = False
    if user_data is not None:
        jl_obj = user_data.get('jti_list')
        if isinstance(jl_obj, list):
            jti_is_active = jti in jl_obj
    if not jti_is_active:
        if auto_register_jti:
            print(f"{tag}: Auto-registering missing JTI for {username}")
            new_cache: dict[str, object] = user_data or {
                'db_user': {
                    'id': cast(
                        object,
                        subject_data.get('user_id')
                        or payload.get('user_id'),
                    ),
                    'username': username,
                    'role': cast(
                        object,
                        subject_data.get('role')
                        or payload.get('role'),
                    ),
                    'group_id': None,
                    'is_active': True,
                },
                'jti_list': [],
                'refresh_tokens': [],
            }
            jtis_obj = new_cache.get('jti_list')
            jtis: list[str]
            if isinstance(jtis_obj, list):
                jtis = list(cast(list[str], jtis_obj))
            else:
                jtis = []
            if jti not in jtis:
                jtis.append(jti)
            new_cache['jti_list'] = jtis
            # record exp if present
            try:
                # Narrow the type and only cast supported types for mypy safety
                exp_val = payload.get('exp')
                exp_ts = None
                if isinstance(exp_val, (int, float, str)):
                    exp_ts = int(exp_val)
                if exp_ts is not None:
                    jti_meta_obj = new_cache.get('jti_meta')
                    if isinstance(jti_meta_obj, dict):
                        jti_meta = cast(dict[str, int], jti_meta_obj)
                    else:
                        jti_meta = {}
                    jti_meta[jti] = exp_ts
                    new_cache['jti_meta'] = jti_meta
            except Exception:
                pass
            await set_user_data(rds, username, new_cache)
        else:
            print(
                f"{tag}: JTI not found in user active tokens for {username}",
            )
            await websocket.close(
                code=1008,
                reason='Token not active',
            )
            raise SystemExit('jti_not_active')

    # Successful authentication
    print(f"{tag}: Authenticated as {username}")
    return username, jti, payload
