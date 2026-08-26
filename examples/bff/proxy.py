from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import httpx
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import Response
from fastapi.responses import StreamingResponse
from redis.asyncio import Redis
from redis.exceptions import RedisError

from examples.auth.deployment_context import DeploymentBinding
from examples.auth.session_store import acquire_refresh_lock
from examples.auth.session_store import auth_tokens
from examples.auth.session_store import delete_auth_session
from examples.auth.session_store import get_auth_session
from examples.auth.session_store import release_refresh_lock
from examples.auth.session_store import save_auth_tokens
from examples.db_management.schemas.auth import RefreshRequest
from examples.db_management.services.auth_services import refresh_tokens
from src.http_client_pool import HttpClientPool

# Uvicorn configures this logger at INFO level in the deployed tmux services.
logger = logging.getLogger('uvicorn.error')

_CHAT_UPSTREAM = os.getenv(
    'BFF_CHAT_API_URL',
    'http://127.0.0.1:8001',
)
_FILES_UPSTREAM = os.getenv(
    'BFF_FILES_API_URL',
    'http://127.0.0.1:8004',
)
_DETECTION_UPSTREAM = os.getenv(
    'BFF_DETECTION_API_URL',
    'http://127.0.0.1:8000',
)
_MANAGEMENT_UPSTREAM = os.getenv(
    'BFF_MANAGEMENT_API_URL',
    'http://127.0.0.1:8005',
)
_FCM_UPSTREAM = os.getenv(
    'BFF_FCM_API_URL',
    'http://127.0.0.1:8003',
)
_VIOLATIONS_UPSTREAM = os.getenv(
    'BFF_VIOLATIONS_API_URL',
    'http://127.0.0.1:8002',
)
_STREAMING_UPSTREAM = os.getenv(
    'BFF_STREAMING_API_URL',
    'http://127.0.0.1:8800',
)

BFF_UPSTREAMS = {
    'chat': _CHAT_UPSTREAM,
    'db_management': _MANAGEMENT_UPSTREAM,
    'detection': _DETECTION_UPSTREAM,
    'fcm': _FCM_UPSTREAM,
    'files': _FILES_UPSTREAM,
    'streaming': _STREAMING_UPSTREAM,
    # The deployed web client addresses the streaming API by this public name.
    'streaming_web': _STREAMING_UPSTREAM,
    'violations': _VIOLATIONS_UPSTREAM,
}
REFRESH_BEFORE_SECONDS = int(os.getenv('BFF_REFRESH_BEFORE_SECONDS', '30'))
UPSTREAM_TIMEOUT_SECONDS = float(
    os.getenv('BFF_UPSTREAM_TIMEOUT_SECONDS', '20'),
)

_DROP_REQUEST_HEADERS = {
    'authorization',
    'connection',
    'content-length',
    'cookie',
    'host',
    'proxy-authorization',
    'te',
    'trailer',
    'transfer-encoding',
    'upgrade',
    'x-csrf-token',
    'x-forwarded-host',
    'x-forwarded-proto',
}
_PASS_RESPONSE_HEADERS = {
    'accept-ranges',
    'content-disposition',
    'content-language',
    'content-range',
    'content-type',
    'etag',
    'last-modified',
    'retry-after',
}
_VIOLATION_MEDIA_FIELDS = frozenset({'image_url', 'thumbnail_url'})
_VIOLATION_MEDIA_ENDPOINTS = frozenset(
    {'/get_violation_image', '/get_violation_thumbnail'},
)


def _is_sse_request(request: Request, suffix: str) -> bool:
    """Return whether the proxied request should remain streaming.

    Args:
        request: Browser request whose accepted media types are inspected.
        suffix: Path after the allow-listed upstream service name.

    Returns:
        ``True`` when the upstream response must not be buffered.
    """
    accept = request.headers.get('accept', '').lower()
    return 'text/event-stream' in accept or suffix.startswith(
        'metadata/stream-id/',
    )


def _bff_service_path(
    request: Request,
    service: str,
) -> str:
    """Return the public BFF prefix for one proxied service.

    The request path retains any deployment-specific API prefix, while the
    upstream service only knows its own root routes. Deriving the prefix from
    the original browser request therefore works for both ``/bff`` and a BFF
    mounted below a public API path.
    """
    marker = f'/{service}'
    request_path = request.url.path.rstrip('/')
    offset = request_path.find(marker)
    if offset >= 0:
        marker_end = offset + len(marker)
        if marker_end == len(request_path) or request_path[
            marker_end
        ] == '/':
            return request_path[:marker_end]
    return f'/bff/{service}'


def _bff_violation_media_url(
    value: object,
    request: Request,
    deployment: DeploymentBinding | None,
) -> str | None:
    """Map one internal violation-media URL to its BFF-protected URL.

    Violation services generate URLs for their own root endpoints, but browser
    sessions are authenticated only by the BFF cookie. Only recognised media
    paths are rewritten so an arbitrary URL in an upstream JSON response can
    never be redirected through the BFF.
    """
    if not isinstance(value, str):
        return None
    source = urlsplit(value)
    if source.path not in _VIOLATION_MEDIA_ENDPOINTS:
        return None

    path = f'{_bff_service_path(request, "violations")}{source.path}'
    if deployment is None:
        return urlunsplit(('', '', path, source.query, ''))
    public_url = urlsplit(deployment.api_base_url)
    return urlunsplit(
        (public_url.scheme, public_url.netloc, path, source.query, ''),
    )


def _rewrite_violation_media_urls(
    payload: object,
    request: Request,
    deployment: DeploymentBinding | None,
) -> bool:
    """Rewrite recognised evidence URLs in a violation JSON response.

    Returns:
        ``True`` when at least one URL was changed.
    """
    changed = False
    if isinstance(payload, list):
        for item in payload:
            changed |= _rewrite_violation_media_urls(
                item,
                request,
                deployment,
            )
        return changed
    if not isinstance(payload, dict):
        return False

    for key, value in payload.items():
        if key in _VIOLATION_MEDIA_FIELDS:
            rewritten = _bff_violation_media_url(
                value,
                request,
                deployment,
            )
            if rewritten is not None and rewritten != value:
                payload[key] = rewritten
                changed = True
            continue
        changed |= _rewrite_violation_media_urls(
            value,
            request,
            deployment,
        )
    return changed


def _rewrite_violation_response_content(
    content: bytes,
    content_type: str | None,
    request: Request,
    deployment: DeploymentBinding | None,
) -> tuple[bytes, bool]:
    """Return violation JSON content with browser-reachable media URLs.

    Non-JSON response bodies, including the protected JPEG endpoints
    themselves, must pass through without parsing or re-encoding.
    """
    if not content_type or content_type.split(';', 1)[0].lower() != (
        'application/json'
    ):
        return content, False
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return content, False
    if not _rewrite_violation_media_urls(payload, request, deployment):
        return content, False
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(',', ':'),
        ).encode('utf-8'),
        True,
    )


def _log_sse_error_events(
    buffer: bytearray,
    chunk: bytes,
    *,
    request_path: str,
    upstream_status: int,
) -> None:
    """Log structured error events without changing streamed SSE bytes.

    Args:
        buffer: Partial SSE event bytes retained across chunks.
        chunk: Newly received SSE bytes.
        request_path: Original BFF request path for observability.
        upstream_status: Status returned by the upstream service.
    """
    buffer.extend(chunk)
    while b'\n\n' in buffer:
        raw_event, _, remaining = buffer.partition(b'\n\n')
        buffer[:] = remaining
        event_type = 'message'
        data_lines: list[str] = []
        for line in (
            bytes(raw_event).decode('utf-8', errors='replace').splitlines()
        ):
            field, separator, value = line.partition(':')
            if not separator:
                continue
            if field == 'event':
                event_type = value.lstrip()
            elif field == 'data':
                data_lines.append(value.lstrip())
        if event_type not in {'error', 'redis_error'}:
            continue
        try:
            payload = json.loads('\n'.join(data_lines))
        except json.JSONDecodeError:
            payload = {}
        if payload.get('source') == 'redis':
            logger.warning(
                'BFF SSE Redis error path=%s upstream_status=%s code=%s',
                request_path,
                upstream_status,
                payload.get('code', 'unknown'),
            )
        else:
            logger.warning(
                'BFF SSE upstream error event path=%s upstream_status=%s',
                request_path,
                upstream_status,
            )


def resolve_upstream(path: str) -> tuple[str, str]:
    """Resolve an allow-listed service name without accepting a caller URL.

    Args:
        path: Service-prefixed BFF path.

    Returns:
        Upstream base URL and remaining path suffix.

    Raises:
        HTTPException: If the service name is not allow-listed.
    """
    route, separator, suffix = path.strip('/').partition('/')
    base = BFF_UPSTREAMS.get(route)
    if not base:
        raise HTTPException(status_code=404, detail='bff_route_not_allowed')
    return base.rstrip('/'), suffix if separator else ''


def _is_terminal_refresh_error(exc: HTTPException) -> bool:
    """Return whether a refresh error permanently invalidates a BFF session.

    Args:
        exc: Refresh-token HTTP exception.

    Returns:
        ``True`` for a permanent authentication failure.
    """
    if exc.status_code == 409:
        return True
    if exc.status_code != 401:
        return False
    detail = str(exc.detail).lower()
    return any(
        marker in detail
        for marker in ('expired', 'revoked', 'reused', 'invalid', 'recognised')
    )


async def get_proxy_access_token(
    redis: Redis,
    session_id: str,
    *,
    force_refresh: bool = False,
    deployment: DeploymentBinding | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return a valid BFF access token, refreshing it when necessary.

    Args:
        redis: Redis connection holding BFF session state.
        session_id: Opaque browser session identifier.
        force_refresh: Whether to rotate despite a still-valid access token.

    Returns:
        Access token and current server-side session record.

    Raises:
        HTTPException: If the session is absent, refresh is rejected, or a peer
            refresh does not complete in time.
    """
    session = await get_auth_session(redis, session_id)
    if session is None:
        raise HTTPException(status_code=401, detail='app_session_expired')

    access_token, refresh_token = auth_tokens(session)
    remaining = int(session.get('access_expires_at') or 0) - int(time.time())
    if not force_refresh and remaining > REFRESH_BEFORE_SECONDS:
        return access_token, session

    lock_owner = await acquire_refresh_lock(redis, session_id)
    if lock_owner is None:
        return await _wait_for_proxy_session_refresh(
            redis,
            session_id,
            access_token,
            session,
            remaining,
        )

    try:
        return await _refresh_proxy_session(
            redis,
            session_id,
            access_token,
            refresh_token,
            force_refresh=force_refresh,
            deployment=deployment,
        )
    finally:
        await release_refresh_lock(redis, session_id, lock_owner)


async def _wait_for_proxy_session_refresh(
    redis: Redis,
    session_id: str,
    access_token: str,
    session: dict[str, Any],
    remaining: int,
) -> tuple[str, dict[str, Any]]:
    """Reuse a peer's token refresh before treating the session as busy.

    Args:
        redis: Redis connection holding BFF session state.
        session_id: Opaque browser session identifier.
        access_token: Token observed before waiting for the lock owner.
        session: Last known server-side session record.
        remaining: Seconds remaining on the last known access token.

    Returns:
        Refreshed token and session, or the still-valid original pair.

    Raises:
        HTTPException: If the session vanishes or no valid token becomes
            available before the wait period ends.
    """
    for _ in range(40):
        await asyncio.sleep(0.05)
        latest = await get_auth_session(redis, session_id)
        if latest is None:
            raise HTTPException(status_code=401, detail='app_session_expired')
        latest_access, _ = auth_tokens(latest)
        if latest_access != access_token:
            return latest_access, latest
    if remaining > 0:
        return access_token, session
    raise HTTPException(status_code=503, detail='session_refresh_busy')


async def _refresh_proxy_session(
    redis: Redis,
    session_id: str,
    access_token: str,
    refresh_token: str,
    *,
    force_refresh: bool,
    deployment: DeploymentBinding | None,
) -> tuple[str, dict[str, Any]]:
    """Refresh a lock-owning BFF session and persist its rotated tokens.

    Args:
        redis: Redis connection holding BFF session and refresh state.
        session_id: Opaque browser session identifier.
        access_token: Token observed before acquiring the refresh lock.
        refresh_token: Current refresh token for the session.
        force_refresh: Whether to rotate even if another worker changed tokens.

    Returns:
        Rotated access token and updated server-side session record.

    Raises:
        HTTPException: If session or refresh-token validation fails.
    """
    latest = await get_auth_session(redis, session_id)
    if latest is None:
        raise HTTPException(status_code=401, detail='app_session_expired')
    latest_access, latest_refresh = auth_tokens(latest)
    if latest_access != access_token and not force_refresh:
        return latest_access, latest
    try:
        result = await refresh_tokens(
            RefreshRequest(refresh_token=latest_refresh or refresh_token),
            redis,
            hash_refresh_token=True,
            deployment=deployment,
        )
    except HTTPException as exc:
        if _is_terminal_refresh_error(exc):
            await delete_auth_session(redis, session_id)
            raise HTTPException(
                status_code=401,
                detail='app_session_expired',
            ) from exc
        raise
    new_access = str(result['access_token'])
    new_refresh = str(result['refresh_token'])
    await save_auth_tokens(
        redis,
        session_id,
        latest,
        new_access,
        new_refresh,
        list(result.get('feature_names') or []),
    )
    return new_access, latest


async def _get_proxy_access_token_or_503(
    redis: Redis,
    session_id: str,
    request_path: str,
    *,
    force_refresh: bool = False,
    deployment: DeploymentBinding | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return a BFF access token or translate Redis failure to HTTP 503.

    Args:
        redis: Redis connection holding BFF session state.
        session_id: Opaque browser session identifier.
        request_path: Original BFF request path for observability.
        force_refresh: Whether to rotate despite a valid access token.

    Returns:
        Valid access token and current server-side session record.

    Raises:
        HTTPException: If Redis is unavailable or token refresh fails.
    """
    try:
        return await get_proxy_access_token(
            redis,
            session_id,
            force_refresh=force_refresh,
            deployment=deployment,
        )
    except RedisError as exc:
        logger.warning(
            'BFF Redis connection failed path=%s error_type=%s',
            request_path,
            type(exc).__name__,
            exc_info=True,
        )
        raise HTTPException(
            status_code=503,
            detail='bff_redis_unavailable',
        ) from exc


def _proxy_request_headers(
    request: Request,
    access_token: str,
    deployment: DeploymentBinding | None = None,
) -> dict[str, str]:
    """Build upstream headers without forwarding browser credentials.

    Args:
        request: Original browser request.
        access_token: Server-side access token for the upstream request.

    Returns:
        Sanitised upstream headers with BFF authentication markers.
    """
    headers = {
        name: value
        for name, value in request.headers.items()
        if name.lower() not in _DROP_REQUEST_HEADERS
    }
    headers['Authorization'] = f"Bearer {access_token}"
    headers['X-BFF-Request'] = '1'
    if deployment is not None:
        # The internal services resolve their deployment from the request's
        # scheme and Host.  ``url`` points to a loopback upstream, so retain
        # the verified public deployment authority instead of forwarding an
        # untrusted browser-supplied Host/origin.
        public_url = urlsplit(deployment.api_base_url)
        headers['Host'] = public_url.netloc
        headers['X-Forwarded-Host'] = public_url.netloc
        headers['X-Forwarded-Proto'] = public_url.scheme
    return headers


async def proxy_request(
    request: Request,
    redis: Redis,
    session_id: str,
    path: str,
    *,
    deployment: DeploymentBinding | None = None,
) -> Response:
    """Forward an authenticated request to an allow-listed upstream service.

    Args:
        request: Original browser request.
        redis: Redis connection holding BFF token state.
        session_id: Opaque browser session identifier.
        path: Service-prefixed allow-listed upstream path.

    Returns:
        Non-cacheable upstream HTTP or streaming response.

    Raises:
        HTTPException: If authentication, routing, or upstream communication
            fails.
    """
    base, suffix = resolve_upstream(path)
    url = f"{base}/{suffix}" if suffix else base

    access_token, _ = await _get_proxy_access_token_or_503(
        redis,
        session_id,
        request.url.path,
        deployment=deployment,
    )
    if request.method == 'GET' and _is_sse_request(request, suffix):
        return await _proxy_streaming_request(
            request,
            redis,
            session_id,
            url,
            access_token,
            deployment,
        )

    upstream = await _send_proxy_request(
        request,
        url,
        access_token,
        deployment,
    )
    if upstream.status_code == 401:
        access_token, _ = await _get_proxy_access_token_or_503(
            redis,
            session_id,
            request.url.path,
            force_refresh=True,
            deployment=deployment,
        )
        upstream = await _send_proxy_request(
            request,
            url,
            access_token,
            deployment,
        )

    response_content = upstream.content
    content_rewritten = False
    if path.strip('/').partition('/')[0] == 'violations':
        response_content, content_rewritten = (
            _rewrite_violation_response_content(
                response_content,
                upstream.headers.get('content-type'),
                request,
                deployment,
            )
        )

    return Response(
        content=response_content,
        status_code=upstream.status_code,
        headers=_proxy_response_headers(
            upstream,
            content_rewritten=content_rewritten,
        ),
    )


async def _send_proxy_request(
    request: Request,
    url: str,
    access_token: str,
    deployment: DeploymentBinding | None = None,
) -> httpx.Response:
    """Send one non-streaming request to an allow-listed upstream service.

    Args:
        request: Original browser request.
        url: Fully resolved allow-listed upstream URL.
        access_token: Server-side access token for the upstream request.

    Returns:
        Complete upstream HTTP response.

    Raises:
        HTTPException: If the upstream connection cannot be established.
    """
    client, close_client = await _proxy_http_client(request)
    try:
        return await client.request(
            request.method,
            url,
            params=request.query_params,
            content=await request.body(),
            headers=_proxy_request_headers(
                request,
                access_token,
                deployment,
            ),
        )
    except (httpx.TimeoutException, httpx.NetworkError) as exc:
        raise HTTPException(
            status_code=502,
            detail='bff_upstream_unavailable',
        ) from exc
    finally:
        if close_client:
            await client.aclose()


async def _proxy_http_client(
    request: Request,
) -> tuple[httpx.AsyncClient, bool]:
    """Return the application pool client or a disposable test fallback."""
    app = getattr(request, 'app', None)
    state = getattr(app, 'state', None)
    pool = getattr(state, 'http_clients', None)
    if isinstance(pool, HttpClientPool):
        return (
            await pool.get(
                'bff-upstream',
                timeout=UPSTREAM_TIMEOUT_SECONDS,
                follow_redirects=False,
            ),
            False,
        )
    return (
        httpx.AsyncClient(
            timeout=UPSTREAM_TIMEOUT_SECONDS,
            follow_redirects=False,
        ),
        True,
    )


def _proxy_response_headers(
    upstream: httpx.Response,
    *,
    streaming: bool = False,
    content_rewritten: bool = False,
) -> dict[str, str]:
    """Select safe upstream response headers for a browser response.

    Args:
        upstream: Response received from the allow-listed service.
        streaming: Whether the response is an SSE stream.

    Returns:
        Safe response headers with BFF no-store policy applied.
    """
    headers = {
        name: value
        for name, value in upstream.headers.items()
        if name.lower() in _PASS_RESPONSE_HEADERS
    }
    headers['Cache-Control'] = 'no-store'
    if content_rewritten:
        # The BFF serialised a different JSON body, so the upstream validator
        # would no longer describe the bytes being returned to the browser.
        headers.pop('etag', None)
    if streaming:
        headers['X-Accel-Buffering'] = 'no'
    return headers


async def _proxy_streaming_request(
    request: Request,
    redis: Redis,
    session_id: str,
    url: str,
    access_token: str,
    deployment: DeploymentBinding | None = None,
) -> StreamingResponse:
    """Proxy one long-lived upstream stream without buffering it.

    Args:
        request: Original browser streaming request.
        redis: Redis connection holding BFF token state.
        session_id: Opaque browser session identifier.
        url: Fully resolved allow-listed upstream URL.
        access_token: Current server-side access token.

    Returns:
        Non-buffering response that forwards upstream stream bytes.
    """
    client, upstream, close_client = await _open_proxy_stream(
        request,
        url,
        access_token,
        deployment,
    )
    if upstream.status_code == 401:
        (
            client,
            upstream,
            close_client,
        ) = await _refresh_unauthorized_proxy_stream(
            request,
            redis,
            session_id,
            url,
            client,
            upstream,
            close_client,
            deployment,
        )
    _log_proxy_stream_open(request.url.path, upstream.status_code)

    headers = _proxy_response_headers(upstream, streaming=True)

    return StreamingResponse(
        _proxy_stream_body(
            request.url.path,
            client,
            upstream,
            close_client=close_client,
        ),
        status_code=upstream.status_code,
        headers=headers,
        media_type=headers.get('content-type'),
    )


async def _open_proxy_stream(
    request: Request,
    url: str,
    access_token: str,
    deployment: DeploymentBinding | None = None,
) -> tuple[httpx.AsyncClient, httpx.Response, bool]:
    """Open an upstream SSE response without buffering its body.

    Args:
        request: Original browser streaming request.
        url: Fully resolved allow-listed upstream URL.
        access_token: Server-side access token for the upstream request.

    Returns:
        Open HTTP client, streaming upstream response, and client ownership.

    Raises:
        HTTPException: If the upstream streaming connection cannot be opened.
    """
    client, close_client = await _proxy_sse_http_client(request)
    try:
        upstream_request = client.build_request(
            request.method,
            url,
            params=request.query_params,
            headers=_proxy_request_headers(
                request,
                access_token,
                deployment,
            ),
        )
        response = await client.send(upstream_request, stream=True)
    except (httpx.TimeoutException, httpx.NetworkError) as exc:
        if close_client:
            await client.aclose()
        logger.warning(
            'BFF SSE upstream connection failed path=%s error_type=%s',
            request.url.path,
            type(exc).__name__,
            exc_info=True,
        )
        raise HTTPException(
            status_code=502,
            detail='bff_upstream_unavailable',
        ) from exc
    return client, response, close_client


async def _proxy_sse_http_client(
    request: Request,
) -> tuple[httpx.AsyncClient, bool]:
    """Return the lifespan SSE client or a disposable non-app fallback."""
    app = getattr(request, 'app', None)
    state = getattr(app, 'state', None)
    pool = getattr(state, 'http_clients', None)
    timeout = httpx.Timeout(UPSTREAM_TIMEOUT_SECONDS, read=None)
    if isinstance(pool, HttpClientPool):
        return (
            await pool.get(
                'bff-upstream-sse',
                timeout=timeout,
                follow_redirects=False,
            ),
            False,
        )
    return (
        httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
        ),
        True,
    )


async def _refresh_unauthorized_proxy_stream(
    request: Request,
    redis: Redis,
    session_id: str,
    url: str,
    client: httpx.AsyncClient,
    upstream: httpx.Response,
    close_client: bool,
    deployment: DeploymentBinding | None = None,
) -> tuple[httpx.AsyncClient, httpx.Response, bool]:
    """Refresh a BFF session once after an unauthorised SSE response.

    Args:
        request: Original browser streaming request.
        redis: Redis connection holding BFF token state.
        session_id: Opaque browser session identifier.
        url: Fully resolved allow-listed upstream URL.
        client: Client used by the rejected upstream response.
        upstream: Rejected upstream response to close before retrying.
        close_client: Whether the rejected client is disposable.

    Returns:
        Replacement client and upstream streaming response.
    """
    logger.info(
        'BFF SSE upstream returned 401; refreshing session path=%s',
        request.url.path,
    )
    await upstream.aclose()
    if close_client:
        await client.aclose()
    access_token, _ = await _get_proxy_access_token_or_503(
        redis,
        session_id,
        request.url.path,
        force_refresh=True,
        deployment=deployment,
    )
    return await _open_proxy_stream(
        request,
        url,
        access_token,
        deployment,
    )


def _log_proxy_stream_open(request_path: str, upstream_status: int) -> None:
    """Log the status returned while opening an SSE upstream.

    Args:
        request_path: Original BFF request path.
        upstream_status: Status returned by the upstream service.
    """
    if upstream_status >= 400:
        logger.warning(
            'BFF SSE upstream response path=%s upstream_status=%s',
            request_path,
            upstream_status,
        )
        return
    logger.info(
        'BFF SSE upstream opened path=%s upstream_status=%s',
        request_path,
        upstream_status,
    )


async def _proxy_stream_body(
    request_path: str,
    client: httpx.AsyncClient,
    upstream: httpx.Response,
    *,
    close_client: bool,
) -> AsyncIterator[bytes]:
    """Yield SSE bytes and close its response and any disposable client.

    Args:
        request_path: Original BFF request path for observability.
        client: Open HTTP client used for the upstream stream.
        upstream: Open upstream response whose bytes are yielded.
        close_client: Whether this request, rather than app lifespan, owns the
            HTTP client.

    Yields:
        Unmodified upstream SSE byte chunks.
    """
    error_buffer = bytearray()
    bytes_sent = 0
    close_reason = 'upstream_completed'
    try:
        async for chunk in upstream.aiter_bytes():
            bytes_sent += len(chunk)
            _log_sse_error_events(
                error_buffer,
                chunk,
                request_path=request_path,
                upstream_status=upstream.status_code,
            )
            yield chunk
    except asyncio.CancelledError:
        close_reason = 'downstream_disconnected'
        raise
    except httpx.TransportError as exc:
        close_reason = type(exc).__name__
    except Exception as exc:
        close_reason = type(exc).__name__
        logger.exception(
            'BFF SSE proxy failed path=%s upstream_status=%s reason=%s',
            request_path,
            upstream.status_code,
            close_reason,
        )
        raise
    finally:
        logger.info(
            'BFF SSE upstream closed path=%s upstream_status=%s reason=%s '
            'bytes_sent=%s',
            request_path,
            upstream.status_code,
            close_reason,
            bytes_sent,
        )
        await upstream.aclose()
        if close_client:
            await client.aclose()
