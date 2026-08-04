from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import Response
from fastapi.responses import StreamingResponse
from redis.asyncio import Redis
from redis.exceptions import RedisError

from examples.auth.session_store import acquire_refresh_lock
from examples.auth.session_store import auth_tokens
from examples.auth.session_store import delete_auth_session
from examples.auth.session_store import get_auth_session
from examples.auth.session_store import release_refresh_lock
from examples.auth.session_store import save_auth_tokens
from examples.db_management.schemas.auth import RefreshRequest
from examples.db_management.services.auth_services import refresh_tokens

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
    os.getenv('DB_MANAGEMENT_API_URL', 'http://127.0.0.1:8005'),
)
_FCM_UPSTREAM = os.getenv(
    'BFF_FCM_API_URL',
    os.getenv('FCM_API_URL', 'http://127.0.0.1:8003'),
)
_VIOLATIONS_UPSTREAM = os.getenv(
    'BFF_VIOLATIONS_API_URL',
    os.getenv('VIOLATION_RECORD_API_URL', 'http://127.0.0.1:8002'),
)
_STREAMING_UPSTREAM = os.getenv(
    'BFF_STREAMING_API_URL',
    os.getenv('STREAMING_API_URL', 'http://127.0.0.1:8800'),
)

# Keep canonical names and the service names embedded in the deployed
# Flutter Web build. The aliases always resolve to the same fixed upstream;
# callers can never provide an arbitrary URL.
BFF_UPSTREAMS = {
    'chat': _CHAT_UPSTREAM,
    'detect': _DETECTION_UPSTREAM,
    'detection': _DETECTION_UPSTREAM,
    'db_management': _MANAGEMENT_UPSTREAM,
    'management': _MANAGEMENT_UPSTREAM,
    'fcm': _FCM_UPSTREAM,
    'file_manage': _FILES_UPSTREAM,
    'files': _FILES_UPSTREAM,
    'streaming': _STREAMING_UPSTREAM,
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


def _is_sse_request(request: Request, suffix: str) -> bool:
    """Return whether the proxied request should stay streaming."""
    accept = request.headers.get('accept', '').lower()
    return 'text/event-stream' in accept or suffix.startswith(
        'metadata/stream-id/',
    )


def _log_sse_error_events(
    buffer: bytearray,
    chunk: bytes,
    *,
    request_path: str,
    upstream_status: int,
) -> None:
    """Log structured error events without changing streamed SSE bytes."""
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
    """Resolve only a compile-time route name; never accept a caller URL."""
    route, separator, suffix = path.strip('/').partition('/')
    base = BFF_UPSTREAMS.get(route)
    if not base:
        raise HTTPException(status_code=404, detail='bff_route_not_allowed')
    return base.rstrip('/'), suffix if separator else ''


def _is_terminal_refresh_error(exc: HTTPException) -> bool:
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
) -> tuple[str, dict[str, Any]]:
    session = await get_auth_session(redis, session_id)
    if session is None:
        raise HTTPException(status_code=401, detail='app_session_expired')

    access_token, refresh_token = auth_tokens(session)
    remaining = int(session.get('access_expires_at') or 0) - int(time.time())
    if not force_refresh and remaining > REFRESH_BEFORE_SECONDS:
        return access_token, session

    lock_owner = await acquire_refresh_lock(redis, session_id)
    if lock_owner is None:
        for _ in range(40):
            await asyncio.sleep(0.05)
            latest = await get_auth_session(redis, session_id)
            if latest is None:
                raise HTTPException(
                    status_code=401,
                    detail='app_session_expired',
                )
            latest_access, _ = auth_tokens(latest)
            if latest_access != access_token:
                return latest_access, latest
        if remaining > 0:
            return access_token, session
        raise HTTPException(status_code=503, detail='session_refresh_busy')

    try:
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
    finally:
        await release_refresh_lock(redis, session_id, lock_owner)


async def _get_proxy_access_token_or_503(
    redis: Redis,
    session_id: str,
    request_path: str,
    *,
    force_refresh: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Return the BFF session token or log a Redis availability failure."""
    try:
        return await get_proxy_access_token(
            redis,
            session_id,
            force_refresh=force_refresh,
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
) -> dict[str, str]:
    headers = {
        name: value
        for name, value in request.headers.items()
        if name.lower() not in _DROP_REQUEST_HEADERS
    }
    headers['Authorization'] = f"Bearer {access_token}"
    headers['X-BFF-Request'] = '1'
    return headers


async def proxy_request(
    request: Request,
    redis: Redis,
    session_id: str,
    path: str,
) -> Response:
    base, suffix = resolve_upstream(path)
    url = f"{base}/{suffix}" if suffix else base
    body = await request.body()

    async def send(access_token: str) -> httpx.Response:
        try:
            async with httpx.AsyncClient(
                timeout=UPSTREAM_TIMEOUT_SECONDS,
                follow_redirects=False,
            ) as client:
                return await client.request(
                    request.method,
                    url,
                    params=request.query_params,
                    content=body,
                    headers=_proxy_request_headers(request, access_token),
                )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise HTTPException(
                status_code=502,
                detail='bff_upstream_unavailable',
            ) from exc

    access_token, _ = await _get_proxy_access_token_or_503(
        redis,
        session_id,
        request.url.path,
    )
    if request.method == 'GET' and _is_sse_request(request, suffix):
        return await _proxy_streaming_request(
            request,
            redis,
            session_id,
            url,
            access_token,
        )

    upstream = await send(access_token)
    if upstream.status_code == 401:
        access_token, _ = await _get_proxy_access_token_or_503(
            redis,
            session_id,
            request.url.path,
            force_refresh=True,
        )
        upstream = await send(access_token)

    headers = {
        name: value
        for name, value in upstream.headers.items()
        if name.lower() in _PASS_RESPONSE_HEADERS
    }
    headers['Cache-Control'] = 'no-store'
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=headers,
    )


async def _proxy_streaming_request(
    request: Request,
    redis: Redis,
    session_id: str,
    url: str,
    access_token: str,
) -> StreamingResponse:
    """Proxy one long-lived upstream streaming response without buffering
    it."""

    async def open_stream(
        token: str,
    ) -> tuple[httpx.AsyncClient, httpx.Response]:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                UPSTREAM_TIMEOUT_SECONDS,
                read=None,
            ),
            follow_redirects=False,
        )
        try:
            upstream_request = client.build_request(
                request.method,
                url,
                params=request.query_params,
                headers=_proxy_request_headers(request, token),
            )
            response = await client.send(upstream_request, stream=True)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
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
        return client, response

    client, upstream = await open_stream(access_token)
    if upstream.status_code == 401:
        logger.info(
            'BFF SSE upstream returned 401; refreshing session path=%s',
            request.url.path,
        )
        await upstream.aclose()
        await client.aclose()
        access_token, _ = await _get_proxy_access_token_or_503(
            redis,
            session_id,
            request.url.path,
            force_refresh=True,
        )
        client, upstream = await open_stream(access_token)

    if upstream.status_code >= 400:
        logger.warning(
            'BFF SSE upstream response path=%s upstream_status=%s',
            request.url.path,
            upstream.status_code,
        )
    else:
        logger.info(
            'BFF SSE upstream opened path=%s upstream_status=%s',
            request.url.path,
            upstream.status_code,
        )

    headers = {
        name: value
        for name, value in upstream.headers.items()
        if name.lower() in _PASS_RESPONSE_HEADERS
    }
    headers['Cache-Control'] = 'no-store'
    headers['X-Accel-Buffering'] = 'no'

    async def body() -> AsyncIterator[bytes]:
        error_buffer = bytearray()
        bytes_sent = 0
        close_reason = 'upstream_completed'
        try:
            async for chunk in upstream.aiter_bytes():
                bytes_sent += len(chunk)
                _log_sse_error_events(
                    error_buffer,
                    chunk,
                    request_path=request.url.path,
                    upstream_status=upstream.status_code,
                )
                yield chunk
        except asyncio.CancelledError:
            close_reason = 'downstream_disconnected'
            raise
        except Exception as exc:
            close_reason = type(exc).__name__
            logger.warning(
                'BFF SSE upstream closed with error path=%s '
                'upstream_status=%s reason=%s',
                request.url.path,
                upstream.status_code,
                close_reason,
                exc_info=True,
            )
            raise
        finally:
            logger.info(
                'BFF SSE upstream closed path=%s upstream_status=%s reason=%s '
                'bytes_sent=%s',
                request.url.path,
                upstream.status_code,
                close_reason,
                bytes_sent,
            )
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        body(),
        status_code=upstream.status_code,
        headers=headers,
        media_type=headers.get('content-type'),
    )
