from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import StreamingResponse
from redis.asyncio import Redis

from examples.auth.deployment_context import DeploymentBinding
from examples.bff.proxy_config import logger
from examples.bff.proxy_http import _proxy_request_headers
from examples.bff.proxy_http import _proxy_response_headers
from examples.bff.proxy_http import _proxy_sse_http_client
from examples.bff.proxy_tokens import _get_proxy_access_token_or_503
from src.async_tasks import finish_on_cancel


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


class SseDiagnostics:
    """Parse bounded diagnostic events without changing forwarded SSE bytes."""

    max_event_bytes = 64 * 1024

    def __init__(self, request_path: str, upstream_status: int):
        self.request_path = request_path
        self.upstream_status = upstream_status
        self.event = bytearray()
        self.line = bytearray()
        self.line_nonempty = False
        self.after_cr = False
        self.discarding = False

    def feed(self, chunk: bytes) -> None:
        for value in chunk:
            if self.after_cr:
                self.after_cr = False
                if value == 10:
                    continue
            if value in (10, 13):
                self.after_cr = value == 13
                if not self.line_nonempty:
                    if not self.discarding:
                        self._log_event()
                    self.event.clear()
                    self.discarding = False
                elif not self.discarding:
                    self.event.extend(self.line)
                    self.event.append(10)
                self.line.clear()
                self.line_nonempty = False
            else:
                self.line_nonempty = True
                if not self.discarding:
                    self.line.append(value)
            if len(self.event) + len(self.line) >= self.max_event_bytes:
                self.event.clear()
                self.line.clear()
                self.discarding = True

    def _log_event(self) -> None:
        event_type = 'message'
        data_lines = []
        for line in self.event.decode('utf-8', errors='replace').splitlines():
            key, separator, value = line.partition(':')
            if separator:
                if key == 'event':
                    event_type = value.lstrip()
                elif key == 'data':
                    data_lines.append(value.lstrip())
        if event_type not in {'error', 'redis_error'}:
            return
        try:
            payload = json.loads('\n'.join(data_lines))
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        if payload.get('source') == 'redis':
            logger.warning(
                'BFF SSE Redis error path=%s upstream_status=%s code=%s',
                self.request_path,
                self.upstream_status,
                payload.get('code', 'unknown'),
            )
        else:
            logger.warning(
                'BFF SSE upstream error event path=%s upstream_status=%s',
                self.request_path,
                self.upstream_status,
            )


class ProxyStream:
    """Own an upstream response and any request-owned HTTP client."""

    def __init__(self, client: httpx.AsyncClient, owns_client: bool):
        self.client = client
        self.owns_client = owns_client
        self.response: httpx.Response | None = None
        self._close_task: asyncio.Task | None = None

    async def close(self):

        if self._close_task is None:
            self._close_task = asyncio.create_task(self._dispose())
        await finish_on_cancel(self._close_task)

    async def _dispose(self):
        try:
            if self.response is not None:
                await self.response.aclose()
        finally:
            if self.owns_client:
                await self.client.aclose()


class ProxyStreamingResponse(StreamingResponse):
    def __init__(self, stream: ProxyStream, request_path: str):
        assert stream.response is not None
        self.stream = stream
        super().__init__(
            _proxy_stream_body(request_path, stream),
            status_code=stream.response.status_code,
            headers=_proxy_response_headers(stream.response, streaming=True),
        )

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Headers may fail before the body iterator ever starts.
            await self.stream.close()


async def _proxy_streaming_request(
    request: Request,
    redis: Redis,
    session_id: str,
    url: str,
    access_token: str,
    deployment: DeploymentBinding | None = None,
) -> StreamingResponse:
    stream = await _open_proxy_stream(request, url, access_token, deployment)
    try:
        assert stream.response is not None
        if stream.response.status_code == 401:
            await stream.close()
            access_token, _ = await _get_proxy_access_token_or_503(
                redis,
                session_id,
                request.url.path,
                force_refresh=True,
                deployment=deployment,
            )
            stream = await _open_proxy_stream(
                request,
                url,
                access_token,
                deployment,
            )
        assert stream.response is not None
        _log_proxy_stream_open(request.url.path, stream.response.status_code)
        return ProxyStreamingResponse(stream, request.url.path)
    except BaseException:
        await stream.close()
        raise


async def _open_proxy_stream(
    request: Request,
    url: str,
    access_token: str,
    deployment: DeploymentBinding | None = None,
) -> ProxyStream:
    client, owns_client = await _proxy_sse_http_client(request)
    stream = ProxyStream(client, owns_client)
    try:
        upstream_request = client.build_request(
            request.method,
            url,
            params=request.query_params,
            headers=_proxy_request_headers(
                request,
                access_token,
                deployment,
                upstream_url=url,
            ),
        )
        stream.response = await client.send(upstream_request, stream=True)
        return stream
    except (httpx.TimeoutException, httpx.NetworkError) as error:
        await stream.close()
        logger.warning(
            'BFF SSE upstream connection failed path=%s error_type=%s',
            request.url.path,
            type(error).__name__,
        )
        raise HTTPException(502, 'bff_upstream_unavailable') from error
    except BaseException:
        await stream.close()
        raise


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
    stream: ProxyStream,
) -> AsyncIterator[bytes]:
    upstream = stream.response
    assert upstream is not None
    diagnostics = SseDiagnostics(request_path, upstream.status_code)
    bytes_sent = 0
    close_reason = 'upstream_completed'
    try:
        async for chunk in upstream.aiter_bytes():
            bytes_sent += len(chunk)
            diagnostics.feed(chunk)
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
            'BFF SSE upstream closed path=%s upstream_status=%s '
            'reason=%s bytes_sent=%s',
            request_path,
            upstream.status_code,
            close_reason,
            bytes_sent,
        )
        await stream.close()
