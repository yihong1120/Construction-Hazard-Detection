from __future__ import annotations

import asyncio
import unittest
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch
from uuid import UUID

import httpx
import pytest
from fastapi import HTTPException
from redis.exceptions import ConnectionError as RedisConnectionError

from examples.auth.deployment_context import DeploymentBinding
from examples.bff import proxy
from examples.bff.proxy import _get_proxy_access_token_or_503
from examples.bff.proxy import _is_sse_request
from examples.bff.proxy import _log_sse_error_events
from examples.bff.proxy import resolve_upstream


class BffServicesTest(unittest.TestCase):
    """Provide BffServicesTest."""

    def test_db_management_service_is_allowlisted(self) -> None:
        """The Web client can reach db_management routes through the BFF."""
        base, suffix = resolve_upstream('db_management/list_sites')

        self.assertTrue(base)
        self.assertEqual(suffix, 'list_sites')

    def test_legacy_management_service_is_not_allowlisted(self) -> None:
        """Only the canonical db_management public name is accepted."""
        with self.assertRaises(HTTPException) as raised:
            resolve_upstream('management/list_sites')

        self.assertEqual(raised.exception.status_code, 404)
        self.assertEqual(raised.exception.detail, 'bff_route_not_allowed')

    def test_streaming_web_labels_route_is_allowlisted(self) -> None:
        """The deployed stream page can reach its labels endpoint."""
        base, suffix = resolve_upstream('streaming_web/labels')

        self.assertTrue(base)
        self.assertEqual(suffix, 'labels')

    def test_fcm_service_is_allowlisted(self) -> None:
        """Test fcm service is allowlisted."""
        base, suffix = resolve_upstream('fcm/notifications/unread_count')

        self.assertTrue(base)
        self.assertEqual(suffix, 'notifications/unread_count')

    def test_metadata_path_uses_sse_streaming_proxy(self) -> None:
        """Test metadata path uses sse streaming proxy."""
        request = type(
            'Request',
            (),
            {
                'headers': {'accept': '*/*'},
            },
        )()

        self.assertTrue(
            _is_sse_request(
                request,  # type: ignore[arg-type]
                'metadata/stream-id/site/cam',
            ),
        )

    def test_accept_event_stream_uses_sse_streaming_proxy(self) -> None:
        """Test accept event stream uses sse streaming proxy."""
        request = type(
            'Request',
            (),
            {
                'headers': {'accept': 'text/event-stream'},
            },
        )()

        self.assertTrue(
            _is_sse_request(
                request,  # type: ignore[arg-type]
                'streams/site',
            ),
        )

    def test_redis_sse_error_is_logged_with_upstream_status(self) -> None:
        """Test redis sse error is logged with upstream status."""
        buffer = bytearray()

        with self.assertLogs('uvicorn.error', level='WARNING') as logs:
            _log_sse_error_events(
                buffer,
                b'event: redis_error\ndata: {"source":"redis",',
                request_path='/bff/streaming_web/metadata/stream-id/site/cam',
                upstream_status=200,
            )
            self.assertEqual(
                buffer,
                b'event: redis_error\ndata: {"source":"redis",',
            )
            _log_sse_error_events(
                buffer,
                b'"code":"redis_unavailable"}\n\n',
                request_path='/bff/streaming_web/metadata/stream-id/site/cam',
                upstream_status=200,
            )

        self.assertIn('BFF SSE Redis error', logs.output[0])
        self.assertIn('upstream_status=200', logs.output[0])
        self.assertIn('code=redis_unavailable', logs.output[0])

    def test_bff_redis_connection_failure_returns_503(self) -> None:
        """Test bff redis connection failure returns 503."""

        async def get_token() -> HTTPException:
            """Perform get token.

            Returns:
                The callable result.
            """
            with (
                patch(
                    'examples.bff.proxy.get_proxy_access_token',
                    new=AsyncMock(side_effect=RedisConnectionError('down')),
                ),
                self.assertLogs('uvicorn.error', level='WARNING') as logs,
            ):
                with self.assertRaises(HTTPException) as raised:
                    await _get_proxy_access_token_or_503(
                        AsyncMock(),
                        'session-id',
                        '/bff/streaming_web/metadata/stream-id/site/cam',
                    )
            self.assertIn('BFF Redis connection failed', logs.output[0])
            return raised.exception

        error = asyncio.run(get_token())

        self.assertEqual(error.status_code, 503)
        self.assertEqual(error.detail, 'bff_redis_unavailable')


if __name__ == '__main__':
    unittest.main()


class FakeResponse:
    """Minimal async httpx response used by proxy unit tests."""

    def __init__(
        self,
        status_code: int = 200,
        *,
        content: bytes = b'',
        headers: dict[str, str] | None = None,
        chunks: tuple[bytes, ...] = (),
        stream_error: BaseException | None = None,
    ) -> None:
        """Perform init.

        Args:
            status_code: Value used by this callable.
            content: Value used by this callable.
            headers: Value used by this callable.
            chunks: Value used by this callable.
            stream_error: Value used by this callable.
        """
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}
        self.chunks = chunks
        self.stream_error = stream_error
        self.closed = False

    async def aclose(self) -> None:
        """Perform aclose."""
        self.closed = True

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        """Perform aiter bytes.

        Returns:
            The callable result.
        """
        for chunk in self.chunks:
            yield chunk
        if self.stream_error is not None:
            raise self.stream_error


class FakeAsyncClient:
    """Configurable stand-in for the two httpx client APIs used by the BFF."""

    def __init__(
        self,
        response: FakeResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        """Perform init.

        Args:
            response: Value used by this callable.
            error: Value used by this callable.
        """
        self.response = response or FakeResponse()
        self.error = error
        self.request_calls: list[dict[str, Any]] = []
        self.closed = False

    async def __aenter__(self) -> FakeAsyncClient:
        """Perform aenter.

        Returns:
            The callable result.
        """
        return self

    async def __aexit__(self, *_args: object) -> None:
        """Perform aexit.

        Args:
            *_args: Value used by this callable.
        """
        await self.aclose()

    async def aclose(self) -> None:
        """Perform aclose."""
        self.closed = True

    async def request(self, *args: object, **kwargs: object) -> FakeResponse:
        """Perform request.

        Args:
            *args: Value used by this callable.
            **kwargs: Value used by this callable.

        Returns:
            The callable result.
        """
        self.request_calls.append({'args': args, 'kwargs': kwargs})
        if self.error is not None:
            raise self.error
        return self.response

    def build_request(self, *args: object, **kwargs: object) -> object:
        """Perform build request.

        Args:
            *args: Value used by this callable.
            **kwargs: Value used by this callable.

        Returns:
            The callable result.
        """
        self.request_calls.append({'args': args, 'kwargs': kwargs})
        return object()

    async def send(self, _request: object, *, stream: bool) -> FakeResponse:
        """Perform send.

        Args:
            _request: Value used by this callable.
            stream: Value used by this callable.

        Returns:
            The callable result.
        """
        assert stream is True
        if self.error is not None:
            raise self.error
        return self.response


def _request(
    *,
    method: str = 'GET',
    accept: str = 'application/json',
) -> SimpleNamespace:
    """Create the subset of FastAPI Request consumed by the proxy."""
    return SimpleNamespace(
        method=method,
        headers={
            'accept': accept,
            'authorization': 'discard',
            'x-custom': 'kept',
            'cookie': 'discard',
        },
        query_params={'page': '2'},
        url=SimpleNamespace(path='/bff/streaming_web/streams/site'),
        body=AsyncMock(return_value=b'body'),
    )


def _session(token: str = 'old', expires_at: int = 10_000) -> dict[str, Any]:
    """Create a BFF session record with deterministic token data."""
    return {
        'access_token': token,
        'refresh_token': 'refresh-token',
        'access_expires_at': expires_at,
    }


def _run(awaitable: Any) -> Any:
    """Run one async helper from normal pytest functions."""
    return asyncio.run(awaitable)


def test_sse_log_handles_ignored_malformed_and_upstream_events() -> None:
    """SSE logging ignores ordinary events and logs both error formats."""
    buffer = bytearray()
    proxy._log_sse_error_events(
        buffer,
        b'not-an-sse-line\n\nevent: update\ndata: {}\n\n',
        request_path='/bff/streaming_web/streams/site',
        upstream_status=200,
    )
    assert buffer == b''

    with pytest.MonkeyPatch.context() as monkeypatch:
        logger = MagicMock()
        monkeypatch.setattr(proxy, 'logger', logger)
        proxy._log_sse_error_events(
            buffer,
            b'event: error\ndata: not-json\n\n',
            request_path='/bff/streaming_web/streams/site',
            upstream_status=502,
        )
        logger.warning.assert_called_once()


@pytest.mark.parametrize(
    ('status', 'detail', 'expected'),
    [
        (500, 'expired', False),
        (401, 'temporary backend error', False),
        (401, 'refresh token revoked', True),
    ],
)
def test_terminal_refresh_error_detection(
    status: int,
    detail: str,
    expected: bool,
) -> None:
    """Only permanent authentication failures invalidate a BFF session."""
    assert (
        proxy._is_terminal_refresh_error(
            HTTPException(status_code=status, detail=detail),
        )
        is expected
    )


def test_proxy_request_headers_drop_credentials_and_keep_safe_values() -> None:
    """The BFF replaces browser credentials with its server-side bearer
    token."""
    headers = proxy._proxy_request_headers(_request(), 'server-token')

    assert headers == {
        'accept': 'application/json',
        'x-custom': 'kept',
        'Authorization': 'Bearer server-token',
        'X-BFF-Request': '1',
    }


def test_proxy_request_headers_use_server_verified_deployment_authority() -> (
    None
):
    """A loopback upstream must receive the canonical public authority."""
    deployment = DeploymentBinding(
        tenant_id=UUID('00000000-0000-0000-0000-000000000001'),
        deployment_id=UUID('00000000-0000-0000-0000-000000000002'),
        api_base_url='https://api.example.com/hazard/api',
        config_revision=1,
    )

    headers = proxy._proxy_request_headers(
        _request(),
        'server-token',
        deployment,
    )

    assert headers['Host'] == 'api.example.com'
    assert headers['X-Forwarded-Host'] == 'api.example.com'
    assert headers['X-Forwarded-Proto'] == 'https'


def test_resolve_upstream_rejects_unknown_service() -> None:
    """Callers cannot use the BFF as an arbitrary outbound proxy."""
    with pytest.raises(HTTPException, match='bff_route_not_allowed') as raised:
        proxy.resolve_upstream('unknown-service/private')
    assert raised.value.status_code == 404


def test_get_proxy_access_token_handles_session_and_lock_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing sessions and a completed peer refresh return stable HTTP
    errors."""
    monkeypatch.setattr(
        proxy,
        'get_auth_session',
        AsyncMock(return_value=None),
    )
    with pytest.raises(HTTPException, match='app_session_expired') as missing:
        _run(proxy.get_proxy_access_token(AsyncMock(), 'session'))
    assert missing.value.status_code == 401

    fresh = _session(expires_at=100)
    acquire = AsyncMock()
    monkeypatch.setattr(proxy, 'time', SimpleNamespace(time=lambda: 10))
    monkeypatch.setattr(
        proxy,
        'get_auth_session',
        AsyncMock(return_value=fresh),
    )
    monkeypatch.setattr(
        proxy,
        'auth_tokens',
        lambda session: (
            session['access_token'],
            session['refresh_token'],
        ),
    )
    monkeypatch.setattr(proxy, 'acquire_refresh_lock', acquire)
    assert _run(proxy.get_proxy_access_token(AsyncMock(), 'session')) == (
        'old',
        fresh,
    )
    acquire.assert_not_awaited()

    old = _session(expires_at=1)
    newer = _session(token='new', expires_at=1)
    monkeypatch.setattr(proxy, 'time', SimpleNamespace(time=lambda: 10))
    monkeypatch.setattr(
        proxy,
        'get_auth_session',
        AsyncMock(side_effect=[old, newer]),
    )
    monkeypatch.setattr(
        proxy,
        'auth_tokens',
        lambda session: (
            session['access_token'],
            session['refresh_token'],
        ),
    )
    monkeypatch.setattr(
        proxy,
        'acquire_refresh_lock',
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(proxy.asyncio, 'sleep', AsyncMock())

    token, session = _run(proxy.get_proxy_access_token(AsyncMock(), 'session'))

    assert token == 'new'
    assert session is newer


def test_get_proxy_access_token_waits_or_reports_busy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lock contention falls back to a valid token or returns a retry
    response."""
    valid = _session(expires_at=20)
    monkeypatch.setattr(proxy, 'time', SimpleNamespace(time=lambda: 10))
    monkeypatch.setattr(
        proxy,
        'auth_tokens',
        lambda session: (
            session['access_token'],
            session['refresh_token'],
        ),
    )
    monkeypatch.setattr(
        proxy,
        'acquire_refresh_lock',
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(proxy.asyncio, 'sleep', AsyncMock())
    monkeypatch.setattr(
        proxy,
        'get_auth_session',
        AsyncMock(side_effect=[valid] + [valid] * 40),
    )

    assert _run(proxy.get_proxy_access_token(AsyncMock(), 'session')) == (
        'old',
        valid,
    )

    expired = _session(expires_at=0)
    monkeypatch.setattr(
        proxy,
        'get_auth_session',
        AsyncMock(side_effect=[expired] + [expired] * 40),
    )
    with pytest.raises(HTTPException, match='session_refresh_busy') as busy:
        _run(proxy.get_proxy_access_token(AsyncMock(), 'session'))
    assert busy.value.status_code == 503

    monkeypatch.setattr(
        proxy,
        'get_auth_session',
        AsyncMock(side_effect=[expired, None]),
    )
    with pytest.raises(HTTPException, match='app_session_expired') as vanished:
        _run(proxy.get_proxy_access_token(AsyncMock(), 'session'))
    assert vanished.value.status_code == 401


def test_get_proxy_access_token_refreshes_and_releases_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lock owner refreshes tokens, stores them, then always releases
    lock."""
    session = _session(expires_at=1)
    owner = 'lock-owner'
    save = AsyncMock()
    release = AsyncMock()
    monkeypatch.setattr(proxy, 'time', SimpleNamespace(time=lambda: 10))
    monkeypatch.setattr(
        proxy,
        'get_auth_session',
        AsyncMock(side_effect=[session, session]),
    )
    monkeypatch.setattr(
        proxy,
        'auth_tokens',
        lambda value: (
            value['access_token'],
            value['refresh_token'],
        ),
    )
    monkeypatch.setattr(
        proxy,
        'acquire_refresh_lock',
        AsyncMock(return_value=owner),
    )
    monkeypatch.setattr(proxy, 'release_refresh_lock', release)
    monkeypatch.setattr(proxy, 'save_auth_tokens', save)
    monkeypatch.setattr(
        proxy,
        'refresh_tokens',
        AsyncMock(
            return_value={
                'access_token': 'refreshed',
                'refresh_token': 'new-refresh',
                'feature_names': ['streaming'],
            },
        ),
    )
    redis = AsyncMock()

    token, returned_session = _run(
        proxy.get_proxy_access_token(redis, 'session'),
    )

    assert token == 'refreshed'
    assert returned_session is session
    save.assert_awaited_once()
    release.assert_awaited_once_with(redis, 'session', owner)


def test_get_proxy_access_token_invalidates_terminal_refresh_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A revoked refresh token deletes the app session and returns 401."""
    session = _session(expires_at=1)
    delete = AsyncMock()
    release = AsyncMock()
    monkeypatch.setattr(proxy, 'time', SimpleNamespace(time=lambda: 10))
    monkeypatch.setattr(
        proxy,
        'get_auth_session',
        AsyncMock(side_effect=[session, session]),
    )
    monkeypatch.setattr(
        proxy,
        'auth_tokens',
        lambda value: (
            value['access_token'],
            value['refresh_token'],
        ),
    )
    monkeypatch.setattr(
        proxy,
        'acquire_refresh_lock',
        AsyncMock(return_value='owner'),
    )
    monkeypatch.setattr(proxy, 'delete_auth_session', delete)
    monkeypatch.setattr(proxy, 'release_refresh_lock', release)
    monkeypatch.setattr(
        proxy,
        'refresh_tokens',
        AsyncMock(side_effect=HTTPException(401, 'refresh token expired')),
    )

    with pytest.raises(HTTPException, match='app_session_expired') as raised:
        _run(proxy.get_proxy_access_token(AsyncMock(), 'session'))

    assert raised.value.status_code == 401
    delete.assert_awaited_once()
    release.assert_awaited_once()


def test_get_proxy_access_token_handles_changed_or_missing_locked_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lock owners reuse a newer token and release locks on missing
    sessions."""
    original = _session(expires_at=1)
    changed = _session(token='newer', expires_at=1)
    release = AsyncMock()
    monkeypatch.setattr(proxy, 'time', SimpleNamespace(time=lambda: 10))
    monkeypatch.setattr(
        proxy,
        'auth_tokens',
        lambda value: (
            value['access_token'],
            value['refresh_token'],
        ),
    )
    monkeypatch.setattr(
        proxy,
        'acquire_refresh_lock',
        AsyncMock(return_value='owner'),
    )
    monkeypatch.setattr(proxy, 'release_refresh_lock', release)
    monkeypatch.setattr(
        proxy,
        'get_auth_session',
        AsyncMock(side_effect=[original, changed]),
    )

    assert _run(proxy.get_proxy_access_token(AsyncMock(), 'session')) == (
        'newer',
        changed,
    )
    release.assert_awaited_once()

    release.reset_mock()
    monkeypatch.setattr(
        proxy,
        'get_auth_session',
        AsyncMock(side_effect=[original, None]),
    )
    with pytest.raises(HTTPException, match='app_session_expired') as missing:
        _run(proxy.get_proxy_access_token(AsyncMock(), 'session'))
    assert missing.value.status_code == 401
    release.assert_awaited_once()


def test_get_proxy_access_token_reraises_nonterminal_refresh_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient refresh failures are preserved for the caller to handle."""
    session = _session(expires_at=1)
    release = AsyncMock()
    monkeypatch.setattr(proxy, 'time', SimpleNamespace(time=lambda: 10))
    monkeypatch.setattr(
        proxy,
        'get_auth_session',
        AsyncMock(side_effect=[session, session]),
    )
    monkeypatch.setattr(
        proxy,
        'auth_tokens',
        lambda value: (
            value['access_token'],
            value['refresh_token'],
        ),
    )
    monkeypatch.setattr(
        proxy,
        'acquire_refresh_lock',
        AsyncMock(return_value='owner'),
    )
    monkeypatch.setattr(proxy, 'release_refresh_lock', release)
    monkeypatch.setattr(
        proxy,
        'refresh_tokens',
        AsyncMock(side_effect=HTTPException(503, 'upstream unavailable')),
    )

    with pytest.raises(HTTPException, match='upstream unavailable') as raised:
        _run(proxy.get_proxy_access_token(AsyncMock(), 'session'))

    assert raised.value.status_code == 503
    release.assert_awaited_once()


def test_proxy_request_forwards_response_and_retries_401(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normal requests forward selected headers and retry once after a 401."""
    first_client = FakeAsyncClient(FakeResponse(status_code=401))
    second_response = FakeResponse(
        content=b'ok',
        headers={'content-type': 'application/json', 'server': 'hidden'},
    )
    second_client = FakeAsyncClient(second_response)
    clients = iter([first_client, second_client])
    tokens = AsyncMock(side_effect=[('old', {}), ('new', {})])
    monkeypatch.setattr(
        proxy.httpx,
        'AsyncClient',
        lambda **_kwargs: next(clients),
    )
    monkeypatch.setattr(proxy, '_get_proxy_access_token_or_503', tokens)

    response = _run(
        proxy.proxy_request(
            _request(method='POST'),
            AsyncMock(),
            'session',
            'streaming/records',
        ),
    )

    assert response.status_code == 200
    assert response.body == b'ok'
    assert response.headers['cache-control'] == 'no-store'
    assert response.headers['content-type'] == 'application/json'
    assert tokens.await_count == 2
    assert second_client.request_calls[0]['kwargs']['headers'][
        'Authorization'
    ] == ('Bearer new')


def test_proxy_request_maps_network_error_to_bad_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-streaming upstream connection failure becomes a safe 502."""
    client = FakeAsyncClient(error=httpx.NetworkError('offline'))
    monkeypatch.setattr(proxy.httpx, 'AsyncClient', lambda **_kwargs: client)
    monkeypatch.setattr(
        proxy,
        '_get_proxy_access_token_or_503',
        AsyncMock(return_value=('token', {})),
    )

    with pytest.raises(
        HTTPException,
        match='bff_upstream_unavailable',
    ) as raised:
        _run(
            proxy.proxy_request(
                _request(),
                AsyncMock(),
                'session',
                'streaming/records',
            ),
        )

    assert raised.value.status_code == 502


def test_proxy_request_delegates_sse_routes_to_streaming_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SSE requests retain their streaming response instead of buffering
    data."""
    expected = MagicMock()
    stream = AsyncMock(return_value=expected)
    monkeypatch.setattr(
        proxy,
        '_get_proxy_access_token_or_503',
        AsyncMock(return_value=('token', {})),
    )
    monkeypatch.setattr(proxy, '_proxy_streaming_request', stream)

    response = _run(
        proxy.proxy_request(
            _request(accept='text/event-stream'),
            AsyncMock(),
            'session',
            'streaming/events',
        ),
    )

    assert response is expected
    stream.assert_awaited_once()


async def _collect_stream(response: Any) -> list[bytes]:
    """Consume a Starlette streaming response in a direct unit test."""
    return [chunk async for chunk in response.body_iterator]


def test_streaming_proxy_forwards_chunks_logs_errors_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful SSE proxy logs embedded errors and closes both
    resources."""
    upstream = FakeResponse(
        headers={'content-type': 'text/event-stream'},
        chunks=(b'event: error\ndata: {"source":"redis"}\n\n', b'next'),
    )
    client = FakeAsyncClient(upstream)
    logger = MagicMock()
    monkeypatch.setattr(proxy.httpx, 'AsyncClient', lambda **_kwargs: client)
    monkeypatch.setattr(proxy, 'logger', logger)

    response = _run(
        proxy._proxy_streaming_request(
            _request(accept='text/event-stream'),
            AsyncMock(),
            'session',
            'http://upstream/events',
            'token',
        ),
    )
    chunks = _run(_collect_stream(response))

    assert chunks == [b'event: error\ndata: {"source":"redis"}\n\n', b'next']
    assert upstream.closed is True
    assert client.closed is True
    assert response.headers['x-accel-buffering'] == 'no'
    assert logger.info.call_count >= 2


def test_streaming_proxy_reuses_lifespan_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SSE responses close independently while their app client stays
    pooled."""
    upstream = FakeResponse(chunks=(b'event: update\n\n',))
    client = FakeAsyncClient(upstream)
    pool = proxy.HttpClientPool()
    pool.get = AsyncMock(return_value=client)  # type: ignore[method-assign]
    request = _request(accept='text/event-stream')
    request.app = SimpleNamespace(
        state=SimpleNamespace(http_clients=pool),
    )

    response = _run(
        proxy._proxy_streaming_request(
            request,
            AsyncMock(),
            'session',
            'http://upstream/events',
            'token',
        ),
    )

    assert _run(_collect_stream(response)) == [b'event: update\n\n']
    assert upstream.closed is True
    assert client.closed is False
    pool.get.assert_awaited_once()


def test_streaming_proxy_retries_401_then_logs_upstream_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A streaming 401 renews the token before retaining a later 5xx stream."""
    first = FakeAsyncClient(FakeResponse(status_code=401))
    second_upstream = FakeResponse(status_code=503)
    second = FakeAsyncClient(second_upstream)
    clients = iter([first, second])
    logger = MagicMock()
    token = AsyncMock(return_value=('refreshed', {}))
    monkeypatch.setattr(
        proxy.httpx,
        'AsyncClient',
        lambda **_kwargs: next(clients),
    )
    monkeypatch.setattr(proxy, '_get_proxy_access_token_or_503', token)
    monkeypatch.setattr(proxy, 'logger', logger)

    response = _run(
        proxy._proxy_streaming_request(
            _request(accept='text/event-stream'),
            AsyncMock(),
            'session',
            'http://upstream/events',
            'old',
        ),
    )

    assert response.status_code == 503
    assert first.response.closed is True
    assert first.closed is True
    token.assert_awaited_once()
    logger.warning.assert_called_once()


def test_streaming_proxy_maps_open_and_iteration_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connection and later stream failures close resources and surface
    errors."""
    failed_client = FakeAsyncClient(error=httpx.NetworkError('offline'))
    monkeypatch.setattr(
        proxy.httpx,
        'AsyncClient',
        lambda **_kwargs: failed_client,
    )
    with pytest.raises(
        HTTPException,
        match='bff_upstream_unavailable',
    ) as raised:
        _run(
            proxy._proxy_streaming_request(
                _request(accept='text/event-stream'),
                AsyncMock(),
                'session',
                'http://upstream/events',
                'token',
            ),
        )
    assert raised.value.status_code == 502
    assert failed_client.closed is True

    upstream = FakeResponse(stream_error=RuntimeError('stream broke'))
    client = FakeAsyncClient(upstream)
    monkeypatch.setattr(proxy.httpx, 'AsyncClient', lambda **_kwargs: client)
    response = _run(
        proxy._proxy_streaming_request(
            _request(accept='text/event-stream'),
            AsyncMock(),
            'session',
            'http://upstream/events',
            'token',
        ),
    )
    with pytest.raises(RuntimeError, match='stream broke'):
        _run(_collect_stream(response))
    assert upstream.closed is True
    assert client.closed is True

    interrupted_upstream = FakeResponse(
        chunks=(b'retry: 15000\n\n',),
        stream_error=httpx.RemoteProtocolError('incomplete chunked read'),
    )
    interrupted_client = FakeAsyncClient(interrupted_upstream)
    monkeypatch.setattr(
        proxy.httpx,
        'AsyncClient',
        lambda **_kwargs: interrupted_client,
    )
    interrupted_response = _run(
        proxy._proxy_streaming_request(
            _request(accept='text/event-stream'),
            AsyncMock(),
            'session',
            'http://upstream/events',
            'token',
        ),
    )

    assert _run(_collect_stream(interrupted_response)) == [b'retry: 15000\n\n']
    assert interrupted_upstream.closed is True
    assert interrupted_client.closed is True

    cancelled_upstream = FakeResponse(stream_error=asyncio.CancelledError())
    cancelled_client = FakeAsyncClient(cancelled_upstream)
    monkeypatch.setattr(
        proxy.httpx,
        'AsyncClient',
        lambda **_kwargs: cancelled_client,
    )
    cancelled_response = _run(
        proxy._proxy_streaming_request(
            _request(accept='text/event-stream'),
            AsyncMock(),
            'session',
            'http://upstream/events',
            'token',
        ),
    )
    with pytest.raises(asyncio.CancelledError):
        _run(_collect_stream(cancelled_response))
    assert cancelled_upstream.closed is True
    assert cancelled_client.closed is True
