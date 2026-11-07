from __future__ import annotations

from urllib.parse import urlsplit

import httpx
from fastapi import HTTPException
from fastapi import Request

from examples.auth.deployment_context import DeploymentBinding
from examples.bff.proxy_config import _DROP_REQUEST_HEADERS
from examples.bff.proxy_config import _PASS_RESPONSE_HEADERS
from examples.bff.proxy_config import BFF_UPSTREAMS
from examples.bff.proxy_config import settings as proxy_settings
from examples.bff.proxy_config import UPSTREAM_TIMEOUT_SECONDS
from src.http_client_pool import HttpClientPool


def _proxy_request_headers(
    request: Request,
    access_token: str,
    deployment: DeploymentBinding | None = None,
    *,
    upstream_url: str | None = None,
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
    ai_base = BFF_UPSTREAMS['ai_access'].rstrip('/')
    if upstream_url and (
            upstream_url == ai_base or upstream_url.startswith(
            ai_base + '/',
            )
    ):
        # Only trusted ASGI proxy resolution supplies this address. Never
        # forward browser-supplied provenance assertions or X-Real-IP.
        proof = proxy_settings.company_ai_network_proxy_key.get_secret_value()
        if len(proof) < 32:
            raise HTTPException(
                503, detail='company_ai_network_proxy_not_configured',
            )
        headers['X-Company-AI-Network-Proxy'] = proof
        headers['X-Company-AI-Client-IP'] = (
            request.client.host if request.client else ''
        )
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
                upstream_url=url,
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
