from __future__ import annotations

import asyncio

import httpx


_application_http_clients: HttpClientPool | None = None


class HttpClientPool:
    """Reuse outbound HTTP connections while keeping transport profiles separate."""

    def __init__(self) -> None:
        """Perform init.
        """
        self._clients: dict[str, httpx.AsyncClient] = {}
        self._lock = asyncio.Lock()

    async def get(
        self,
        name: str,
        *,
        timeout: httpx.Timeout | float,
        follow_redirects: bool = False,
    ) -> httpx.AsyncClient:
        """Return a live client for one named transport profile."""
        async with self._lock:
            client = self._clients.get(name)
            if client is None or client.is_closed:
                client = httpx.AsyncClient(
                    timeout=timeout,
                    follow_redirects=follow_redirects,
                    limits=httpx.Limits(
                        max_connections=40,
                        max_keepalive_connections=20,
                        keepalive_expiry=30,
                    ),
                    http2=True,
                )
                self._clients[name] = client
            return client

    async def close(self) -> None:
        """Close every pooled client during application shutdown."""
        async with self._lock:
            clients, self._clients = self._clients, {}
        await asyncio.gather(
            *(client.aclose() for client in clients.values()),
            return_exceptions=True,
        )


def set_application_http_clients(pool: HttpClientPool | None) -> None:
    """Register the lifespan-owned pool for service functions without a request."""
    global _application_http_clients
    _application_http_clients = pool


async def get_application_http_client(
    name: str,
    *,
    timeout: httpx.Timeout | float,
    follow_redirects: bool = False,
) -> httpx.AsyncClient | None:
    """Return the current app client, or ``None`` outside an app lifespan."""
    if _application_http_clients is None:
        return None
    return await _application_http_clients.get(
        name,
        timeout=timeout,
        follow_redirects=follow_redirects,
    )
