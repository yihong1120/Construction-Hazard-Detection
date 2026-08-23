from __future__ import annotations

import asyncio

import httpx


class AsyncHttpClientOwner:
    """Provide one lazily-created, safely closed HTTP client per instance."""

    def __init__(self, timeout: int | float) -> None:
        """Perform init.

        Args:
            timeout: Value used by this callable.
        """
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        """Return the live pooled client, creating it only when necessary."""
        async with self._client_lock:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(self.timeout),
                    limits=httpx.Limits(
                        max_keepalive_connections=5,
                        max_connections=10,
                        keepalive_expiry=30,
                    ),
                    http2=True,
                )
            return self._client

    async def close(self) -> None:
        """Close and discard the pooled client, if it is still open."""
        async with self._client_lock:
            if self._client is not None and not self._client.is_closed:
                await self._client.aclose()
            self._client = None
