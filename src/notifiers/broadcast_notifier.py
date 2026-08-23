from __future__ import annotations

import logging

import httpx


class BroadcastNotifier:
    """Send text messages to an HTTP broadcast endpoint asynchronously."""

    def __init__(
        self,
        broadcast_url: str,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Perform init.

        Args:
            broadcast_url: Value used by this callable.
            client: Value used by this callable.
        """
        self.broadcast_url = broadcast_url
        self._client = client
        self._owns_client = client is None
        self.logger = logging.getLogger(__name__)

    async def broadcast_message(self, message: str) -> bool:
        """Post one message and return whether the endpoint accepted it."""
        client = self._http_client()
        try:
            response = await client.post(
                self.broadcast_url,
                json={'message': message},
            )
            if response.is_success:
                return True
            self.logger.warning(
                'Broadcast endpoint rejected request status=%s',
                response.status_code,
            )
            return False
        except httpx.HTTPError as exc:
            self.logger.warning(
                'Broadcast request failed error_type=%s',
                type(exc).__name__,
            )
            return False

    async def aclose(self) -> None:
        """Close a client created for standalone use."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _http_client(self) -> httpx.AsyncClient:
        """Return the injected transport or lazily create one for reuse."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, connect=5.0),
            )
        return self._client
