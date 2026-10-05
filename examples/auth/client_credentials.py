"""Cache a short-lived provider token with one in-flight refresh per client."""
from __future__ import annotations

import asyncio
import hashlib
import math
import time
from collections.abc import Callable

import httpx


class ClientCredentialsTokenCache:
    """Keep tokens private to an HTTP client's lifetime and configuration."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = asyncio.Lock()
        self._token: str | None = None
        self._context: tuple[str, str, str] | None = None
        self._valid_until = 0.0

    def invalidate(self, rejected_token: str) -> None:
        """Discard only the rejected token, preserving a concurrent refresh."""
        if self._token == rejected_token:
            self._token = None
            self._valid_until = 0.0

    async def get(
        self,
        client: httpx.AsyncClient,
        *,
        token_url: str,
        client_id: str,
        client_secret: str,
    ) -> str:
        """Reuse a valid token or acquire one using the supplied provider."""
        context = (
            token_url, client_id,
            hashlib.sha256(client_secret.encode()).hexdigest(),
        )
        async with self._lock:
            if (
                context == self._context
                and self._token is not None
                and self._clock() < self._valid_until
            ):
                return self._token
            started_at = self._clock()
            response = await client.post(
                token_url,
                data={
                    'grant_type': 'client_credentials',
                    'client_id': client_id,
                    'client_secret': client_secret,
                },
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError('Invalid client-credentials response')
            token = payload.get('access_token')
            if not isinstance(token, str) or not token:
                raise ValueError('Missing client-credentials access token')
            lifetime = float(payload.get('expires_in', 0))
            if not math.isfinite(lifetime) or lifetime < 0:
                raise ValueError('Invalid client-credentials token lifetime')
            # Missing lifetime disables reuse; short-lived tokens keep a
            # proportional margin, while longer tokens refresh 30s early.
            self._valid_until = started_at + max(
                0.0, lifetime - min(30.0, lifetime * 0.1),
            )
            self._context = context
            self._token = token
            return token
