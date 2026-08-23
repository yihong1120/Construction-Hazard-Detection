from __future__ import annotations

import unittest

import httpx

import src.http_client_pool as client_pool
from src.http_client_pool import HttpClientPool


class HttpClientPoolTests(unittest.IsolatedAsyncioTestCase):
    """Verify application-lifetime HTTP clients are reused and released."""

    async def test_reuses_each_named_transport_profile(self) -> None:
        """Test reuses each named transport profile."""
        pool = HttpClientPool()
        first = await pool.get('upstream', timeout=5.0)
        second = await pool.get('upstream', timeout=5.0)

        self.assertIs(first, second)
        await pool.close()
        self.assertTrue(first.is_closed)

    async def test_separates_profiles_and_closes_all_clients(self) -> None:
        """Test separates profiles and closes all clients."""
        pool = HttpClientPool()
        upstream = await pool.get('upstream', timeout=5.0)
        streaming = await pool.get(
            'streaming',
            timeout=httpx.Timeout(5.0, read=None),
        )

        self.assertIsNot(upstream, streaming)
        await pool.close()
        self.assertTrue(upstream.is_closed)
        self.assertTrue(streaming.is_closed)

    async def test_application_client_is_optional_outside_lifespan(
        self,
    ) -> None:
        """Services receive no shared client before the application starts."""
        client_pool.set_application_http_clients(None)
        self.assertIsNone(
            await client_pool.get_application_http_client(
                'upstream',
                timeout=5.0,
            ),
        )

    async def test_application_client_uses_the_registered_pool(self) -> None:
        """Lifespan callers receive the registered named HTTP client."""
        pool = HttpClientPool()
        client_pool.set_application_http_clients(pool)
        try:
            client = await client_pool.get_application_http_client(
                'upstream',
                timeout=5.0,
                follow_redirects=True,
            )
        finally:
            client_pool.set_application_http_clients(None)
            await pool.close()

        self.assertIsNotNone(client)
