from __future__ import annotations

import unittest

import httpx

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
