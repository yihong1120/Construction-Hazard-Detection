from __future__ import annotations

import asyncio
import unittest

import httpx

from examples.auth.client_credentials import ClientCredentialsTokenCache


class ClientCredentialsTokenTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.calls = 0
        self.now = 100.0
        self.cache = ClientCredentialsTokenCache(clock=lambda: self.now)

        async def handle(request):
            self.calls += 1
            await asyncio.sleep(0)
            return httpx.Response(
                200,
                json={
                    'access_token': f'token-{self.calls}',
                    'expires_in': 300,
                },
            )

        self.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        self.options = dict(
            token_url='https://id.example/token',
            client_id='service',
            client_secret='private',
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def test_concurrent_requests_share_one_token_acquisition(
            self,
    ) -> None:
        tokens = await asyncio.gather(
            *(self.cache.get(self.client, **self.options) for _ in range(10)),
        )
        self.assertEqual(tokens, ['token-1'] * 10)
        self.assertEqual(self.calls, 1)

    async def test_refresh_margin_and_credential_rotation(self) -> None:
        self.assertEqual(
            await self.cache.get(self.client, **self.options), 'token-1',
        )
        self.now = 369.0
        self.assertEqual(
            await self.cache.get(self.client, **self.options), 'token-1',
        )
        self.now = 370.0
        self.assertEqual(
            await self.cache.get(self.client, **self.options), 'token-2',
        )
        self.options['client_secret'] = 'rotated'
        self.assertEqual(
            await self.cache.get(self.client, **self.options), 'token-3',
        )

    async def test_stale_401_does_not_discard_newly_refreshed_token(
            self,
    ) -> None:
        old = await self.cache.get(self.client, **self.options)
        self.cache.invalidate(old)
        new = await self.cache.get(self.client, **self.options)
        self.cache.invalidate(old)
        self.assertEqual(
            await self.cache.get(self.client, **self.options), new,
        )
        self.assertEqual(self.calls, 2)

    async def test_provider_errors_and_invalid_payloads_are_not_cached(
            self,
    ) -> None:
        for status, payload in (
            (500, {}), (200, []), (
                200, {
                'access_token': '',
                },
            ), (
                200, {
                'access_token': 'bad', 'expires_in': 'nan',
                },
            ),
        ):
            with self.subTest(status=status, payload=payload):
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(
                        lambda request: httpx.Response(status, json=payload),
                    ),
                ) as client:
                    with self.assertRaises(
                        (httpx.HTTPStatusError, ValueError),
                    ):
                        await self.cache.get(client, **self.options)
        self.assertEqual(
            await self.cache.get(self.client, **self.options), 'token-1',
        )

    async def test_missing_lifetime_disables_reuse(self) -> None:
        requests = []

        def handle(request):
            requests.append(request)
            return httpx.Response(200, json={'access_token': 'uncached'})

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle),
        ) as client:
            await self.cache.get(client, **self.options)
            await self.cache.get(client, **self.options)
        self.assertEqual(len(requests), 2)
