from __future__ import annotations

import json
import unittest
from functools import partial

import httpx

from src.notifiers.broadcast_notifier import BroadcastNotifier


def _broadcast_message_success_handler(
    requests: list[httpx.Request],
    /,
    request: httpx.Request,
) -> httpx.Response:
    requests.append(request)
    return httpx.Response(204)


def _broadcast_message_failure_or_network_Response(_request):
    return httpx.Response(503)


class BroadcastNotifierTests(unittest.IsolatedAsyncioTestCase):
    """Verify broadcast delivery uses async HTTP transport."""

    async def test_broadcast_message_success(self) -> None:
        """Test broadcast message success."""
        requests: list[httpx.Request] = []

        handler = partial(_broadcast_message_success_handler, requests)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        ) as client:
            notifier = BroadcastNotifier(
                'https://example.test/broadcast',
                client=client,
            )
            self.assertTrue(await notifier.broadcast_message('hello'))

        self.assertEqual(json.loads(requests[0].content), {'message': 'hello'})

    async def test_broadcast_message_failure_or_network_error(self) -> None:
        """Test broadcast message failure or network error."""

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                _broadcast_message_failure_or_network_Response,
            ),
        ) as client:
            notifier = BroadcastNotifier(
                'https://example.test/broadcast',
                client=client,
            )
            self.assertFalse(await notifier.broadcast_message('hello'))

    async def test_standalone_notifier_reuses_and_closes_its_client(
        self,
    ) -> None:
        """A library consumer gets one reusable transport per notifier."""
        notifier = BroadcastNotifier('https://example.test/broadcast')
        client = notifier._http_client()
        self.assertIs(client, notifier._http_client())
        await notifier.aclose()
        self.assertTrue(client.is_closed)
