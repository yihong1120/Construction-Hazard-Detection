from __future__ import annotations

import json
import unittest

import httpx
import numpy as np

from src.notifiers.messenger_notifier import MessengerNotifier


class MessengerNotifierTests(unittest.IsolatedAsyncioTestCase):
    """Verify Messenger delivery uses the injected async transport."""

    async def test_sends_text_and_image(self) -> None:
        """Test sends text and image."""
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            """Perform handler.

            Args:
                request: Value used by this callable.

            Returns:
                The callable result.
            """
            requests.append(request)
            return httpx.Response(200)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        ) as client:
            notifier = MessengerNotifier('token', client=client)
            self.assertEqual(
                await notifier.send_notification('user', 'hello'), 200,
            )
            self.assertEqual(
                await notifier.send_notification(
                    'user',
                    'image',
                    image=np.zeros((2, 2, 3), dtype=np.uint8),
                ),
                200,
            )

        self.assertEqual(requests[0].url.params['access_token'], 'token')
        self.assertEqual(
            json.loads(requests[0].content)['recipient'],
            {'id': 'user'},
        )
        self.assertIn(b'filedata', requests[1].content)

    async def test_requires_a_page_access_token(self) -> None:
        """Test requires a page access token."""
        notifier = MessengerNotifier(page_access_token='')
        with self.assertRaisesRegex(ValueError, 'FACEBOOK_PAGE_ACCESS_TOKEN'):
            await notifier.send_notification('user', 'hello')

    async def test_standalone_notifier_reuses_and_closes_its_client(
        self,
    ) -> None:
        """A standalone notifier owns a reusable client until closed."""
        notifier = MessengerNotifier('token')
        client = notifier._http_client()
        self.assertIs(client, notifier._http_client())
        await notifier.aclose()
        self.assertTrue(client.is_closed)
