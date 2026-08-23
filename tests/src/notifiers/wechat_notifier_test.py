from __future__ import annotations

import unittest

import httpx
import numpy as np

from src.notifiers.wechat_notifier import WeChatNotifier


class WeChatNotifierTests(unittest.IsolatedAsyncioTestCase):
    """Verify WeChat token caching and async image delivery."""

    async def test_token_is_cached_and_image_is_uploaded(self) -> None:
        """Test token is cached and image is uploaded.
        """
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            """Perform handler.

            Args:
                request: Value used by this callable.

            Returns:
                The callable result.
            """
            requests.append(request)
            if request.url.path.endswith('/gettoken'):
                return httpx.Response(
                    200,
                    json={'access_token': 'token', 'expires_in': 7200},
                )
            if request.url.path.endswith('/media/upload'):
                return httpx.Response(200, json={'media_id': 'media'})
            return httpx.Response(200, json={'errcode': 0, 'errmsg': 'ok'})

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        ) as client:
            notifier = WeChatNotifier(
                corp_id='corp',
                corp_secret='secret',
                agent_id=1,
                client=client,
            )
            self.assertEqual(
                await notifier.send_notification('user', 'hello'),
                {'errcode': 0, 'errmsg': 'ok'},
            )
            self.assertEqual(
                await notifier.send_notification(
                    'user',
                    'image',
                    image=np.zeros((2, 2, 3), dtype=np.uint8),
                ),
                {'errcode': 0, 'errmsg': 'ok'},
            )

        token_requests = [
            request for request in requests if request.url.path.endswith('/gettoken')
        ]
        self.assertEqual(len(token_requests), 1)
        self.assertTrue(any(b'media' in request.content for request in requests))

    async def test_requires_wechat_credentials(self) -> None:
        """Test requires wechat credentials.
        """
        notifier = WeChatNotifier(corp_id='', corp_secret='')
        with self.assertRaisesRegex(ValueError, 'WECHAT_CORP_ID'):
            await notifier.get_access_token()

    async def test_rejects_a_token_response_without_a_token(self) -> None:
        """Malformed token responses fail explicitly instead of caching None."""
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(200, json={}),
            ),
        ) as client:
            notifier = WeChatNotifier('corp', 'secret', client=client)
            with self.assertRaisesRegex(ValueError, 'did not return an access token'):
                await notifier.get_access_token()

    async def test_standalone_notifier_reuses_and_closes_its_client(self) -> None:
        """A standalone notifier owns a reusable client until closed."""
        notifier = WeChatNotifier('corp', 'secret')
        client = notifier._http_client()
        self.assertIs(client, notifier._http_client())
        await notifier.aclose()
        self.assertTrue(client.is_closed)
