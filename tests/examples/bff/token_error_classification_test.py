from __future__ import annotations

import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

import httpx
import jwt
from fastapi import HTTPException
from redis.asyncio import Redis

from examples.auth import session_store as sessions
from examples.bff import oidc_services as oidc
from examples.bff import proxy_tokens as tokens


class TokenErrorClassificationTest(unittest.IsolatedAsyncioTestCase):
    async def test_only_explicit_invalid_grant_is_terminal(self):
        cases = [
            (429, {'error': 'temporarily_unavailable'}, 429, False),
            (503, {'error': 'server_error'}, 503, False),
            (408, {'error': 'invalid_grant'}, 503, False),
            (400, {'error': 'invalid_grant'}, 401, True),
            (401, {'error': 'invalid_client'}, 503, False),
            (403, {'error': 'access_denied'}, 503, False),
            (400, {'error': 'invalid_request'}, 503, False),
            (400, [], 503, False),
            (400, None, 503, False),
        ]
        for status, payload, expected, terminal in cases:
            with self.subTest(status=status, payload=payload):
                response = httpx.Response(
                    status,
                    request=httpx.Request('POST', 'https://idp.example/token'),
                    headers={'Retry-After': '30'},
                    json=payload,
                )
                client = SimpleNamespace(post=AsyncMock(return_value=response))
                with patch.object(oidc, '_require_web_client'), patch.object(
                    oidc,
                    'get_application_http_client',
                    AsyncMock(return_value=client),
                ):
                    with self.assertRaises(HTTPException) as caught:
                        await oidc._post_token_form(
                            {'grant_type': 'refresh_token'},
                        )
                self.assertEqual(caught.exception.status_code, expected)
                self.assertEqual(
                    tokens._is_terminal_refresh_error(caught.exception),
                    terminal,
                )
                if not terminal:
                    self.assertEqual(
                        caught.exception.headers, {'Retry-After': '30'},
                    )

    async def test_malformed_success_is_retryable(self):
        for payload in [
            [],
            {},
            {'access_token': 'access'},
            {'access_token': '', 'refresh_token': 'refresh'},
        ]:
            response = httpx.Response(
                200,
                request=httpx.Request('POST', 'https://idp.example/token'),
                json=payload,
            )
            with patch.object(oidc, '_require_web_client'), patch.object(
                oidc,
                'get_application_http_client',
                AsyncMock(
                    return_value=SimpleNamespace(
                        post=AsyncMock(return_value=response),
                    ),
                ),
            ):
                with self.assertRaises(HTTPException) as caught:
                    await oidc._post_token_form({})
            self.assertEqual(caught.exception.status_code, 503)

    @unittest.skipUnless(
        os.getenv('NOTIFICATION_REDIS_TEST_URL'), 'Requires isolated Redis',
    )
    async def test_throttling_preserves_the_session_and_releases_ownership(
        self,
    ):
        redis = Redis.from_url(os.environ['NOTIFICATION_REDIS_TEST_URL'])
        token = jwt.encode(
            {'exp': int(time.time()) + 300}, key='', algorithm='none',
        )
        sid, _ = await sessions.create_auth_session(
            redis,
            {
                'access_token': token,
                'refresh_token': 'refresh',
                'feature_names': [],
            },
            {'id': 1},
        )
        try:
            original = await sessions.get_auth_session(redis, sid)
            response = httpx.Response(
                429,
                request=httpx.Request('POST', 'https://idp.example/token'),
                headers={'Retry-After': '5'},
                json={'error': 'temporarily_unavailable'},
            )
            client = SimpleNamespace(post=AsyncMock(return_value=response))
            with patch.object(oidc, '_require_web_client'), patch.object(
                oidc,
                'get_application_http_client',
                AsyncMock(return_value=client),
            ), patch.object(
                tokens, 'refresh_oidc_tokens', oidc.refresh_oidc_tokens,
            ):
                with self.assertRaises(HTTPException) as caught:
                    await tokens.get_proxy_access_token(
                        redis, sid, force_refresh=True,
                    )
            self.assertEqual(caught.exception.status_code, 429)
            self.assertEqual(caught.exception.headers, {'Retry-After': '5'})
            self.assertEqual(
                await sessions.get_auth_session(redis, sid), original,
            )
            self.assertIsNone(
                await redis.get(
                    sessions.auth_session_key(sid) + ':refresh-lock',
                ),
            )
        finally:
            await sessions.delete_auth_session(redis, sid)
            await redis.aclose()
