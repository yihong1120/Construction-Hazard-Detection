from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch
from uuid import UUID

from fastapi import HTTPException
from fastapi import Request
from jwt.exceptions import InvalidTokenError
from redis.exceptions import RedisError

from examples.auth.deployment_context import DeploymentBinding
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.jwt_config import PyJWTBearer


class TestOidcBearer(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.binding = DeploymentBinding(
            tenant_id=UUID('00000000-0000-0000-0000-000000000001'),
            deployment_id=UUID('00000000-0000-0000-0000-000000000002'),
            api_base_url='https://api.example.com', config_revision=1,
        )
        self.subject = {
            'username': 'alice',
            'user_id': 1,
            'role': 'user',
            'jti': 'oidc:test:jti',
            'features': [],
            'tenant_id': str(
                self.binding.tenant_id,
            ),
            'deployment_id': str(
                self.binding.deployment_id,
            ),
            'config_revision': 1,
        }
        self.verifier = MagicMock()
        self.verifier.matches_configured_issuer.return_value = True
        self.verifier.decode_access_token = AsyncMock(
            return_value={'typ': 'Bearer'},
        )
        self.bearer = PyJWTBearer(oidc_verifier=self.verifier)

    async def test_only_oidc_issuer_is_accepted(self):
        self.verifier.matches_configured_issuer.return_value = False
        with self.assertRaises(InvalidTokenError):
            await self.bearer.decode_access_token_for_deployment(
                'local-token', AsyncMock(), self.binding,
            )
        self.verifier.decode_access_token.assert_not_awaited()

    async def test_missing_provider_fails_closed(self):
        with self.assertRaises(HTTPException) as raised:
            await PyJWTBearer().decode_access_token_for_deployment(
                'token', AsyncMock(), self.binding,
            )
        self.assertEqual(raised.exception.status_code, 503)

    async def test_id_token_is_not_an_api_access_token(self):
        self.verifier.decode_access_token.return_value = {'typ': 'ID'}
        with self.assertRaises(InvalidTokenError):
            await self.bearer.decode_access_token_for_deployment(
                'token', AsyncMock(), self.binding,
            )

    async def test_identity_mapping_requires_current_deployment(self):
        with patch(
            'examples.auth.jwt_config.subject_from_oidc_identity',
            AsyncMock(return_value=self.subject),
        ):
            result = await self.bearer.decode_access_token_for_deployment(
                'token', AsyncMock(), self.binding,
            )
            self.assertTrue(result.is_oidc)
            self.assertEqual(result['username'], 'alice')
            self.assertEqual(result.get('role'), 'user')
            self.subject['config_revision'] = 2
            with self.assertRaises(HTTPException) as raised:
                await self.bearer.decode_access_token_for_deployment(
                    'token', AsyncMock(), self.binding,
                )
            self.assertEqual(raised.exception.status_code, 409)

    async def test_http_bearer_checks_revocation_and_availability(self):
        request = Request({
            'type': 'http', 'method': 'GET', 'scheme': 'https',
            'path': '/', 'query_string': b'', 'headers': [],
        })
        request.scope['app'] = SimpleNamespace(
            state=SimpleNamespace(
                redis_client=SimpleNamespace(client=AsyncMock()),
            ),
        )
        verified = JwtAuthorizationCredentials(subject=self.subject)
        with (
            patch.object(
                self.bearer,
                'bearer_scheme',
                AsyncMock(
                    return_value=SimpleNamespace(credentials='token'),
                ),
            ),
            patch('examples.auth.jwt_config.AsyncSessionLocal'),
            patch(
                'examples.auth.jwt_config.resolve_request_deployment',
                AsyncMock(return_value=self.binding),
            ),
            patch.object(
                self.bearer,
                'decode_access_token_for_deployment',
                AsyncMock(
                    return_value=verified,
                ),
            ),
            patch(
                'examples.auth.jwt_config.is_access_token_revoked',
                AsyncMock(
                    side_effect=[False, True, RedisError('unavailable')],
                ),
            ) as revoked,
        ):
            self.assertTrue((await self.bearer(request)).is_oidc)
            for expected in (401, 503):
                with self.assertRaises(HTTPException) as raised:
                    await self.bearer(request)
                self.assertEqual(raised.exception.status_code, expected)
            self.assertEqual(revoked.await_count, 3)
        with patch.object(
            self.bearer, 'bearer_scheme', AsyncMock(return_value=None),
        ):
            with self.assertRaises(HTTPException) as raised:
                await self.bearer(request)
            self.assertEqual(raised.exception.status_code, 401)
