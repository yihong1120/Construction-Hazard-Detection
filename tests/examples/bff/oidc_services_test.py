from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from urllib.parse import parse_qs
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import HTTPException

from examples.auth.deployment_context import DeploymentBinding
from examples.bff import oidc_services
from examples.bff.schemas import UserSummary
from tests.examples.auth.session_store_test import FakeRedis

_DEPLOYMENT = DeploymentBinding(
    tenant_id=UUID('00000000-0000-0000-0000-000000000001'),
    deployment_id=UUID('00000000-0000-0000-0000-000000000002'),
    api_base_url='https://api.example.com',
    config_revision=1,
)


class TestBffOidcServices(unittest.IsolatedAsyncioTestCase):
    """Exercise the BFF-only authorization-code and refresh boundaries."""

    def setUp(self) -> None:
        """Install deterministic OIDC settings without changing process env."""
        self.redis = FakeRedis()
        self.request = MagicMock()
        self.db = AsyncMock()
        self.settings_patch = unittest.mock.patch.object(
            oidc_services,
            'settings',
            SimpleNamespace(
                oidc_state_ttl_seconds=300,
                oidc_web_authorization_endpoint=(
                    'https://sso.example.com/authorize'
                ),
                oidc_web_client_configured=True,
                oidc_web_client_id='visionnaire-web',
                oidc_web_client_secret='client-secret',
                oidc_web_redirect_uri=(
                    'https://app.example.com/bff/auth/oidc/callback'
                ),
                oidc_identity_provider='keycloak',
            ),
        )
        self.settings_patch.start()
        self.addCleanup(self.settings_patch.stop)
        self.verifier = MagicMock()
        self.verifier.decode_access_token = AsyncMock(
            return_value={
                'iss': 'https://sso.example.com/realms/visionnaire',
                'sub': 'keycloak-user',
            },
        )
        self.verifier_patch = unittest.mock.patch.object(
            oidc_services,
            '_access_verifier',
            self.verifier,
        )
        self.verifier_patch.start()
        self.addCleanup(self.verifier_patch.stop)

    async def test_login_redirect_uses_single_use_state_and_pkce(self) -> None:
        """The browser redirect has state, S256 PKCE, and no open redirect."""
        with unittest.mock.patch.object(
            oidc_services,
            'resolve_request_deployment',
            new=AsyncMock(return_value=_DEPLOYMENT),
        ):
            response = await oidc_services.oidc_login_redirect(
                self.request,
                self.redis,  # type: ignore[arg-type]
                self.db,
                return_to='https://attacker.example.invalid',
            )

        self.assertEqual(response.status_code, 307)
        query = parse_qs(urlsplit(response.headers['location']).query)
        self.assertEqual(query['code_challenge_method'], ['S256'])
        self.assertEqual(query['client_id'], ['visionnaire-web'])
        state = query['state'][0]
        raw = await self.redis.get(oidc_services._state_key(state))
        assert raw is not None
        state_record = json.loads(raw)
        self.assertEqual(state_record['return_to'], '/')
        self.assertIn('code_verifier', state_record)

    async def test_login_redirect_allows_only_known_social_provider_hints(
        self,
    ) -> None:
        """A branded button can select a broker without arbitrary redirects."""
        with unittest.mock.patch.object(
            oidc_services,
            'resolve_request_deployment',
            new=AsyncMock(return_value=_DEPLOYMENT),
        ):
            response = await oidc_services.oidc_login_redirect(
                self.request,
                self.redis,  # type: ignore[arg-type]
                self.db,
                return_to='/violations',
                idp_hint='Google',
            )

        query = parse_qs(urlsplit(response.headers['location']).query)
        self.assertEqual(query['kc_idp_hint'], ['google'])

        with self.assertRaises(HTTPException) as error:
            await oidc_services.oidc_login_redirect(
                self.request,
                self.redis,  # type: ignore[arg-type]
                self.db,
                return_to='/violations',
                idp_hint='attacker-identity-provider',
            )
        self.assertEqual(error.exception.status_code, 400)

    async def test_callback_creates_a_token_private_session(self) -> None:
        """Callback maps a verified OIDC token to an opaque BFF session."""
        state = 'single-use-state'
        await self.redis.set(
            oidc_services._state_key(state),
            json.dumps(
                {
                    'code_verifier': 'pkce-verifier',
                    'deployment': _DEPLOYMENT.as_response(),
                    'return_to': '/violations',
                },
            ).encode('utf-8'),
            ex=300,
        )
        with (
            unittest.mock.patch.object(
                oidc_services,
                'resolve_request_deployment',
                new=AsyncMock(return_value=_DEPLOYMENT),
            ),
            unittest.mock.patch.object(
                oidc_services,
                '_post_token_form',
                new=AsyncMock(
                    return_value={
                        'access_token': 'keycloak-access',
                        'refresh_token': 'keycloak-refresh',
                    },
                ),
            ) as token_exchange,
            unittest.mock.patch.object(
                oidc_services,
                'subject_from_oidc_identity',
                new=AsyncMock(
                    return_value={
                        'user_id': 7,
                        'features': ['streaming'],
                    },
                ),
            ),
            unittest.mock.patch.object(
                oidc_services,
                '_user_summary',
                new=AsyncMock(
                    return_value=UserSummary(
                        id=7,
                        username='alice',
                        display_name='Alice',
                        role='user',
                        group_id=2,
                        status='active',
                    ),
                ),
            ),
            unittest.mock.patch.object(
                oidc_services,
                'create_auth_session',
                new=AsyncMock(return_value=('opaque-session', {})),
            ),
        ):
            response = await oidc_services.complete_oidc_login(
                self.request,
                self.redis,  # type: ignore[arg-type]
                self.db,
                code='authorisation-code',
                state=state,
            )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers['location'], '/violations')
        self.assertIn('HttpOnly', response.headers['set-cookie'])
        assert token_exchange.await_args is not None
        self.assertEqual(
            token_exchange.await_args.args[0]['code_verifier'],
            'pkce-verifier',
        )
        with self.assertRaises(HTTPException):
            await oidc_services._consume_state(
                self.redis,  # type: ignore[arg-type]
                state,
            )

    async def test_refresh_rejects_invalid_new_access_token(self) -> None:
        """A refresh result is not trusted before API-token validation."""
        self.verifier.decode_access_token.side_effect = HTTPException(
            status_code=401,
            detail='invalid',
        )
        with unittest.mock.patch.object(
            oidc_services,
            '_post_token_form',
            new=AsyncMock(
                return_value={
                    'access_token': 'bad-access',
                    'refresh_token': 'new-refresh',
                },
            ),
        ):
            with self.assertRaises(HTTPException):
                await oidc_services.refresh_oidc_tokens('old-refresh')
