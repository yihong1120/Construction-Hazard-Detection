from __future__ import annotations

import json
import time
import unittest
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException
from starlette.requests import Request

from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.db_management.schemas.auth import (
    NativeSocialEmailLinkConfirmRequest,
)
from examples.db_management.schemas.auth import (
    NativeSocialExchangeBeginRequest,
)
from examples.db_management.schemas.auth import (
    NativeSocialExchangeCompleteRequest,
)
from examples.db_management.schemas.auth import NativeSocialLinkCompleteRequest
from examples.db_management.schemas.auth import ProviderClaims
from examples.db_management.services import (
    native_social_broker as social_broker,
)
from examples.db_management.services import (
    native_social_exchange_services as svc,
)
from examples.db_management.services import native_social_links as social_links
from examples.db_management.services import native_social_proof as social_proof
from tests.examples.auth.session_store_test import FakeRedis


class _RequestCallbacks:
    body: Any
    sent: Any

    async def _call_receive(self) -> dict[str, object]:

        if self.sent:
            return {'type': 'http.request', 'body': b'', 'more_body': False}
        self.sent = True
        return {
            'type': 'http.request',
            'body': self.body,
            'more_body': False,
        }


def _request(
    body: bytes = b'',
    headers: dict[str, str] | None = None,
) -> Request:
    """Build a loopback Starlette request with an optional raw body."""
    _callbacks = _RequestCallbacks()
    _callbacks.body = body
    _callbacks.sent = False

    receive = _callbacks._call_receive

    return Request(
        {
            'type': 'http',
            'method': 'POST',
            'path': '/auth/native-social/exchanges',
            'scheme': 'http',
            'server': ('127.0.0.1', 8005),
            'client': ('127.0.0.1', 43210),
            'headers': [
                (key.lower().encode('ascii'), value.encode('ascii'))
                for key, value in (headers or {}).items()
            ],
        },
        receive,
    )


class _NativeSocialRedis(FakeRedis):
    """Add the counter primitive used by the anonymous begin limiter."""

    async def incr(self, key: str) -> int:
        value = int(self.data.get(key, b'0')) + 1
        self.data[key] = str(value).encode('ascii')
        return value


class NativeSocialExchangeServicesTest(unittest.IsolatedAsyncioTestCase):
    """Verify nonce, PKCE, one-use redemption, and account-link boundaries."""

    def setUp(self) -> None:
        self.redis = _NativeSocialRedis()
        self.db = MagicMock()
        self.db.scalar = AsyncMock()
        self.db.execute = AsyncMock()
        self.user = MagicMock(id=1)
        self._settings = {
            name: getattr(svc.settings, name)
            for name in (
                'native_social_exchange_enabled',
                'native_social_exchange_shared_secret',
                'native_social_allowed_clients_json',
                'native_social_exchange_ttl_seconds',
                'native_social_link_ttl_seconds',
                'native_social_link_max_auth_age_seconds',
                'oidc_issuer_url',
            )
        }
        svc.settings.native_social_exchange_enabled = True
        svc.settings.native_social_exchange_shared_secret = 'a' * 64
        svc.settings.native_social_allowed_clients_json = (
            '{"visionnaire-mobile":["com.changdar.visionnaire'
            ':/oauthredirect"]}'
        )
        svc.settings.native_social_exchange_ttl_seconds = 90
        svc.settings.native_social_link_ttl_seconds = 120
        svc.settings.native_social_link_max_auth_age_seconds = 300
        svc.settings.oidc_issuer_url = (
            'https://sso.example.com/keycloak/realms/visionnaire'
        )

    def tearDown(self) -> None:
        for name, value in self._settings.items():
            setattr(svc.settings, name, value)

    def _begin_payload(self) -> NativeSocialExchangeBeginRequest:
        return NativeSocialExchangeBeginRequest(
            provider='google',
            client_id='visionnaire-mobile',
            redirect_uri='com.changdar.visionnaire:/oauthredirect',
            code_challenge='a' * 43,
            code_challenge_method='S256',
            state='opaque-state',
        )

    async def test_exchange_is_nonce_and_pkce_bound_then_redeemed_once(
        self,
    ) -> None:
        begin = await svc.begin_native_social_exchange(
            self._begin_payload(),
            _request(),
            self.redis,
        )
        claims = ProviderClaims.model_validate({'sub': 'google-stable-sub'})
        with patch.object(
            social_proof,
            'verify_google_id_token',
            AsyncMock(return_value=claims),
        ) as verify, patch.object(
            svc,
            '_find_email_link_candidate',
            AsyncMock(return_value=None),
        ):
            completed = await svc.complete_native_social_exchange(
                NativeSocialExchangeCompleteRequest(
                    transaction_id=begin.transaction_id,
                    id_token='provider-id-token',
                ),
                self.redis,
                self.db,
            )
        verify.assert_awaited_once_with(
            'provider-id-token',
            expected_nonce=begin.nonce,
            require_verified_email=False,
        )
        self.assertIn('native_social_exchange=', completed.authorization_url)
        self.assertIn(
            'code_challenge=' + 'a' * 43,
            completed.authorization_url,
        )
        body = json.dumps(
            {
                'transaction_id': begin.transaction_id,
                'client_id': 'visionnaire-mobile',
                'redirect_uri': 'com.changdar.visionnaire:/oauthredirect',
                'code_challenge': 'a' * 43,
            },
            separators=(',', ':'),
        ).encode('utf-8')
        timestamp = str(int(time.time()))
        headers = {
            'X-Visionnaire-Timestamp': timestamp,
            'X-Visionnaire-Signature': social_broker._hmac_signature(
                timestamp,
                body,
            ),
        }
        redeemed = await social_broker.redeem_keycloak_native_social_exchange(
            _request(body, headers),
            self.redis,
        )
        self.assertEqual(redeemed['provider'], 'google')
        self.assertIn('provider_subject_b64', redeemed)
        with self.assertRaises(HTTPException) as ctx:
            await social_broker.redeem_keycloak_native_social_exchange(
                _request(body, headers),
                self.redis,
            )
        self.assertEqual(ctx.exception.status_code, 401)

    async def test_redeem_rejects_changed_pkce_binding(self) -> None:
        begin = await svc.begin_native_social_exchange(
            self._begin_payload(),
            _request(),
            self.redis,
        )
        with patch.object(
            social_proof,
            'verify_google_id_token',
            AsyncMock(
                return_value=ProviderClaims.model_validate({'sub': 'g'}),
            ),
        ), patch.object(
            svc,
            '_find_email_link_candidate',
            AsyncMock(return_value=None),
        ):
            await svc.complete_native_social_exchange(
                NativeSocialExchangeCompleteRequest(
                    transaction_id=begin.transaction_id,
                    id_token='id-token',
                ),
                self.redis,
                self.db,
            )
        body = json.dumps(
            {
                'transaction_id': begin.transaction_id,
                'client_id': 'visionnaire-mobile',
                'redirect_uri': 'com.changdar.visionnaire:/oauthredirect',
                'code_challenge': 'b' * 43,
            },
            separators=(',', ':'),
        ).encode('utf-8')
        timestamp = str(int(time.time()))
        headers = {
            'X-Visionnaire-Timestamp': timestamp,
            'X-Visionnaire-Signature': social_broker._hmac_signature(
                timestamp,
                body,
            ),
        }
        with self.assertRaises(HTTPException) as ctx:
            await social_broker.redeem_keycloak_native_social_exchange(
                _request(body, headers),
                self.redis,
            )
        self.assertEqual(ctx.exception.status_code, 401)

    async def test_link_requires_recent_keycloak_auth_and_target_is_derived(
        self,
    ) -> None:
        credentials = JwtAuthorizationCredentials(
            subject={
                'username': 'alice',
                'user_id': 1,
                'role': 'user',
                'jti': 'jti',
                'features': [],
            },
            payload={
                'iss': svc.settings.oidc_issuer_url,
                'sub': 'keycloak-user-id',
                'sid': 'session-id',
                'auth_time': int(time.time()),
            },
            token='keycloak-token',
        )
        started = await svc.begin_native_social_link(
            'apple',
            credentials,
            self.redis,
        )
        with patch.object(
            social_proof,
            'verify_apple_identity_token',
            AsyncMock(
                return_value=ProviderClaims.model_validate(
                    {'sub': 'apple-sub'},
                ),
            ),
        ) as verify, patch.object(
            social_links.NativeIdentityLink,
            'execute',
            autospec=True,
            return_value='linked',
        ) as link:
            result = await svc.complete_native_social_link(
                NativeSocialLinkCompleteRequest(
                    transaction_id=started.transaction_id,
                    id_token='apple-id-token',
                    authorization_code='apple-code',
                ),
                credentials,
                self.redis,
                self.db,
                self.user,
            )
        self.assertEqual(result.status, 'linked')
        verify.assert_awaited_once_with(
            'apple-id-token',
            'apple-code',
            expected_nonce=started.nonce,
        )
        link.assert_awaited_once_with(
            social_links.NativeIdentityLink(
                user_id=self.user.id,
                keycloak_subject='keycloak-user-id',
                identity=social_proof.VerifiedSocialIdentity(
                    'apple',
                    'apple-sub',
                ),
            ),
            self.db,
        )

    async def test_verified_email_requires_reauthentication_then_links(
        self,
    ) -> None:
        begin = await svc.begin_native_social_exchange(
            self._begin_payload(),
            _request(),
            self.redis,
        )
        claims = ProviderClaims.model_validate(
            {
                'sub': 'google-sub',
                'email': 'alice@example.com',
                'email_verified': True,
            },
        )
        candidate = social_links.EmailLinkCandidate(
            local_user_id=1,
            keycloak_subject='keycloak-user-id',
        )
        with patch.object(
            social_proof,
            'verify_google_id_token',
            AsyncMock(return_value=claims),
        ), patch.object(
            svc,
            '_find_email_link_candidate',
            AsyncMock(return_value=candidate),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await svc.complete_native_social_exchange(
                    NativeSocialExchangeCompleteRequest(
                        transaction_id=begin.transaction_id,
                        id_token='provider-id-token',
                    ),
                    self.redis,
                    self.db,
                )
        self.assertEqual(ctx.exception.status_code, 409)
        detail = ctx.exception.detail
        self.assertEqual(detail['code'], 'account_link_required')
        credentials = JwtAuthorizationCredentials(
            subject={
                'username': 'alice',
                'user_id': 1,
                'role': 'user',
                'jti': 'jti',
                'features': [],
            },
            payload={
                'iss': svc.settings.oidc_issuer_url,
                'sub': 'keycloak-user-id',
                'sid': 'session-id',
                'auth_time': int(time.time()),
            },
            token='keycloak-token',
        )
        with patch.object(
            social_links.NativeIdentityLink,
            'execute',
            autospec=True,
            return_value='linked',
        ) as link:
            result = await svc.confirm_native_social_email_link(
                NativeSocialEmailLinkConfirmRequest(
                    transaction_id=detail['link_transaction_id'],
                ),
                credentials,
                self.redis,
                self.db,
                self.user,
            )
        self.assertEqual(result.status, 'linked')
        link.assert_awaited_once_with(
            social_links.NativeIdentityLink(
                user_id=self.user.id,
                keycloak_subject='keycloak-user-id',
                identity=social_proof.VerifiedSocialIdentity(
                    'google',
                    'google-sub',
                ),
            ),
            self.db,
        )

    def test_email_candidate_requires_provider_verified_email(self) -> None:
        """Never use a provider's unverified email for account discovery."""
        unverified = ProviderClaims.model_validate(
            {
                'sub': 'google-sub',
                'email': 'Alice@Example.com',
                'email_verified': False,
            },
        )
        verified = ProviderClaims.model_validate(
            {
                'sub': 'google-sub',
                'email': ' Alice@Example.com ',
                'email_verified': True,
            },
        )
        self.assertIsNone(social_proof._normalised_verified_email(unverified))
        self.assertEqual(
            social_proof._normalised_verified_email(
                verified,
            ),
            'alice@example.com',
        )

    async def test_link_rejects_stale_auth_time(self) -> None:
        credentials = JwtAuthorizationCredentials(
            subject={
                'username': 'alice',
                'user_id': 1,
                'role': 'user',
                'jti': 'jti',
                'features': [],
            },
            payload={
                'iss': svc.settings.oidc_issuer_url,
                'sub': 'keycloak-user-id',
                'auth_time': int(time.time()) - 301,
            },
        )
        with self.assertRaises(HTTPException) as ctx:
            await svc.begin_native_social_link(
                'google',
                credentials,
                self.redis,
            )
        self.assertEqual(
            ctx.exception.detail,
            'keycloak_reauthentication_required',
        )
