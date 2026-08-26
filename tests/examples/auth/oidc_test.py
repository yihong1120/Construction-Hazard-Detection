from __future__ import annotations

import unittest
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from uuid import UUID

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.exceptions import InvalidTokenError

from examples.auth.deployment_context import DeploymentBinding
from examples.auth.oidc import OidcTokenVerifier
from examples.auth.oidc_identity import subject_from_oidc_identity

_ISSUER = 'https://sso.example.com/realms/visionnaire'
_AUDIENCE = 'visionnaire-api'
_DEPLOYMENT = DeploymentBinding(
    tenant_id=UUID('00000000-0000-0000-0000-000000000001'),
    deployment_id=UUID('00000000-0000-0000-0000-000000000002'),
    api_base_url='https://api.example.com',
    config_revision=1,
)


def _claims(**overrides: object) -> dict[str, object]:
    """Build a valid Keycloak-like access token payload for a test."""
    now = datetime.now(timezone.utc)
    claims: dict[str, object] = {
        'iss': _ISSUER,
        'sub': 'keycloak-user-id',
        'aud': _AUDIENCE,
        'iat': now,
        'exp': now + timedelta(minutes=5),
        'jti': 'keycloak-token-id',
    }
    claims.update(overrides)
    return claims


class TestOidcTokenVerifier(unittest.IsolatedAsyncioTestCase):
    """Validate the fixed issuer/JWKS/token-audience security contract."""

    def setUp(self) -> None:
        """Create an in-memory RSA key pair and configured verifier."""
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        self.verifier = OidcTokenVerifier(
            issuer=_ISSUER,
            jwks_url='https://sso.example.com/certs',
            audiences=(_AUDIENCE,),
            algorithms=('RS256',),
        )
        self.verifier._jwks_client = MagicMock()
        self.verifier._jwks_client.get_signing_key_from_jwt.return_value = (
            SimpleNamespace(key=self.private_key.public_key())
        )

    def _token(self, **overrides: object) -> str:
        """Sign a test access token with the temporary private key."""
        return jwt.encode(
            _claims(**overrides),
            self.private_key,
            algorithm='RS256',
            headers={'kid': 'test-key'},
        )

    async def test_verifies_issuer_audience_and_signature(self) -> None:
        """Only the configured issuer and API audience are accepted."""
        token = self._token()

        claims = await self.verifier.decode_access_token(token)

        self.assertEqual(claims['sub'], 'keycloak-user-id')
        self.assertTrue(self.verifier.matches_configured_issuer(token))

        with self.assertRaises(InvalidTokenError):
            await self.verifier.decode_access_token(
                self._token(aud='open-webui'),
            )

    async def test_rejects_a_token_with_an_untrusted_signature(self) -> None:
        """A matching issuer alone never authorises a forged token."""
        other_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        token = jwt.encode(
            _claims(),
            other_key,
            algorithm='RS256',
            headers={'kid': 'test-key'},
        )

        with self.assertRaises(InvalidTokenError):
            await self.verifier.decode_access_token(token)


class TestOidcIdentityMapping(unittest.IsolatedAsyncioTestCase):
    """Ensure OIDC authentication retains local authorisation boundaries."""

    def setUp(self) -> None:
        """Provide a local linked user and feature query result."""
        self.db = AsyncMock()
        self.db.scalar.return_value = SimpleNamespace(user_id=7)
        self.db.get.return_value = SimpleNamespace(
            id=7,
            username='alice',
            role='user',
            status='active',
            group_id=2,
            tenant_id=_DEPLOYMENT.tenant_id,
        )
        features = MagicMock()
        features.scalars.return_value = ['streaming', 'violations']
        self.db.execute.return_value = features

    async def test_linked_user_receives_local_permissions(self) -> None:
        """The external subject maps to current local groups and deployment."""
        subject = await subject_from_oidc_identity(
            self.db,
            _claims(),
            provider='keycloak',
            binding=_DEPLOYMENT,
        )

        self.assertEqual(subject['username'], 'alice')
        self.assertEqual(subject['user_id'], 7)
        self.assertEqual(subject['features'], ['streaming', 'violations'])
        self.assertEqual(subject['tenant_id'], str(_DEPLOYMENT.tenant_id))
        self.assertEqual(
            subject['deployment_id'],
            str(_DEPLOYMENT.deployment_id),
        )
        self.assertEqual(
            subject['jti'],
            f'oidc:{_ISSUER}:keycloak-token-id',
        )

    async def test_unlinked_external_user_is_rejected(self) -> None:
        """Username or email are never used as an unsafe implicit link."""
        self.db.scalar.return_value = None

        with self.assertRaises(InvalidTokenError):
            await subject_from_oidc_identity(
                self.db,
                _claims(),
                provider='keycloak',
                binding=_DEPLOYMENT,
            )
