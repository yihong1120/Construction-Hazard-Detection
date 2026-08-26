"""Standards-based OIDC access-token verification for API services.

The application accepts an external token only when its unverified issuer is
the server-configured issuer and its signature, issuer, expiration, and API
audience subsequently validate against the configured JWKS. This issuer check
is merely a routing decision; it never grants access before full verification.
"""
from __future__ import annotations

import asyncio
from typing import Any

import jwt
from jwt.exceptions import InvalidTokenError

from examples.auth.config import Settings

_ASYMMETRIC_OIDC_ALGORITHMS = frozenset(
    {
        'ES256',
        'ES384',
        'ES512',
        'PS256',
        'PS384',
        'PS512',
        'RS256',
        'RS384',
        'RS512',
    },
)


class OidcTokenVerifier:
    """Verify external OIDC JWTs through a cached, configured JWKS."""

    def __init__(
        self,
        *,
        issuer: str,
        jwks_url: str,
        audiences: tuple[str, ...],
        algorithms: tuple[str, ...],
        jwks_cache_seconds: int = 300,
        jwks_timeout_seconds: float = 5,
    ) -> None:
        """Create a verifier with a fixed issuer and asymmetric algorithms."""
        self.issuer = issuer.rstrip('/')
        self.audiences = audiences
        self.algorithms = algorithms
        if not self.issuer or not jwks_url or not audiences:
            raise ValueError(
                'OIDC issuer, JWKS URL, and audience are required',
            )
        if (
            not algorithms
            or not set(algorithms) <= _ASYMMETRIC_OIDC_ALGORITHMS
        ):
            raise ValueError(
                'OIDC algorithms must use explicit asymmetric JWT algorithms',
            )
        self._jwks_client = jwt.PyJWKClient(
            jwks_url,
            cache_keys=True,
            lifespan=jwks_cache_seconds,
            timeout=jwks_timeout_seconds,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> OidcTokenVerifier | None:
        """Build the enabled verifier, or return ``None`` during migration."""
        if not settings.oidc_enabled:
            return None
        return cls(
            issuer=settings.oidc_issuer_url,
            jwks_url=settings.oidc_jwks_url,
            audiences=settings.oidc_audiences,
            algorithms=settings.oidc_algorithms,
            jwks_cache_seconds=settings.oidc_jwks_cache_seconds,
            jwks_timeout_seconds=settings.oidc_jwks_timeout_seconds,
        )

    def matches_configured_issuer(self, token: str) -> bool:
        """Return whether a token claims this verifier's issuer.

        The result is used only to choose the verifier. Signature validation is
        still mandatory in :meth:`decode_access_token`.
        """
        try:
            payload = jwt.decode(
                token,
                options={
                    'verify_signature': False,
                    'verify_exp': False,
                    'verify_aud': False,
                    'verify_iss': False,
                },
            )
        except jwt.PyJWTError:
            return False
        return payload.get('iss') == self.issuer

    async def decode_access_token(self, token: str) -> dict[str, Any]:
        """Verify and decode an OIDC access token using the provider JWKS."""
        try:
            signing_key = await asyncio.to_thread(
                self._jwks_client.get_signing_key_from_jwt,
                token,
            )
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=list(self.algorithms),
                audience=list(self.audiences),
                issuer=self.issuer,
                options={'require': ['aud', 'exp', 'iat', 'iss', 'sub']},
            )
        except jwt.PyJWTError as exc:
            raise InvalidTokenError('Invalid OIDC access token') from exc
        if not isinstance(payload.get('sub'), str) or not payload['sub']:
            raise InvalidTokenError('OIDC access token has an invalid subject')
        return dict(payload)
