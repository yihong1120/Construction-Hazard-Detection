"""Token-shaped fixtures for storage tests; OIDC verifier tests use RSA
keys.
"""
from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from datetime import timezone

import jwt


def fixture_token(
        subject,
        expires_delta=None,
        *,
        issuer='https://sso.example.test/realms/test',
        audience='test-api',
):
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {
            'sub': subject['username'], 'subject': subject,
            'jti': subject.get('jti', 'test-jti'),
            'iss': issuer,
            'aud': audience,
            'iat': now, 'exp': now + (expires_delta or timedelta(minutes=15)),
            'typ': 'Bearer',
        }, 'storage-fixture-secret-at-least-32-bytes',
    )
