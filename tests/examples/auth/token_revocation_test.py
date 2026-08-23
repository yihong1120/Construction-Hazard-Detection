from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

from examples.auth import token_revocation


def test_access_token_jti_reads_canonical_value() -> None:
    """The helper reads the canonical top-level access-token claim."""
    assert token_revocation.access_token_jti({'jti': 'current'}) == 'current'


def test_remaining_lifetime_handles_expired_values(
        monkeypatch: Any,
) -> None:
    """Expired access tokens cannot create permanent revocations."""
    monkeypatch.setattr(token_revocation.time, 'time', lambda: 100)

    assert token_revocation._remaining_lifetime(
        {'jti': 'expired', 'exp': 90},
    ) == 0
    assert token_revocation._remaining_lifetime(
        {'jti': 'live', 'exp': 120},
    ) == 20


def test_revoke_and_query_access_tokens() -> None:
    """Only live token identifiers are stored and queried in Redis."""
    async def run_case() -> None:
        """Perform run case.
        """
        redis = AsyncMock()
        redis.exists.return_value = 1
        original_time = token_revocation.time.time
        token_revocation.time.time = lambda: 100
        try:
            assert await token_revocation.revoke_access_token(
                redis,
                {'jti': 'live', 'exp': 130},
            ) is True
            assert await token_revocation.revoke_access_token(
                redis,
                {'jti': 'expired', 'exp': 90},
            ) is False
            assert await token_revocation.revoke_access_token_jtis(
                redis,
                {'another-live': 140, 'expired': 80},
            ) == 1
            assert await token_revocation.is_access_token_revoked(
                redis,
                {'jti': 'live'},
            ) is True
        finally:
            token_revocation.time.time = original_time

    asyncio.run(run_case())
