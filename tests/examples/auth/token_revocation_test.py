from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

from examples.auth import token_revocation


def test_access_token_jti_reads_direct_and_nested_values() -> None:
    """The helper supports both current and older signed token payloads."""
    assert token_revocation.access_token_jti({'jti': 'current'}) == 'current'
    assert token_revocation.access_token_jti(
        {'subject': {'jti': 'nested'}},
    ) == 'nested'
    assert token_revocation.access_token_jti({'jti': ''}) is None


def test_remaining_lifetime_handles_invalid_and_expired_values(
        monkeypatch: Any,
) -> None:
    """Malformed expiration claims cannot create permanent revocations."""
    monkeypatch.setattr(token_revocation.time, 'time', lambda: 100)

    assert token_revocation._remaining_lifetime({}) == 0
    assert token_revocation._remaining_lifetime({'exp': 'not-a-time'}) == 0
    assert token_revocation._remaining_lifetime({'exp': 90}) == 0
    assert token_revocation._remaining_lifetime({'exp': 120}) == 20


def test_revoke_and_query_access_tokens() -> None:
    """Only live token identifiers are stored and queried in Redis."""
    async def run_case() -> None:
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
            assert await token_revocation.is_access_token_revoked(
                redis,
                {},
            ) is True
        finally:
            token_revocation.time.time = original_time

    asyncio.run(run_case())
