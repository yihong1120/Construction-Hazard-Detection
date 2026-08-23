from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import patch

from jwt.exceptions import ExpiredSignatureError
from jwt.exceptions import InvalidTokenError

from examples.auth import token_cleanup as tc


def _cache(**overrides: object) -> dict[str, object]:
    """Build the canonical user-cache payload used by production code."""
    cache: dict[str, object] = {
        'db_user': {
            'id': 1,
            'username': 'user',
            'role': 'user',
            'group_id': None,
            'status': 'active',
        },
        'jti_list': [],
        'jti_meta': {},
        'refresh_tokens': [],
        'refresh_token_hashes': [],
        'refresh_token_families': {},
        'feature_names': [],
    }
    cache.update(overrides)
    return cache


class TestPruneUserCache(unittest.IsolatedAsyncioTestCase):
    """Behavioural tests for prune_user_cache covering key branches."""

    async def test_no_cache_returns_none(self) -> None:
        """When user cache is missing, return None and do not write back."""
        with (
            patch.object(
                tc.rate_limiter_service,
                'get_user_data',
                new=AsyncMock(return_value=None),
            ) as mock_get,
            patch.object(
                tc.rate_limiter_service,
                'set_user_data',
                new=AsyncMock(),
            ) as mock_set,
        ):
            out = await tc.prune_user_cache(object(), 'alice')
        self.assertIsNone(out)
        mock_get.assert_awaited_once()
        mock_set.assert_not_awaited()

    async def test_prune_refresh_tokens_mixed_validity(self) -> None:
        """Expired/invalid refresh tokens are removed; valid ones remain."""
        now = 1_000
        cache = _cache(
            refresh_tokens=['valid1', 'expired1', 'invalid1'],
        )
        # Configure refresh-token verification for valid and invalid tokens.

        def decode_side_effect(tok: str) -> dict[str, object]:
            """Support decode_side_effect."""
            if tok == 'valid1':
                return {'ok': True}
            if tok == 'expired1':
                raise ExpiredSignatureError('expired')
            raise InvalidTokenError('bad')

        with (
            patch.object(tc.time, 'time', return_value=now),
            patch.object(
                tc.rate_limiter_service,
                'get_user_data',
                new=AsyncMock(return_value=cache.copy()),
            ) as mock_get,
            patch.object(
                tc.rate_limiter_service,
                'set_user_data',
                new=AsyncMock(),
            ) as mock_set,
            patch.object(
                tc.jwt_refresh,
                'decode_token',
                side_effect=decode_side_effect,
            ) as mock_decode,
        ):
            out = await tc.prune_user_cache(object(), 'bob')

            self.assertEqual(
                out,
                _cache(refresh_tokens=['valid1']),
            )
        mock_get.assert_awaited_once()
        mock_set.assert_awaited_once()
        await_call = mock_set.await_args
        assert await_call is not None
        self.assertEqual(
            await_call.args[1:],
            (
                'bob',
                _cache(refresh_tokens=['valid1']),
            ),
        )
        # Ensure decode was attempted for each token
        self.assertEqual(
            [c.args[0] for c in mock_decode.call_args_list],
            ['valid1', 'expired1', 'invalid1'],
        )

    async def test_prune_jti_list_and_meta(self) -> None:
        """JTI list pruned by expiry; stale meta entries removed and preserved
        strictly."""
        now = 2_000
        cache = _cache(
            jti_list=['a', 'b', 'c'],
            # a unexpired, b expired, 'stale' not present in list (stale)
            jti_meta={'a': now + 10, 'b': now - 1, 'stale': now + 10},
        )
        with (
            patch.object(tc.time, 'time', return_value=now),
            patch.object(
                tc.rate_limiter_service,
                'get_user_data',
                new=AsyncMock(return_value=cache.copy()),
            ) as mock_get,
            patch.object(
                tc.rate_limiter_service,
                'set_user_data',
                new=AsyncMock(),
            ) as mock_set,
            # not used in this path
            patch.object(tc.jwt_refresh, 'decode_token', return_value={}),
        ):
            out = await tc.prune_user_cache(object(), 'carol')

        self.assertEqual(
            out,
            _cache(
                # c kept because missing meta counts as 0 -> keep
                jti_list=['a', 'c'],
                jti_meta={'a': now + 10},
            ),
        )
        mock_get.assert_awaited_once()
        mock_set.assert_awaited_once()
        await_call = mock_set.await_args
        assert await_call is not None
        self.assertEqual(
            await_call.args[1:],
            (
                'carol',
                _cache(
                    jti_list=['a', 'c'],
                    jti_meta={'a': now + 10},
                ),
            ),
        )

    async def test_no_change_does_not_write(self) -> None:
        """If nothing changes, avoid unnecessary writes to the cache."""
        now = 3_000
        cache = _cache(
            refresh_tokens=['still_valid'],
            jti_list=['x'],
            jti_meta={},  # empty -> pruning of jti is skipped
        )

        with (
            patch.object(tc.time, 'time', return_value=now),
            patch.object(
                tc.rate_limiter_service,
                'get_user_data',
                new=AsyncMock(return_value=cache.copy()),
            ) as mock_get,
            patch.object(
                tc.rate_limiter_service,
                'set_user_data',
                new=AsyncMock(),
            ) as mock_set,
            patch.object(
                tc.jwt_refresh,
                'decode_token',
                return_value={'ok': True},
            ) as mock_decode,
        ):
            out = await tc.prune_user_cache(object(), 'dave')

        # No changes: same cache returned and no write
        self.assertEqual(out, cache)
        mock_get.assert_awaited_once()
        mock_set.assert_not_awaited()
        mock_decode.assert_called_once_with('still_valid')

    async def test_combined_refresh_and_jti_changes(self) -> None:
        """Prune both refresh tokens and JTIs then persist the updated
        cache."""
        now = 4_000
        cache = _cache(
            refresh_tokens=['ok', 'bad'],
            jti_list=['keep', 'drop'],
            jti_meta={'keep': now + 5, 'drop': now - 5},
        )

        def decode_side_effect(tok: str) -> dict[str, object]:
            """Support decode_side_effect."""
            if tok == 'ok':
                return {}
            raise InvalidTokenError('bad')

        with (
            patch.object(tc.time, 'time', return_value=now),
            patch.object(
                tc.rate_limiter_service,
                'get_user_data',
                new=AsyncMock(return_value=cache.copy()),
            ) as mock_get,
            patch.object(
                tc.rate_limiter_service,
                'set_user_data',
                new=AsyncMock(),
            ) as mock_set,
            patch.object(
                tc.jwt_refresh,
                'decode_token',
                side_effect=decode_side_effect,
            ),
        ):
            out = await tc.prune_user_cache(object(), 'erin')

        self.assertEqual(
            out,
            _cache(
                refresh_tokens=['ok'],
                jti_list=['keep'],
                jti_meta={'keep': now + 5},
            ),
        )
        mock_get.assert_awaited_once()
        mock_set.assert_awaited_once()
        await_call = mock_set.await_args
        assert await_call is not None
        self.assertEqual(
            await_call.args[1:],
            (
                'erin',
                _cache(
                    refresh_tokens=['ok'],
                    jti_list=['keep'],
                    jti_meta={'keep': now + 5},
                ),
            ),
        )


if __name__ == '__main__':
    unittest.main()
