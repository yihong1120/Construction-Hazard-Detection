from __future__ import annotations

import types
import unittest
from unittest.mock import AsyncMock
from unittest.mock import patch

from examples.auth import token_cleanup as tc


class TestPruneUserCache(unittest.IsolatedAsyncioTestCase):
    """Behavioural tests for prune_user_cache covering key branches."""

    def setUp(self) -> None:
        """Provide shared fakes and defaults for tests."""
        # Minimal Redis substitute, not used directly by the function.
        self.rds: object = object()
        # Fixed settings for jwt decode path.
        self.settings = types.SimpleNamespace(
            authjwt_secret_key='s',
            ALGORITHM='HS256',
        )

    async def test_no_cache_returns_none(self) -> None:
        """When user cache is missing, return None and do not write back."""
        with (
            patch.object(tc, 'settings', self.settings),
            patch.object(
                tc,
                'get_user_data',
                new=AsyncMock(return_value=None),
            ) as mock_get,
            patch.object(tc, 'set_user_data', new=AsyncMock()) as mock_set,
        ):
            out = await tc.prune_user_cache(self.rds, 'alice')
        self.assertIsNone(out)
        mock_get.assert_awaited_once_with(self.rds, 'alice')
        mock_set.assert_not_awaited()

    async def test_prune_refresh_tokens_mixed_validity(self) -> None:
        """Expired/invalid refresh tokens are removed; valid ones remain."""
        now = 1_000
        cache = {
            'refresh_tokens': ['valid1', 'expired1', 'invalid1'],
        }
        # Configure jwt.decode to succeed for valid1, fail for others.

        def decode_side_effect(
            tok: str,
            key: str,
            algorithms: list[str],
        ) -> dict[str, object]:
            if tok == 'valid1':
                self.assertEqual(key, 's')
                self.assertEqual(algorithms, ['HS256'])
                return {'ok': True}
            if tok == 'expired1':
                raise tc.jwt.ExpiredSignatureError('expired')
            raise tc.jwt.InvalidTokenError('bad')

        with (
            patch.object(tc, 'settings', self.settings),
            patch.object(tc.time, 'time', return_value=now),
            patch.object(
                tc,
                'get_user_data',
                new=AsyncMock(return_value=cache.copy()),
            ) as mock_get,
            patch.object(tc, 'set_user_data', new=AsyncMock()) as mock_set,
            patch.object(
                tc.jwt,
                'decode',
                side_effect=decode_side_effect,
            ) as mock_decode,
        ):
            out = await tc.prune_user_cache(self.rds, 'bob')

        self.assertEqual(out, {'refresh_tokens': ['valid1']})
        mock_get.assert_awaited_once_with(self.rds, 'bob')
        mock_set.assert_awaited_once_with(
            self.rds,
            'bob',
            {'refresh_tokens': ['valid1']},
        )
        # Ensure decode was attempted for each token
        self.assertEqual(
            [c.args[0] for c in mock_decode.call_args_list],
            ['valid1', 'expired1', 'invalid1'],
        )

    async def test_prune_jti_list_and_meta(self) -> None:
        """JTI list pruned by expiry; stale meta entries removed and
        preserved strictly.
        """
        now = 2_000
        cache = {
            'jti_list': ['a', 'b', 'c'],
            # a unexpired, b expired, 'stale' not present in list (stale)
            'jti_meta': {'a': now + 10, 'b': now - 1, 'stale': now + 10},
        }
        with (
            patch.object(tc, 'settings', self.settings),
            patch.object(tc.time, 'time', return_value=now),
            patch.object(
                tc,
                'get_user_data',
                new=AsyncMock(return_value=cache.copy()),
            ) as mock_get,
            patch.object(tc, 'set_user_data', new=AsyncMock()) as mock_set,
            # not used in this path
            patch.object(tc.jwt, 'decode', return_value={}),
        ):
            out = await tc.prune_user_cache(self.rds, 'carol')

        self.assertEqual(
            out,
            {
                # c kept because missing meta counts as 0 -> keep
                'jti_list': ['a', 'c'],
                'jti_meta': {'a': now + 10},
            },
        )
        mock_get.assert_awaited_once_with(self.rds, 'carol')
        mock_set.assert_awaited_once_with(
            self.rds,
            'carol',
            {'jti_list': ['a', 'c'], 'jti_meta': {'a': now + 10}},
        )

    async def test_no_change_does_not_write(self) -> None:
        """If nothing changes, avoid unnecessary writes to the cache."""
        now = 3_000
        cache = {
            'refresh_tokens': ['still_valid'],
            'jti_list': ['x'],
            'jti_meta': {},  # empty -> pruning of jti is skipped
        }

        with (
            patch.object(tc, 'settings', self.settings),
            patch.object(tc.time, 'time', return_value=now),
            patch.object(
                tc,
                'get_user_data',
                new=AsyncMock(return_value=cache.copy()),
            ) as mock_get,
            patch.object(tc, 'set_user_data', new=AsyncMock()) as mock_set,
            patch.object(
                tc.jwt,
                'decode',
                return_value={'ok': True},
            ) as mock_decode,
        ):
            out = await tc.prune_user_cache(self.rds, 'dave')

        # No changes: same cache returned and no write
        self.assertEqual(out, cache)
        mock_get.assert_awaited_once_with(self.rds, 'dave')
        mock_set.assert_not_awaited()
        mock_decode.assert_called_once_with(
            'still_valid',
            's',
            algorithms=['HS256'],
        )

    async def test_combined_refresh_and_jti_changes(self) -> None:
        """Prune both refresh tokens and JTIs then persist the updated
        cache.
        """
        now = 4_000
        cache = {
            'refresh_tokens': ['ok', 'bad'],
            'jti_list': ['keep', 'drop'],
            'jti_meta': {'keep': now + 5, 'drop': now - 5},
        }

        def decode_side_effect(
            tok: str,
            key: str,
            algorithms: list[str],
        ) -> dict[str, object]:
            if tok == 'ok':
                return {}
            raise tc.jwt.InvalidTokenError('bad')

        with (
            patch.object(tc, 'settings', self.settings),
            patch.object(tc.time, 'time', return_value=now),
            patch.object(
                tc,
                'get_user_data',
                new=AsyncMock(return_value=cache.copy()),
            ) as mock_get,
            patch.object(tc, 'set_user_data', new=AsyncMock()) as mock_set,
            patch.object(
                tc.jwt,
                'decode',
                side_effect=decode_side_effect,
            ),
        ):
            out = await tc.prune_user_cache(self.rds, 'erin')

        self.assertEqual(
            out,
            {
                'refresh_tokens': ['ok'],
                'jti_list': ['keep'],
                'jti_meta': {'keep': now + 5},
            },
        )
        mock_get.assert_awaited_once_with(self.rds, 'erin')
        mock_set.assert_awaited_once_with(
            self.rds,
            'erin',
            {
                'refresh_tokens': ['ok'],
                'jti_list': ['keep'],
                'jti_meta': {'keep': now + 5},
            },
        )

    async def test_refresh_tokens_not_list_is_ignored(self) -> None:
        """Non-list refresh_tokens should be ignored and not cause writes."""
        now = 5_000
        cache = {
            'refresh_tokens': 'oops-not-a-list',
        }
        with (
            patch.object(tc, 'settings', self.settings),
            patch.object(tc.time, 'time', return_value=now),
            patch.object(
                tc, 'get_user_data', new=AsyncMock(return_value=cache.copy()),
            ) as mock_get,
            patch.object(tc, 'set_user_data', new=AsyncMock()) as mock_set,
            patch.object(tc.jwt, 'decode', return_value={}),
        ):
            out = await tc.prune_user_cache(self.rds, 'fred')

        # No change expected since tokens are not a list
        self.assertEqual(out, cache)
        mock_get.assert_awaited_once_with(self.rds, 'fred')
        mock_set.assert_not_awaited()

    async def test_jti_meta_not_dict_skips_processing(self) -> None:
        """Non-dict jti_meta should be treated as empty and not processed."""
        now = 6_000
        cache = {
            'jti_list': ['a'],
            'jti_meta': 'not-a-dict',
        }
        with (
            patch.object(tc, 'settings', self.settings),
            patch.object(tc.time, 'time', return_value=now),
            patch.object(
                tc, 'get_user_data', new=AsyncMock(return_value=cache.copy()),
            ) as mock_get,
            patch.object(tc, 'set_user_data', new=AsyncMock()) as mock_set,
        ):
            out = await tc.prune_user_cache(self.rds, 'gina')

        # No changes since jti_meta is invalid type (treated as empty),
        # and no refresh tokens provided to change state
        self.assertEqual(out, cache)
        mock_get.assert_awaited_once_with(self.rds, 'gina')
        mock_set.assert_not_awaited()

    async def test_jti_list_not_list_prunes_all_meta(self) -> None:
        """
        Non-list jti_list should cause all jti_meta entries to be dropped.
        """
        now = 7_000
        cache = {
            'jti_list': 'oops',
            'jti_meta': {'keep': now + 10},
        }
        with (
            patch.object(tc, 'settings', self.settings),
            patch.object(tc.time, 'time', return_value=now),
            patch.object(
                tc, 'get_user_data', new=AsyncMock(return_value=cache.copy()),
            ) as mock_get,
            patch.object(tc, 'set_user_data', new=AsyncMock()) as mock_set,
        ):
            out = await tc.prune_user_cache(self.rds, 'hank')

        # jti_list remains the original invalid value as per current logic,
        # but jti_meta is pruned to empty and a write occurs
        self.assertEqual(out, {'jti_list': 'oops', 'jti_meta': {}})
        mock_get.assert_awaited_once_with(self.rds, 'hank')
        mock_set.assert_awaited_once_with(
            self.rds,
            'hank',
            {'jti_list': 'oops', 'jti_meta': {}},
        )


if __name__ == '__main__':
    unittest.main()

'''
pytest \
    --cov=examples.auth.token_cleanup \
    --cov-report=term-missing tests/examples/auth/token_cleanup_test.py
'''
