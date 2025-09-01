from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import HTTPException

from examples.auth.user_service import _cache_ttl
from examples.auth.user_service import _user_sites_cache
from examples.auth.user_service import get_user_and_sites
from examples.auth.user_service import get_user_sites_cached


class TestGetUserSitesCached(unittest.IsolatedAsyncioTestCase):
    """Unit tests for cache-assisted site lookups.

    The tests follow a simple Given/When/Then structure:
    - Given a mocked database session and a clean process-local cache
    - When invoking the helper functions under various conditions
    - Then the correct values are returned and the cache/database
      interactions are observed as expected
    """

    async def asyncSetUp(self) -> None:
        # Fresh DB mock and clear cache before each test.
        self.db: SimpleNamespace = SimpleNamespace(execute=AsyncMock())

        # Helper to mimic result of execute().scalar(): returns a namespace
        # exposing ``scalar() -> value`` where ``value`` is provided.
        self.scalar_result = lambda value: SimpleNamespace(
            scalar=lambda: value,
        )
        _user_sites_cache.clear()

    async def test_user_not_found_raises_404(self) -> None:
        """When the user cannot be found, raise ``HTTPException`` 404.

        Given: the database returns ``None`` for the user lookup
        When: calling ``get_user_sites_cached``
        Then: a 404 error is raised and the DB was awaited once
        """
        self.db.execute.return_value = self.scalar_result(None)

        with self.assertRaises(HTTPException) as ctx:
            await get_user_sites_cached('ghost', self.db)

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.detail, 'User not found')
        self.db.execute.assert_awaited()

    async def test_populates_cache_on_success(self) -> None:
        """First call queries DB, returns names, and populates the cache.

        Given: a user with two sites in the database
        When: time is frozen and the helper is called
        Then: the returned names and cached tuple match expectations
        """
        user = SimpleNamespace(
            sites=[SimpleNamespace(name='A'), SimpleNamespace(name='B')],
        )
        self.db.execute.return_value = self.scalar_result(user)

        base_time: float = 1_000_000.0
        with patch(
            'examples.auth.user_service.time.time',
            return_value=base_time,
        ):
            names: list[str] = await get_user_sites_cached('alice', self.db)

        self.assertEqual(names, ['A', 'B'])
        self.assertIn('alice', _user_sites_cache)
        cached_names, cached_time = _user_sites_cache['alice']
        self.assertEqual(cached_names, ['A', 'B'])
        self.assertEqual(cached_time, base_time)
        self.db.execute.assert_awaited_once()

    async def test_cache_hit_returns_without_db_call(self) -> None:
        """If cache is fresh, return cached names and skip DB access.

        Given: a fresh cache entry exists for the user
        When: current time is within the TTL window
        Then: the cached names are returned and DB is not called
        """
        base_time: float = 2_000_000.0
        _user_sites_cache['bob'] = (['X', 'Y'], base_time)

        # time within TTL window
        with patch(
                'examples.auth.user_service.time.time',
                return_value=base_time + (_cache_ttl - 1),
        ):
            names: list[str] = await get_user_sites_cached('bob', self.db)

        self.assertEqual(names, ['X', 'Y'])
        self.db.execute.assert_not_called()

    async def test_cache_expired_triggers_refresh(self) -> None:
        """If cache is expired, refresh via DB and update cache.

        Given: a stale cache entry exists
        When: the helper is called at the base time
        Then: it refreshes via DB and updates the cache timestamp and names
        """
        base_time: float = 3_000_000.0
        _user_sites_cache['carol'] = (['Old'], base_time - _cache_ttl - 10)
        new_user = SimpleNamespace(sites=[SimpleNamespace(name='New')])
        self.db.execute.return_value = self.scalar_result(new_user)

        with patch(
            'examples.auth.user_service.time.time',
            return_value=base_time,
        ):
            names: list[str] = await get_user_sites_cached('carol', self.db)

        self.assertEqual(names, ['New'])
        self.db.execute.assert_awaited_once()
        self.assertEqual(_user_sites_cache['carol'][0], ['New'])
        self.assertEqual(_user_sites_cache['carol'][1], base_time)

    async def test_ttl_boundary_is_still_valid(self) -> None:
        """Exactly at TTL-1 seconds, cache is valid; at TTL+1, it is not.

        This asserts the boundary conditions at the TTL horizon to ensure the
        inequality logic is correct and stable across refactors.
        """
        base_time: float = 4_000_000.0
        _user_sites_cache['dave'] = (['C1'], base_time)

        # Still valid at TTL-1
        with patch(
                'examples.auth.user_service.time.time',
                return_value=base_time + _cache_ttl - 1,
        ):
            names: list[str] = await get_user_sites_cached('dave', self.db)
        self.assertEqual(names, ['C1'])
        self.db.execute.assert_not_called()

        # Expired at TTL+1, force DB fetch
        new_user = SimpleNamespace(sites=[SimpleNamespace(name='C2')])
        self.db.execute.return_value = self.scalar_result(new_user)
        with patch(
            'examples.auth.user_service.time.time',
            return_value=base_time + _cache_ttl + 1,
        ):
            names_after: list[str] = await get_user_sites_cached(
                'dave', self.db,
            )
        self.assertEqual(names_after, ['C2'])
        self.db.execute.assert_awaited_once()

    async def test_get_user_and_sites_user_not_found(self) -> None:
        """``get_user_and_sites`` raises 401 when user is invalid.

        Given: the DB returns ``None`` for ``scalars().first()``
        When: calling the helper
        Then: an HTTP 401 error is raised with the expected detail
        """
        # execute().scalars().first() -> None
        scalars_result = SimpleNamespace(first=lambda: None)
        self.db.execute.return_value = SimpleNamespace(
            scalars=lambda: scalars_result,
        )

        with self.assertRaises(HTTPException) as ctx:
            await get_user_and_sites(self.db, 'nobody')

        self.assertEqual(ctx.exception.status_code, 401)
        self.assertEqual(ctx.exception.detail, 'Invalid user')

    async def test_get_user_and_sites_success(self) -> None:
        """
        On success, returns user, site names, and role as expected.
        """
        user = SimpleNamespace(
            username='eve',
            role='admin',
            sites=[SimpleNamespace(name='S1'), SimpleNamespace(name='S2')],
        )
        scalars_result = SimpleNamespace(first=lambda: user)
        self.db.execute.return_value = SimpleNamespace(
            scalars=lambda: scalars_result,
        )

        u, site_names, role = await get_user_and_sites(self.db, 'eve')

        self.assertIs(u, user)
        self.assertEqual(site_names, ['S1', 'S2'])
        self.assertEqual(role, 'admin')


if __name__ == '__main__':
    unittest.main()

'''
pytest \
    --cov=examples.auth.user_service \
    --cov-report=term-missing tests/examples/auth/user_service_test.py
'''
