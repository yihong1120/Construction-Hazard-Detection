from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy.dialects import postgresql

from examples.auth.user_service import _cache_ttl
from examples.auth.user_service import _user_sites_cache
from examples.auth.user_service import get_cached_effective_site_names
from examples.auth.user_service import invalidate_effective_site_cache
from examples.auth.user_service import list_effective_site_names_for_user
from examples.auth.user_service import list_effective_sites_for_user
from examples.auth.user_service import load_user_access_context
from examples.auth.user_service import load_user_with_effective_sites


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
        """Prepare test fixtures."""
        self.db: SimpleNamespace = SimpleNamespace(execute=AsyncMock())

        # Supports: .scalar_one_or_none()
        #       and .unique().scalars().one_or_none()
        #       (new _load_user_by_username)
        self.scalar_result = lambda value: SimpleNamespace(
            scalar=lambda: value,
            scalar_one_or_none=lambda: value,
            unique=lambda: SimpleNamespace(
                scalars=lambda: SimpleNamespace(one_or_none=lambda: value),
            ),
        )
        # Supports: .scalars().all()
        #       and .scalars().unique().all()
        #       (new list_effective_sites_for_user)
        self.scalars_all_result = lambda values: SimpleNamespace(
            scalars=lambda: SimpleNamespace(
                all=lambda: values,
                unique=lambda: SimpleNamespace(all=lambda: values),
            ),
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
            await get_cached_effective_site_names('ghost', self.db)

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.detail, 'User not found')
        self.db.execute.assert_awaited()

    async def test_populates_cache_on_success(self) -> None:
        """First call queries DB, returns names, and populates the cache.

        Given: a user with two sites in the database
        When: time is frozen and the helper is called
        Then: the returned names and cached tuple match expectations
        """
        user = SimpleNamespace(id=1, role='user', group_id=7)
        self.db.execute.side_effect = [
            self.scalar_result(user),
            self.scalars_all_result([
                SimpleNamespace(name='A'),
                SimpleNamespace(name='B'),
            ]),
        ]

        base_time: float = 1_000_000.0
        with patch(
            'examples.auth.user_service.time.time',
            return_value=base_time,
        ):
            names: list[str] = await get_cached_effective_site_names(
                'alice', self.db,
            )

        self.assertEqual(names, ['A', 'B'])
        self.assertIn('alice', _user_sites_cache)
        cached_names, cached_time = _user_sites_cache['alice']
        self.assertEqual(cached_names, ['A', 'B'])
        self.assertEqual(cached_time, base_time)
        self.assertEqual(self.db.execute.await_count, 2)

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
            names: list[str] = await get_cached_effective_site_names(
                'bob', self.db,
            )

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
        new_user = SimpleNamespace(id=2, role='user', group_id=7)
        self.db.execute.side_effect = [
            self.scalar_result(new_user),
            self.scalars_all_result([SimpleNamespace(name='New')]),
        ]

        with patch(
            'examples.auth.user_service.time.time',
            return_value=base_time,
        ):
            names: list[str] = await get_cached_effective_site_names(
                'carol', self.db,
            )

        self.assertEqual(names, ['New'])
        self.assertEqual(self.db.execute.await_count, 2)
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
            names: list[str] = await get_cached_effective_site_names(
                'dave', self.db,
            )
        self.assertEqual(names, ['C1'])
        self.db.execute.assert_not_called()

        # Expired at TTL+1, force DB fetch
        new_user = SimpleNamespace(id=3, role='user', group_id=7)
        self.db.execute.side_effect = [
            self.scalar_result(new_user),
            self.scalars_all_result([SimpleNamespace(name='C2')]),
        ]
        with patch(
            'examples.auth.user_service.time.time',
            return_value=base_time + _cache_ttl + 1,
        ):
            names_after: list[str] = await get_cached_effective_site_names(
                'dave', self.db,
            )
        self.assertEqual(names_after, ['C2'])
        self.assertEqual(self.db.execute.await_count, 2)

    async def test_load_user_access_context_user_not_found(self) -> None:
        """``load_user_access_context`` raises 401 when user is invalid.

        Given: the DB returns ``None`` for ``scalars().first()``
        When: calling the helper
        Then: an HTTP 401 error is raised with the expected detail
        """
        # execute().scalars().first() -> None
        self.db.execute.return_value = self.scalar_result(None)

        with self.assertRaises(HTTPException) as ctx:
            await load_user_access_context(self.db, 'nobody')

        self.assertEqual(ctx.exception.status_code, 401)
        self.assertEqual(ctx.exception.detail, 'Invalid user')

    async def test_load_user_access_context_success(self) -> None:
        """
        On success, returns user, site names, and role as expected.
        """
        user = SimpleNamespace(
            id=5,
            username='eve',
            role='admin',
            group_id=3,
        )
        self.db.execute.side_effect = [
            self.scalar_result(user),
            self.scalars_all_result([
                SimpleNamespace(name='S1'),
                SimpleNamespace(name='S2'),
            ]),
        ]

        u, site_names, role = await load_user_access_context(self.db, 'eve')

        self.assertIs(u, user)
        self.assertEqual(site_names, ['S1', 'S2'])
        self.assertEqual(role, 'admin')

    async def test_load_user_with_effective_sites_super_admin_gets_all_sites(
        self,
    ) -> None:
        """Super admin should receive all sites without group filtering."""
        user = SimpleNamespace(id=9, username='root', role='super_admin')
        sites = [SimpleNamespace(name='A'), SimpleNamespace(name='B')]
        self.db.execute.side_effect = [
            self.scalar_result(user),
            self.scalars_all_result(sites),
        ]

        _, resolved_sites = await load_user_with_effective_sites(
            'root', self.db,
        )

        self.assertEqual(resolved_sites, sites)

    async def test_load_user_with_effective_sites_no_group_returns_empty(
        self,
    ) -> None:
        """Users without a group should have no effective site access."""
        user = SimpleNamespace(
            id=11, username='nogroup',
            role='user', group_id=None,
        )
        self.db.execute.return_value = self.scalar_result(user)

        _, resolved_sites = await load_user_with_effective_sites(
            'nogroup', self.db,
        )

        self.assertEqual(resolved_sites, [])
        self.db.execute.assert_awaited_once()

    async def test_list_effective_sites_for_user_filters_group_mismatch(
        self,
    ) -> None:
        """Direct site rows outside the user's group must not be effective."""
        user = SimpleNamespace(
            id=21, username='alice',
            role='admin', group_id=9,
        )
        self.db.execute.return_value = self.scalars_all_result([])

        sites = await list_effective_sites_for_user(
            cast(Any, user), self.db,
        )

        self.assertEqual(sites, [])

    async def test_effective_site_names_are_distinct_and_ordered_by_name(
        self,
    ) -> None:
        """Compile valid PostgreSQL SQL for the narrow site-name projection."""
        user = SimpleNamespace(
            id=21,
            username='alice',
            role='admin',
            group_id=9,
        )
        self.db.execute.return_value = self.scalars_all_result(['Alpha'])

        await list_effective_site_names_for_user(cast(Any, user), self.db)

        statement = self.db.execute.await_args.args[0]
        sql = str(statement.compile(dialect=postgresql.dialect()))
        self.assertIn('SELECT DISTINCT sites.name', sql)
        self.assertIn('ORDER BY sites.name', sql)

    async def test_load_user_with_effective_sites_success(self) -> None:
        """The loader should return the correct effective site payload."""
        user = SimpleNamespace(id=8, username='wrap', role='user', group_id=4)
        sites = [SimpleNamespace(name='S1')]
        self.db.execute.side_effect = [
            self.scalar_result(user),
            self.scalars_all_result(sites),
        ]

        loaded_user, resolved_sites = await load_user_with_effective_sites(
            'wrap',
            self.db,
        )

        self.assertIs(loaded_user, user)
        self.assertEqual(resolved_sites, sites)

    async def test_get_cached_effective_site_names_success(self) -> None:
        """The cache helper should resolve and cache site names."""
        user = SimpleNamespace(
            id=31, username='cache',
            role='user', group_id=4,
        )
        self.db.execute.side_effect = [
            self.scalar_result(user),
            self.scalars_all_result([SimpleNamespace(name='SX')]),
        ]

        site_names = await get_cached_effective_site_names('cache', self.db)

        self.assertEqual(site_names, ['SX'])
        self.assertIn('cache', _user_sites_cache)

    async def test_load_user_access_context_role_and_names(self) -> None:
        """The access-context helper should preserve role and names."""
        user = SimpleNamespace(id=41, username='ctx', role='admin', group_id=2)
        self.db.execute.side_effect = [
            self.scalar_result(user),
            self.scalars_all_result([SimpleNamespace(name='SA')]),
        ]

        loaded_user, site_names, role = await load_user_access_context(
            self.db,
            'ctx',
        )

        self.assertIs(loaded_user, user)
        self.assertEqual(site_names, ['SA'])
        self.assertEqual(role, 'admin')

    async def test_invalidate_effective_site_cache_specific_user(self) -> None:
        """Invalidating one user should keep unrelated cache entries."""
        _user_sites_cache['alice'] = (['A'], 1.0)
        _user_sites_cache['bob'] = (['B'], 1.0)

        invalidate_effective_site_cache(['alice'])

        self.assertNotIn('alice', _user_sites_cache)
        self.assertIn('bob', _user_sites_cache)

    async def test_invalidate_effective_site_cache_all(self) -> None:
        """The renamed invalidation helper should clear the full cache."""
        _user_sites_cache['alice'] = (['A'], 1.0)
        _user_sites_cache['bob'] = (['B'], 1.0)

        invalidate_effective_site_cache()

        self.assertEqual(_user_sites_cache, {})


if __name__ == '__main__':
    unittest.main()

'''
pytest \
    --cov=examples.auth.user_service \
    --cov-report=term-missing tests/examples/auth/user_service_test.py
'''
