from __future__ import annotations

import unittest
from typing import ClassVar
from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import HTTPException

from examples.auth.user_service import _cache_ttl
from examples.auth.user_service import _user_sites_cache
from examples.auth.user_service import get_user_sites_cached


class FakeExecuteResult:
    """A small helper to mimic SQLAlchemy execute().scalar() result.

    It exposes a scalar() method that returns the provided value.
    """

    def __init__(self, scalar_result) -> None:
        self._scalar_result = scalar_result

    def scalar(self):  # noqa: D401 - tiny and clear
        """Return the preconfigured scalar result."""
        return self._scalar_result


class FakeAsyncDB:
    """A fake async DB session exposing an AsyncMock execute()."""

    def __init__(self) -> None:
        self.execute: AsyncMock = AsyncMock()


class MockSite:
    """Simple site stub with a name attribute."""

    def __init__(self, name: str) -> None:
        self.name = name


class MockUser:
    """Simple user stub with a sites list."""

    def __init__(self, sites: list[MockSite]) -> None:
        self.sites = sites


class TestGetUserSitesCached(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for get_user_sites_cached and its in-memory cache behaviour.
    """

    db: ClassVar[FakeAsyncDB]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.db = FakeAsyncDB()

    async def asyncSetUp(self) -> None:
        # Reset mocks and clear cache before each test
        self.db.execute.reset_mock()
        _user_sites_cache.clear()

    async def test_user_not_found_raises_404(self) -> None:
        """When the user cannot be found, raise HTTPException 404."""
        self.db.execute.return_value = FakeExecuteResult(None)

        with self.assertRaises(HTTPException) as ctx:
            await get_user_sites_cached('ghost', self.db)

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.detail, 'User not found')
        self.db.execute.assert_awaited()

    async def test_populates_cache_on_success(self) -> None:
        """First call queries DB, returns names, and populates the cache."""
        user = MockUser([MockSite('A'), MockSite('B')])
        self.db.execute.return_value = FakeExecuteResult(user)

        base_time = 1_000_000.0
        with patch(
            'examples.auth.user_service.time.time',
            return_value=base_time,
        ):
            names = await get_user_sites_cached('alice', self.db)

        self.assertEqual(names, ['A', 'B'])
        self.assertIn('alice', _user_sites_cache)
        cached_names, cached_time = _user_sites_cache['alice']
        self.assertEqual(cached_names, ['A', 'B'])
        self.assertEqual(cached_time, base_time)
        self.db.execute.assert_awaited_once()

    async def test_cache_hit_returns_without_db_call(self) -> None:
        """If cache is fresh, return cached names and skip DB access."""
        base_time = 2_000_000.0
        _user_sites_cache['bob'] = (['X', 'Y'], base_time)

        # time within TTL window
        with patch(
                'examples.auth.user_service.time.time',
                return_value=base_time + (_cache_ttl - 1),
        ):
            names = await get_user_sites_cached('bob', self.db)

        self.assertEqual(names, ['X', 'Y'])
        self.db.execute.assert_not_called()

    async def test_cache_expired_triggers_refresh(self) -> None:
        """If cache is expired, refresh via DB and update cache."""
        base_time = 3_000_000.0
        _user_sites_cache['carol'] = (['Old'], base_time - _cache_ttl - 10)

        new_user = MockUser([MockSite('New')])
        self.db.execute.return_value = FakeExecuteResult(new_user)

        with patch(
            'examples.auth.user_service.time.time',
            return_value=base_time,
        ):
            names = await get_user_sites_cached('carol', self.db)

        self.assertEqual(names, ['New'])
        self.db.execute.assert_awaited_once()
        self.assertEqual(_user_sites_cache['carol'][0], ['New'])
        self.assertEqual(_user_sites_cache['carol'][1], base_time)

    async def test_ttl_boundary_is_still_valid(self) -> None:
        """Exactly at TTL-1 seconds, cache is valid; at TTL+1, it is not."""
        base_time = 4_000_000.0
        _user_sites_cache['dave'] = (['C1'], base_time)

        # Still valid at TTL-1
        with patch(
                'examples.auth.user_service.time.time',
                return_value=base_time + _cache_ttl - 1,
        ):
            names = await get_user_sites_cached('dave', self.db)
        self.assertEqual(names, ['C1'])
        self.db.execute.assert_not_called()

        # Expired at TTL+1, force DB fetch
        new_user = MockUser([MockSite('C2')])
        self.db.execute.return_value = FakeExecuteResult(new_user)
        with patch(
                'examples.auth.user_service.time.time',
                return_value=base_time + _cache_ttl + 1,
        ):
            names_after = await get_user_sites_cached('dave', self.db)
        self.assertEqual(names_after, ['C2'])
        self.db.execute.assert_awaited_once()


'''
pytest \
    --cov=examples.auth.user_service \
    --cov-report=term-missing tests/examples/auth/user_service_test.py
'''
