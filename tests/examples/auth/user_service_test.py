from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import AsyncMock

from fastapi import HTTPException
from sqlalchemy.dialects import postgresql

from examples.auth.user_service import get_effective_site_names
from examples.auth.user_service import list_effective_site_names_for_user
from examples.auth.user_service import list_effective_sites_for_user
from examples.auth.user_service import load_user_access_context
from examples.auth.user_service import load_user_with_effective_sites


class TestCurrentUserSites(unittest.IsolatedAsyncioTestCase):
    """Unit tests for current permission lookups.

    The tests follow a simple Given/When/Then structure:
    - Given a mocked database session
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

    async def test_user_not_found_raises_404(self) -> None:
        """When the user cannot be found, raise ``HTTPException`` 404.

        Given: the database returns ``None`` for the user lookup
        Then: a 404 error is raised and the DB was awaited once
        """
        self.db.execute.return_value = self.scalar_result(None)

        with self.assertRaises(HTTPException) as ctx:
            await get_effective_site_names('ghost', self.db)

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.detail, 'User not found')
        self.db.execute.assert_awaited()

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
        """On success, returns user, site names, and role as expected."""
        user = SimpleNamespace(
            id=5,
            username='eve',
            role='admin',
            group_id=3,
        )
        self.db.execute.side_effect = [
            self.scalar_result(user),
            self.scalars_all_result(
                [
                    SimpleNamespace(name='S1'),
                    SimpleNamespace(name='S2'),
                ],
            ),
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
            'root',
            self.db,
        )

        self.assertEqual(resolved_sites, sites)

    async def test_load_user_with_effective_sites_no_group_returns_empty(
        self,
    ) -> None:
        """Users without a group should have no effective site access."""
        user = SimpleNamespace(
            id=11,
            username='nogroup',
            role='user',
            group_id=None,
        )
        self.db.execute.return_value = self.scalar_result(user)

        _, resolved_sites = await load_user_with_effective_sites(
            'nogroup',
            self.db,
        )

        self.assertEqual(resolved_sites, [])
        self.db.execute.assert_awaited_once()

    async def test_list_effective_sites_for_user_filters_group_mismatch(
        self,
    ) -> None:
        """Direct site rows outside the user's group must not be effective."""
        user = SimpleNamespace(
            id=21,
            username='alice',
            role='admin',
            group_id=9,
        )
        self.db.execute.return_value = self.scalars_all_result([])

        sites = await list_effective_sites_for_user(
            cast(Any, user),
            self.db,
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

    async def test_site_name_projection_handles_super_and_ungrouped_users(
        self,
    ) -> None:
        """Super administrators list all sites; ungrouped users list none."""
        super_admin = SimpleNamespace(id=1, role='super_admin', group_id=None)
        self.db.execute.return_value = self.scalars_all_result(
            [SimpleNamespace(name='Alpha')],
        )

        names = await list_effective_site_names_for_user(
            cast(Any, super_admin),
            self.db,
        )
        self.assertEqual(names, ['Alpha'])

        ungrouped = SimpleNamespace(id=2, role='admin', group_id=None)
        self.db.execute.reset_mock()
        self.assertEqual(
            await list_effective_site_names_for_user(
                cast(Any, ungrouped),
                self.db,
            ),
            [],
        )
        self.db.execute.assert_not_awaited()

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

    async def test_get_effective_site_names_success(self) -> None:
        """The cache helper should resolve and cache site names."""
        user = SimpleNamespace(
            id=31,
            username='cache',
            role='user',
            group_id=4,
        )
        self.db.execute.side_effect = [
            self.scalar_result(user),
            self.scalars_all_result([SimpleNamespace(name='SX')]),
        ]

        site_names = await get_effective_site_names('cache', self.db)

        self.assertEqual(site_names, ['SX'])

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


if __name__ == '__main__':
    unittest.main()
