"""Exercise permission revocation across independent database sessions."""
from __future__ import annotations

import unittest
from uuid import uuid4

from sqlalchemy import delete
from sqlalchemy import event
from sqlalchemy import update
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine

from examples.auth.models import Group
from examples.auth.models import Site
from examples.auth.models import site_groups_table
from examples.auth.models import User
from examples.auth.models import user_sites_table
from examples.auth.user_service import get_effective_site_names
from examples.auth.user_service import load_user_access_context


class CurrentSitePermissionsTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.engine = create_async_engine('sqlite+aiosqlite:///:memory:')
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        self.queries: list[str] = []
        event.listen(
            self.engine.sync_engine,
            'before_cursor_execute',
            lambda conn,
            cursor,
            sql,
            parameters,
            context,
            many: self.queries.append(sql),
        )
        async with self.engine.begin() as conn:
            await conn.run_sync(
                lambda sync: User.metadata.create_all(
                    sync, tables=[
                        User.__table__, Site.__table__, Group.__table__,
                        user_sites_table, site_groups_table,
                    ],
                ),
            )
            await conn.execute(
                Group.__table__.insert().values(
                    id=1, name='group', uniform_number='12345678',
                ),
            )
            await conn.execute(
                User.__table__.insert().values(
                    id=1,
                    username='alice',
                    role='user',
                    group_id=1,
                    tenant_id=uuid4(),
                ),
            )
            await conn.execute(
                Site.__table__.insert().values(id=1, name='site-a'),
            )
            await conn.execute(
                user_sites_table.insert().values(user_id=1, site_id=1),
            )
            await conn.execute(
                site_groups_table.insert().values(group_id=1, site_id=1),
            )
        self.queries.clear()

    async def asyncTearDown(self) -> None:
        await self.engine.dispose()

    async def test_other_session_observes_membership_revocation_immediately(
            self,
    ) -> None:
        async with self.sessions() as worker_a:
            self.assertEqual(
                await get_effective_site_names('alice', worker_a), ['site-a'],
            )
        async with self.sessions() as administrator:
            await administrator.execute(
                delete(user_sites_table).where(
                    user_sites_table.c.user_id == 1,
                ),
            )
            await administrator.commit()
        async with self.sessions() as worker_b:
            self.assertEqual(
                await get_effective_site_names('alice', worker_b), [],
            )

    async def test_access_context_uses_two_queries_without_collection_joins(
        self,
    ) -> None:
        async with self.sessions() as worker:
            user, sites, role = await load_user_access_context(worker, 'alice')
            self.assertEqual((user.id, sites, role), (1, ['site-a'], 'user'))
        self.assertEqual(len(self.queries), 2)
        self.assertNotIn('JOIN', self.queries[0].upper())
        self.assertNotIn('group_features', ''.join(self.queries))
        self.assertNotIn('user_profiles', ''.join(self.queries))

    async def test_group_membership_revocation_also_removes_site(self) -> None:
        async with self.sessions() as administrator:
            await administrator.execute(delete(site_groups_table))
            await administrator.commit()
        async with self.sessions() as worker:
            self.assertEqual(
                await get_effective_site_names('alice', worker), [],
            )

    async def test_long_lived_session_refreshes_cached_user_group(
            self,
    ) -> None:
        async with self.sessions() as worker:
            user, sites, _ = await load_user_access_context(worker, 'alice')
            self.assertEqual(sites, ['site-a'])
            async with self.sessions() as administrator:
                await administrator.execute(
                    update(User).where(User.id == 1).values(group_id=None),
                )
                await administrator.commit()
            updated, sites, _ = await load_user_access_context(worker, 'alice')
            self.assertIs(updated, user)
            self.assertIsNone(updated.group_id)
            self.assertEqual(sites, [])
