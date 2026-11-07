from __future__ import annotations

import asyncio
from functools import partial
from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from examples.auth.models import Group
from examples.auth.models import Site
from examples.auth.models import SiteNotificationPreference
from examples.auth.models import Tenant
from examples.auth.models import User
from examples.auth.models import UserIdentity
from examples.auth.models import UserProfile
from examples.db_management.schemas.user import UserCreate
from examples.db_management.schemas.user import UserProfileBase
from examples.db_management.services import (
    keycloak_user_management_services as provider,
)
from examples.db_management.services import user_action_services as actions
from tests.database_fixtures_test import OrmDatabase


async def _preference_failure_rolls_back_and_fail_after_insert(
    testcase,
    /,
    **kwargs,
):
    await testcase.seed_preferences(**kwargs)
    raise RuntimeError('preference failure')


class AccountDatabaseTestCase(OrmDatabase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        async with self.sessions() as db:
            tenant = Tenant(name='test')
            group = Group(name='test', uniform_number='12345678')
            site = Site(name='test')
            db.add_all([tenant, group, site])
            await db.flush()
            self.me = User(
                username='operator',
                role='admin',
                group_id=group.id,
                tenant_id=tenant.id,
            )
            self.site_id = site.id
            db.add(self.me)
            await db.commit()
        self.payload = UserCreate(
            username='created',
            group_id=None,
            password='Strong-Password-123!',
            role='user',
            profile=UserProfileBase(
                family_name='Test',
                given_name='User',
                email='created@example.com',
            ),
        )
        self.provision = AsyncMock(return_value='remote-subject')
        self.compensate = AsyncMock()
        for name, replacement in [
            ('create_keycloak_user', self.provision),
            ('delete_keycloak_user', self.compensate),
            ('reset_keycloak_password', AsyncMock()),
            ('set_keycloak_user_enabled', AsyncMock()),
            (
                'list_site_ids_for_group',
                AsyncMock(
                    return_value=[
                        self.site_id,
                    ],
                ),
            ),
            ('seed_site_notification_preferences', self.seed_preferences),
        ]:
            patcher = patch.object(
                (
                    provider
                    if name
                    in (
                        'create_keycloak_user',
                        'delete_keycloak_user',
                        'reset_keycloak_password',
                        'set_keycloak_user_enabled',
                    )
                    else actions
                ),
                name,
                replacement,
            )
            patcher.start()
            self.addCleanup(patcher.stop)

    async def asyncTearDown(self) -> None:
        await super().asyncTearDown()

    async def seed_preferences(self, *, user_ids, site_ids, db) -> None:
        # Use portable inserts within the same transaction in SQLite.
        db.add_all(
            [
                SiteNotificationPreference(
                    user_id=user_id,
                    site_id=site_id,
                    is_enabled=True,
                )
                for user_id in user_ids
                for site_id in site_ids
            ],
        )
        await db.flush()

    async def assert_no_created_rows(self) -> None:
        async with self.sessions() as db:
            self.assertIsNone(
                await db.scalar(
                    select(User).where(User.username == 'created'),
                ),
            )
            for model in [
                UserIdentity,
                UserProfile,
                SiteNotificationPreference,
            ]:
                self.assertEqual(
                    list((await db.scalars(select(model))).all()),
                    [],
                )
        self.compensate.assert_awaited_once_with(
            'remote-subject',
        )


class UserCreationTransactionTest(AccountDatabaseTestCase):
    async def test_success_commits_account_and_preferences_together(
        self,
    ) -> None:
        async with self.sessions() as db:
            result = await actions.add_user(self.payload, db, self.me)
        async with self.sessions() as db:
            user = await db.scalar(
                select(User).where(User.username == 'created'),
            )
            self.assertEqual(user.id, result.id)
            self.assertIsNotNone(
                await db.get(
                    SiteNotificationPreference,
                    (user.id, self.site_id),
                ),
            )
            self.assertIsNotNone(
                await db.scalar(
                    select(UserIdentity).where(
                        UserIdentity.user_id == user.id,
                    ),
                ),
            )
        self.compensate.assert_not_awaited()

    async def test_preference_failure_rolls_back_and_compensates(self) -> None:
        fail_after_insert = partial(
            _preference_failure_rolls_back_and_fail_after_insert,
            self,
        )

        with patch.object(
            actions,
            'seed_site_notification_preferences',
            fail_after_insert,
        ):
            async with self.sessions() as db:
                with self.assertRaises(HTTPException) as raised:
                    await actions.add_user(self.payload, db, self.me)
        self.assertEqual(raised.exception.status_code, 500)
        await self.assert_no_created_rows()

    async def test_commit_failure_rolls_back_all_rows(self) -> None:
        async with self.sessions() as db:
            with patch.object(
                db,
                'commit',
                AsyncMock(
                    side_effect=IntegrityError('commit', {}, RuntimeError()),
                ),
            ):
                with self.assertRaises(HTTPException) as raised:
                    await actions.add_user(self.payload, db, self.me)
        self.assertEqual(raised.exception.status_code, 409)
        await self.assert_no_created_rows()

    async def test_cancellation_cleans_up_remote_and_local_identity(
        self,
    ) -> None:
        with patch.object(
            actions,
            'load_user_read',
            AsyncMock(side_effect=asyncio.CancelledError()),
        ):
            async with self.sessions() as db:
                with self.assertRaises(asyncio.CancelledError):
                    await actions.add_user(self.payload, db, self.me)
        await self.assert_no_created_rows()
