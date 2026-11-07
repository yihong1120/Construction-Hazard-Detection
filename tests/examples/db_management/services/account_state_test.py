from __future__ import annotations

import asyncio
from functools import partial
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm.exc import StaleDataError

from examples.auth.models import AccountOperation
from examples.auth.models import User
from examples.db_management.schemas.user import SetUserStatus
from examples.db_management.schemas.user import UserProfileUpdate
from examples.db_management.services import account_lifecycle as module
from examples.db_management.services import user_action_services as actions
from tests.examples.db_management.services import (
    user_creation_transaction_test,
)

AccountDatabaseTestCase = (
    user_creation_transaction_test.AccountDatabaseTestCase
)


async def _late_activation_cannot_overwrite_completed_suspend(
    created,
    testcase,
    /,
):
    async with testcase.sessions() as db:
        return await actions.update_user_status(
            SetUserStatus(user_id=created.id, status='suspended'),
            db,
            testcase.me,
        )


async def _ambiguous_commit_reconciles_from_the_lost_reply(commit, /):
    await commit()
    raise SQLAlchemyError('lost commit response')


class _LateActivationCannotOverwriteCompletedSuspensionCallbacks:
    entered: Any
    release: Any
    remote: Any

    async def _call_enable(self, subject, *, enabled):
        if enabled:
            self.entered.set()
            await self.release.wait()
        self.remote['enabled'] = enabled


class AccountStateTest(AccountDatabaseTestCase):
    async def create(self):
        async with self.sessions() as db:
            return await actions.add_user(self.payload, db, self.me)

    async def test_late_activation_cannot_overwrite_completed_suspension(self):
        _callbacks = (
            _LateActivationCannotOverwriteCompletedSuspensionCallbacks()
        )
        created = await self.create()
        async with self.sessions() as db:
            operation = await db.scalar(select(AccountOperation))
            operation.phase = 'adopted'
            await db.commit()
            operation_id = operation.id
        _callbacks.entered, _callbacks.release = (
            asyncio.Event(),
            asyncio.Event(),
        )
        _callbacks.remote = {'enabled': True}

        enable = _callbacks._call_enable

        with patch.object(
            module.provider,
            'set_keycloak_user_enabled',
            enable,
        ):
            recovery = asyncio.create_task(
                module.AccountLifecycle(self.engine).process(operation_id),
            )
            await asyncio.wait_for(_callbacks.entered.wait(), 2)

            suspend = partial(
                _late_activation_cannot_overwrite_completed_suspend,
                created,
                self,
            )

            task = asyncio.create_task(suspend())
            await asyncio.sleep(0.05)
            self.assertFalse(task.done())
            _callbacks.release.set()
            self.assertTrue(await recovery)
            await asyncio.wait_for(task, 3)
        async with self.sessions() as db:
            current = await db.get(User, created.id)
            self.assertEqual(current.status, 'suspended')
            self.assertGreater(current.state_revision, 1)
            record = await db.scalar(
                select(AccountOperation).where(
                    AccountOperation.kind == 'status',
                ),
            )
            self.assertEqual(record.target_revision, current.state_revision)
            self.assertEqual(record.phase, 'completed')
        self.assertFalse(_callbacks.remote['enabled'])

    async def test_failed_provider_update_is_durable_and_retries_latest_state(
        self,
    ):
        created = await self.create()
        module.provider.set_keycloak_user_enabled.side_effect = HTTPException(
            503,
        )
        async with self.sessions() as db:
            with self.assertRaises(HTTPException):
                await actions.update_user_status(
                    SetUserStatus(user_id=created.id, status='suspended'),
                    db,
                    self.me,
                )
        async with self.sessions() as db:
            self.assertEqual(
                (await db.get(User, created.id)).status,
                'suspended',
            )
            record = await db.scalar(
                select(AccountOperation).where(
                    AccountOperation.kind == 'status',
                ),
            )
            self.assertEqual(record.phase, 'pending')
        module.provider.set_keycloak_user_enabled.side_effect = None
        self.assertTrue(
            await module.AccountLifecycle(self.engine).process(record.id),
        )
        module.provider.set_keycloak_user_enabled.assert_awaited_with(
            'remote-subject',
            enabled=False,
        )

    async def test_ambiguous_commit_reconciles_from_the_committed_row(self):
        created = await self.create()
        module.provider.set_keycloak_user_enabled.reset_mock()
        async with self.sessions() as db:
            commit = db.commit

            lost_reply = partial(
                _ambiguous_commit_reconciles_from_the_lost_reply,
                commit,
            )

            with patch.object(db, 'commit', lost_reply):
                with self.assertRaises(SQLAlchemyError):
                    await actions.update_user_status(
                        SetUserStatus(user_id=created.id, status='suspended'),
                        db,
                        self.me,
                    )
        module.provider.set_keycloak_user_enabled.assert_awaited_once_with(
            'remote-subject',
            enabled=False,
        )

    async def test_profile_and_rename_reconcile_only_committed_values(self):
        created = await self.create()
        with patch.object(
            module.provider,
            'update_keycloak_user',
            new=AsyncMock(),
        ) as remote:
            async with self.sessions() as db:
                await actions.update_profile(
                    UserProfileUpdate(
                        user_id=created.id,
                        email='new@example.com',
                    ),
                    db,
                    self.me,
                )
                user = await db.get(User, created.id)
                await actions.rename_account(user, 'renamed', db, self.me)
        self.assertEqual(remote.await_count, 2)
        self.assertEqual(
            remote.await_args_list[0].kwargs['email'],
            'new@example.com',
        )
        self.assertEqual(
            remote.await_args_list[1].kwargs,
            {'username': 'renamed'},
        )

    async def test_stale_user_object_cannot_overwrite_a_newer_state(self):
        created = await self.create()
        async with self.sessions() as one, self.sessions() as two:
            old = await one.get(User, created.id)
            current = await two.get(User, created.id)
            current.status = 'suspended'
            await two.commit()
            old.role = 'admin'
            with self.assertRaises(StaleDataError):
                await one.commit()

    async def test_background_recovery_defers_while_an_update_owns_the_account(
        self,
    ):
        created = await self.create()
        async with self.sessions() as db:
            record = await db.scalar(select(AccountOperation))
            record.phase = 'adopted'
            await db.commit()
            operation_id = record.id
        async with self.sessions() as db:
            from examples.db_management.services import account_state

            await account_state.lock_account(
                db,
                await db.get(User, created.id),
            )
            async with asyncio.timeout(1):
                self.assertFalse(
                    await module.AccountLifecycle(self.engine).process(
                        operation_id,
                    ),
                )
        self.assertTrue(
            await module.AccountLifecycle(self.engine).process(operation_id),
        )

    async def test_duplicate_rename_preserves_both_accounts(
        self,
    ):
        created = await self.create()
        with patch.object(
            module.provider,
            'update_keycloak_user',
            new=AsyncMock(),
        ) as remote:
            async with self.sessions() as db:
                with self.assertRaises(HTTPException) as caught:
                    await actions.rename_account(
                        await db.get(User, created.id),
                        'operator',
                        db,
                        self.me,
                    )
        self.assertEqual(caught.exception.status_code, 400)
        remote.assert_awaited_once_with('remote-subject', username='created')
