from __future__ import annotations

from datetime import timedelta
from functools import partial
from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from examples.auth.models import AccountDeletionReceipt
from examples.auth.models import AccountOperation
from examples.auth.models import User
from examples.db_management.schemas.openwebui import OpenWebUIUserUpdate
from examples.db_management.schemas.user import UserDelete
from examples.db_management.services import account_lifecycle as lifecycle
from examples.db_management.services import openwebui_account_update as module
from examples.db_management.services import user_action_services as actions
from tests.examples.db_management.services import (
    user_creation_transaction_test,
)

AccountDatabaseTestCase = (
    user_creation_transaction_test.AccountDatabaseTestCase
)


async def _lost_commit_reply_reconciles_the_lost_reply(commit, /):
    await commit()
    raise SQLAlchemyError('commit response lost')


class OpenWebUISyncTest(AccountDatabaseTestCase):
    async def create(self):
        async with self.sessions() as db:
            return await actions.add_user(self.payload, db, self.me)

    async def operation(self):
        async with self.sessions() as db:
            return await db.scalar(
                select(AccountOperation).where(
                    AccountOperation.kind == 'openwebui',
                ),
            )

    async def test_lost_commit_reply_reconciles_the_committed_status(self):
        created = await self.create()
        lifecycle.provider.set_keycloak_user_enabled.reset_mock()
        async with self.sessions() as db:
            user = await db.get(User, created.id)
            commit = db.commit

            lost_reply = partial(
                _lost_commit_reply_reconciles_the_lost_reply,
                commit,
            )

            with patch.object(db, 'commit', lost_reply):
                with self.assertRaises(SQLAlchemyError):
                    await module.OpenWebUIAccountUpdate(
                        user,
                        OpenWebUIUserUpdate(role='pending'),
                        self.me,
                        db,
                    ).execute('remote-subject')
        lifecycle.provider.set_keycloak_user_enabled.assert_awaited_once_with(
            'remote-subject',
            enabled=False,
        )
        record = await self.operation()
        self.assertEqual(record.phase, 'completed')
        self.assertEqual(record.sync_parts, {'state': 'completed'})

    async def test_retries_keep_completed_parts_and_rebase_on_new_revision(
        self,
    ):
        created = await self.create()
        operator = User(
            username='ChangDar',
            role='super_admin',
            tenant_id=self.me.tenant_id,
        )
        with patch.object(
            lifecycle.provider,
            'synchronize_openwebui_roles',
            new=AsyncMock(),
        ) as roles, patch.object(
            lifecycle.provider,
            'update_keycloak_user',
            new=AsyncMock(side_effect=HTTPException(503)),
        ) as profile:
            async with self.sessions() as db:
                with self.assertRaises(HTTPException):
                    await module.OpenWebUIAccountUpdate(
                        await db.get(User, created.id),
                        OpenWebUIUserUpdate(
                            role='admin',
                            email='new@example.com',
                        ),
                        operator,
                        db,
                    ).execute('remote-subject')
            record = await self.operation()
            self.assertEqual(record.sync_parts['roles'], 'completed')
            self.assertEqual(record.sync_parts['profile'], 'pending')
            roles.assert_awaited_once_with('remote-subject', 'admin')
            profile.side_effect = None
            self.assertTrue(
                await lifecycle.AccountLifecycle(self.engine).process(
                    record.id,
                ),
            )
            self.assertEqual(roles.await_count, 1)
            self.assertEqual((await self.operation()).phase, 'completed')
        async with self.sessions() as db:
            row = await db.get(AccountOperation, record.id)
            row.phase = 'pending'
            row.sync_parts = {'state': 'pending', 'roles': 'completed'}
            user = await db.get(User, created.id)
            user.role = 'user'
            await db.commit()
        with patch.object(
            lifecycle.provider,
            'synchronize_openwebui_roles',
            new=AsyncMock(),
        ) as roles:
            self.assertTrue(
                await lifecycle.AccountLifecycle(self.engine).process(
                    record.id,
                ),
            )
        roles.assert_awaited_once_with('remote-subject', 'user')

    async def test_purging_jobs_preserves_delete_idempotency(
        self,
    ):
        created = await self.create()
        async with self.sessions() as db:
            await actions.remove_user(
                UserDelete(user_id=created.id),
                db,
                self.me,
            )
            record = await db.scalar(
                select(AccountOperation).where(
                    AccountOperation.kind == 'delete',
                ),
            )
            record.completed_at = lifecycle.now() - timedelta(days=31)
            await db.commit()
            identifier = record.id
        await lifecycle.AccountLifecycle(self.engine).purge_completed()
        async with self.sessions() as db:
            self.assertIsNone(await db.get(AccountOperation, identifier))
            self.assertIsNotNone(
                await db.get(AccountDeletionReceipt, created.id),
            )
            result = await actions.remove_user(
                UserDelete(user_id=created.id),
                db,
                self.me,
            )
            self.assertIn('message', result)
            receipt = await db.get(AccountDeletionReceipt, created.id)
            receipt.expires_at = lifecycle.now() - timedelta(seconds=1)
            await db.commit()
        await lifecycle.AccountLifecycle(self.engine).purge_completed()
        async with self.sessions() as db:
            with self.assertRaises(HTTPException) as caught:
                await actions.remove_user(
                    UserDelete(user_id=created.id),
                    db,
                    self.me,
                )
        self.assertEqual(caught.exception.status_code, 404)
