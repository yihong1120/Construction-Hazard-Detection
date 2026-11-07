from __future__ import annotations

import asyncio
from datetime import timedelta
from functools import partial
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

from examples.auth.models import AccountOperation
from examples.auth.models import User
from examples.auth.operation_database import close_operation_database
from examples.db_management.schemas.user import UserDelete
from examples.db_management.services import account_lifecycle as module
from examples.db_management.services import user_action_services as actions
from scripts import account_operations
from tests.examples.db_management.services import (
    user_creation_transaction_test,
)

AccountDatabaseTestCase = (
    user_creation_transaction_test.AccountDatabaseTestCase
)


async def _cancelled_creation_waits_for_subject_remote(
    entered,
    release,
    /,
    *args,
):
    entered.set()
    await release.wait()
    return 'remote-subject'


class _CancelledCommitPreservesAnAdoptedAccountCallbacks:
    entered: Any
    original: Any
    release: Any

    async def _call_commit(self):
        await self.original()
        self.entered.set()
        await self.release.wait()


class AccountLifecycleTest(AccountDatabaseTestCase):
    async def operation(self):
        async with self.sessions() as db:
            return await db.scalar(
                select(AccountOperation).order_by(
                    AccountOperation.created_at.desc(),
                ),
            )

    async def create(self):
        async with self.sessions() as db:
            return await actions.add_user(self.payload, db, self.me)

    async def test_password_cancellation_deletes_disabled_remote_account(self):
        with patch.object(
            module.provider,
            'reset_keycloak_password',
            AsyncMock(side_effect=asyncio.CancelledError()),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await self.create()
        await self.assert_no_created_rows()
        self.assertEqual((await self.operation()).phase, 'completed')

    async def test_cancelled_creation_waits_for_subject_then_compensates(
        self,
    ):
        entered, release = asyncio.Event(), asyncio.Event()

        remote = partial(
            _cancelled_creation_waits_for_subject_remote,
            entered,
            release,
        )

        with patch.object(module.provider, 'create_keycloak_user', remote):
            task = asyncio.create_task(self.create())
            await entered.wait()
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        await self.assert_no_created_rows()
        self.assertEqual((await self.operation()).subject, 'remote-subject')

    async def test_lost_create_reply_is_uncertain_and_never_guesses_identity(
        self,
    ):
        self.provision.side_effect = HTTPException(503, 'unavailable')
        with self.assertRaises(HTTPException):
            await self.create()
        record = await self.operation()
        self.assertEqual(record.phase, 'uncertain')
        self.assertIsNone(record.subject)
        self.compensate.assert_not_awaited()
        lifecycle = module.AccountLifecycle(self.engine)
        await lifecycle.sweep()
        self.provision.assert_awaited_once()

    async def test_activation_retry_preserves_adopted_local_identity(self):
        module.provider.set_keycloak_user_enabled.side_effect = RuntimeError(
            'offline',
        )
        with self.assertRaises(HTTPException) as caught:
            await self.create()
        self.assertEqual(caught.exception.status_code, 503)
        record = await self.operation()
        self.assertEqual(record.phase, 'adopted')
        async with self.sessions() as db:
            self.assertIsNotNone(await db.get(User, record.user_id))
        self.compensate.assert_not_awaited()
        module.provider.set_keycloak_user_enabled.side_effect = None
        self.assertTrue(
            await module.AccountLifecycle(self.engine).process(record.id),
        )
        self.assertEqual((await self.operation()).phase, 'completed')

    async def test_delete_failure_remains_suspended_and_retries_idempotently(
        self,
    ):
        created = await self.create()
        self.compensate.side_effect = RuntimeError('offline')
        async with self.sessions() as db:
            with self.assertRaises(HTTPException) as caught:
                await actions.remove_user(
                    UserDelete(user_id=created.id),
                    db,
                    self.me,
                )
        self.assertEqual(caught.exception.status_code, 503)
        async with self.sessions() as db:
            user = await db.get(User, created.id)
            self.assertTrue(user.deletion_pending)
            self.assertEqual(user.status, 'suspended')
            with self.assertRaises(HTTPException):
                module.account_mutable(user)
        self.compensate.side_effect = None
        for _ in range(2):
            async with self.sessions() as db:
                result = await actions.remove_user(
                    UserDelete(user_id=created.id),
                    db,
                    self.me,
                )
            self.assertIn('message', result)
        async with self.sessions() as db:
            self.assertIsNone(await db.get(User, created.id))
        self.assertEqual(self.compensate.await_count, 2)

    async def test_recovery_skips_an_active_owner(self):
        lifecycle = module.AccountLifecycle(self.engine)
        async with lifecycle.provisioning(
            self.payload,
            self.me.tenant_id,
        ) as remote:
            self.assertFalse(await lifecycle.process(remote.id))
            await lifecycle.sweep()
            self.compensate.assert_not_awaited()
        self.compensate.assert_awaited_once()

    async def test_one_connection_request_pool_cannot_starve_journal(self):
        small = create_async_engine(
            self.engine.url,
            pool_size=1,
            max_overflow=0,
            pool_timeout=0.2,
        )
        try:
            async with small.begin():
                lifecycle = module.AccountLifecycle(small)
                async with asyncio.timeout(3):
                    async with lifecycle.provisioning(
                        self.payload,
                        self.me.tenant_id,
                    ):
                        pass
        finally:
            await close_operation_database(small)
            await small.dispose()
        self.compensate.assert_awaited_once()

    async def test_operator_lists_and_resumes_uncertain_operation(self):
        self.provision.side_effect = HTTPException(503)
        with self.assertRaises(HTTPException):
            await self.create()
        record = await self.operation()
        lifecycle = module.AccountLifecycle(self.engine)
        with patch.object(
            account_operations,
            'AccountLifecycle',
            return_value=lifecycle,
        ), patch('builtins.print') as output:
            self.assertEqual(
                await account_operations.run(
                    SimpleNamespace(command='list', limit=10),
                ),
                0,
            )
            self.assertIn(record.id, output.call_args.args[0])
            self.assertEqual(
                await account_operations.run(
                    SimpleNamespace(
                        command='retry',
                        operation=record.id,
                        subject='remote-subject',
                    ),
                ),
                0,
            )
        self.compensate.assert_awaited_once_with('remote-subject')

    async def test_background_retries_due_operations_and_stops_cleanly(self):
        lifecycle = module.AccountLifecycle(self.engine)
        self.compensate.side_effect = RuntimeError('offline')
        with self.assertRaises(HTTPException):
            async with lifecycle.provisioning(self.payload, self.me.tenant_id):
                pass
        record = await self.operation()
        async with self.sessions() as db:
            row = await db.get(AccountOperation, record.id)
            row.next_attempt = module.now() - timedelta(seconds=1)
            await db.commit()
        self.compensate.side_effect = None
        await lifecycle.sweep()
        self.assertEqual((await self.operation()).phase, 'completed')
        async with lifecycle.lifespan():
            self.assertFalse(lifecycle.stop.is_set())
        self.assertTrue(lifecycle.stop.is_set())

    async def test_cancelled_commit_preserves_an_adopted_account(self):
        _callbacks = _CancelledCommitPreservesAnAdoptedAccountCallbacks()
        _callbacks.entered, _callbacks.release = (
            asyncio.Event(),
            asyncio.Event(),
        )
        async with self.sessions() as db:
            _callbacks.original = db.commit

            commit = _callbacks._call_commit

            with patch.object(db, 'commit', commit):
                task = asyncio.create_task(
                    actions.add_user(self.payload, db, self.me),
                )
                await asyncio.wait_for(_callbacks.entered.wait(), 2)
                task.cancel()
                _callbacks.release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        record = await self.operation()
        self.assertEqual(record.phase, 'completed')
        async with self.sessions() as db:
            self.assertIsNotNone(await db.get(User, record.user_id))
        self.compensate.assert_not_awaited()
        module.provider.set_keycloak_user_enabled.assert_awaited_once_with(
            'remote-subject',
            enabled=True,
        )

    async def test_local_cleanup_failure_keeps_a_retryable_deletion(self):
        created = await self.create()
        await self.db.execute(
            'CREATE TABLE deletion_blocker(user_id integer REFERENCES '
            'users(id))',
        )
        await self.db.execute(
            'INSERT INTO deletion_blocker VALUES($1)',
            created.id,
        )
        async with self.sessions() as db:
            with self.assertRaises(HTTPException):
                await actions.remove_user(
                    UserDelete(user_id=created.id),
                    db,
                    self.me,
                )
        record = await self.operation()
        self.assertEqual(record.kind, 'delete')
        self.assertNotEqual(record.phase, 'completed')
        async with self.sessions() as db:
            self.assertTrue((await db.get(User, created.id)).deletion_pending)
        await self.db.execute('DELETE FROM deletion_blocker')
        self.assertTrue(
            await module.AccountLifecycle(self.engine).process(record.id),
        )
        self.assertEqual(self.compensate.await_count, 2)
