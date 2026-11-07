from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from .user_creation_transaction_test import (
    AccountDatabaseTestCase,
)
from examples.auth.models import User
from examples.db_management.schemas.openwebui import OpenWebUIUserCreate
from examples.db_management.schemas.openwebui import OpenWebUIUserUpdate
from examples.db_management.services import account_lifecycle as lifecycle
from examples.db_management.services import (
    openwebui_account_update as mutations,
)
from examples.db_management.services import openwebui_services as bridge
from examples.db_management.services import user_action_services as actions
from examples.db_management.services.user_services import get_user_by_id


class OpenWebUIAccountTransactionTest(AccountDatabaseTestCase):
    async def test_create_role_failure_rolls_back_all_local_rows(self) -> None:
        with patch.object(
            bridge,
            'assign_keycloak_client_role',
            AsyncMock(side_effect=HTTPException(503, 'provider offline')),
        ):
            async with self.sessions() as db:
                with self.assertRaises(HTTPException):
                    await bridge.create_user_from_openwebui(
                        OpenWebUIUserCreate(**self.payload.model_dump()),
                        db,
                        self.me,
                    )
        await self.assert_no_created_rows()

    async def test_pending_creation_returns_suspended_account_after_one_commit(
        self,
    ) -> None:
        with (
            patch.object(bridge, 'assign_keycloak_client_role', AsyncMock()),
            patch.object(
                bridge,
                'set_keycloak_user_enabled',
                AsyncMock(),
            ) as disable,
        ):
            async with self.sessions() as db:
                result = await bridge.create_user_from_openwebui(
                    OpenWebUIUserCreate(
                        **self.payload.model_dump(),
                        openwebui_role='pending',
                    ),
                    db,
                    self.me,
                )
        self.assertEqual(result['user'].status, 'suspended')
        disable.assert_awaited_once_with('remote-subject', enabled=False)
        self.compensate.assert_not_awaited()

    async def update_setup(self, db):
        created = await actions.add_user(self.payload, db, self.me)
        return await get_user_by_id(created.id, db)

    def remote_patches(self):
        self.remote_update = AsyncMock()
        self.snapshot = AsyncMock(
            return_value={
                'enabled': True,
                'email': 'created@example.com',
                'firstName': 'User',
                'lastName': 'Test',
            },
        )
        self.role_snapshot = AsyncMock(return_value={'openwebui-user'})
        self.assign = AsyncMock()
        self.remove = AsyncMock()
        self.restore = AsyncMock()
        self.password = AsyncMock()
        self.enabled = AsyncMock()
        self.roles = AsyncMock()
        for owner, name, value in (
            (lifecycle.provider, 'update_keycloak_user', self.remote_update),
            (lifecycle.provider, 'set_keycloak_user_enabled', self.enabled),
            (lifecycle.provider, 'synchronize_openwebui_roles', self.roles),
            (mutations, 'reset_keycloak_password', self.password),
        ):
            operation = patch.object(owner, name, value)
            operation.start()
            self.addCleanup(operation.stop)

    async def test_invalid_group_fails_before_any_remote_or_local_mutation(
        self,
    ) -> None:
        self.remote_patches()
        async with self.sessions() as db:
            target = await self.update_setup(db)
            with self.assertRaises(HTTPException) as raised:
                await mutations.OpenWebUIAccountUpdate(
                    target,
                    OpenWebUIUserUpdate(role='pending', group_id=999),
                    self.me,
                    db,
                ).execute('subject')
            self.assertEqual(raised.exception.status_code, 403)
            self.assertEqual(target.status, 'active')
        self.snapshot.assert_not_awaited()
        self.remote_update.assert_not_awaited()

    async def test_role_failure_restores_remote_and_local_profile(
        self,
    ) -> None:
        self.remote_patches()
        self.roles.side_effect = HTTPException(503, 'role unavailable')
        async with self.sessions() as db:
            target = await self.update_setup(db)
            operator = User(
                username='ChangDar',
                role='super_admin',
                tenant_id=self.me.tenant_id,
            )
            with self.assertRaises(HTTPException):
                await mutations.OpenWebUIAccountUpdate(
                    target,
                    OpenWebUIUserUpdate(
                        role='admin',
                        email='changed@example.com',
                        password='new-password-safe',
                    ),
                    operator,
                    db,
                ).execute('subject')
        async with self.sessions() as db:
            user = await db.scalar(
                select(User).where(User.username == 'created'),
            )
            loaded = await get_user_by_id(user.id, db)
            self.assertEqual(loaded.role, 'admin')
            self.assertEqual(loaded.profile.email, 'changed@example.com')
        self.roles.assert_awaited_once_with('subject', 'admin')
        self.password.assert_awaited_once()

    async def test_commit_failure_compensates_status_and_profile(self) -> None:
        self.remote_patches()
        async with self.sessions() as db:
            target = await self.update_setup(db)
            with patch.object(
                db,
                'commit',
                AsyncMock(
                    side_effect=IntegrityError('commit', {}, RuntimeError()),
                ),
            ):
                with self.assertRaises(IntegrityError):
                    await mutations.OpenWebUIAccountUpdate(
                        target,
                        OpenWebUIUserUpdate(role='pending', name='Changed'),
                        self.me,
                        db,
                    ).execute('subject')
        self.assertEqual(self.remote_update.await_count, 1)
        self.assertEqual(
            self.remote_update.await_args.kwargs['email'],
            'created@example.com',
        )
        self.assertTrue(self.enabled.await_args.kwargs['enabled'])

    async def test_password_runs_after_validation_and_local_flush(
        self,
    ) -> None:
        self.remote_patches()
        async with self.sessions() as db:
            target = await self.update_setup(db)
            with patch.object(
                db,
                'flush',
                AsyncMock(
                    side_effect=IntegrityError('flush', {}, RuntimeError()),
                ),
            ):
                with self.assertRaises(IntegrityError):
                    await mutations.OpenWebUIAccountUpdate(
                        target,
                        OpenWebUIUserUpdate(
                            password='new-password-safe',
                            name='Changed',
                        ),
                        self.me,
                        db,
                    ).execute('subject')
        self.password.assert_not_awaited()
        self.assertEqual(
            self.remote_update.await_args.kwargs['email'],
            'created@example.com',
        )

    async def test_failed_compensation_reports_reconciliation(self) -> None:
        self.remote_patches()
        async with self.sessions() as db:
            target = await self.update_setup(db)
            self.enabled.side_effect = HTTPException(503, 'offline')
            with self.assertRaises(HTTPException) as raised:
                await mutations.OpenWebUIAccountUpdate(
                    target,
                    OpenWebUIUserUpdate(role='pending'),
                    self.me,
                    db,
                ).execute('subject')
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(
            raised.exception.detail,
            {'code': 'account_reconciliation_required'},
        )
        self.assertGreaterEqual(self.enabled.await_count, 1)

    async def test_commit_failure_after_password_reports_reconciliation(
        self,
    ) -> None:
        self.remote_patches()
        async with self.sessions() as db:
            target = await self.update_setup(db)
            with patch.object(
                db,
                'commit',
                AsyncMock(side_effect=RuntimeError('offline')),
            ):
                with self.assertRaises(HTTPException) as raised:
                    await mutations.OpenWebUIAccountUpdate(
                        target,
                        OpenWebUIUserUpdate(password='new-password-safe'),
                        self.me,
                        db,
                    ).execute('subject')
        self.assertEqual(
            raised.exception.detail,
            'account_reconciliation_required',
        )
        self.password.assert_awaited_once()
