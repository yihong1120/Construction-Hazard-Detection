from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import patch

import httpx
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from .user_creation_transaction_test import (
    AccountDatabaseTestCase,
)
from examples.auth.models import UserIdentity
from examples.db_management.services import (
    native_social_links as service,
)
from examples.db_management.services.native_social_proof import (
    VerifiedSocialIdentity,
)


def response(status: int, payload=None) -> httpx.Response:
    return httpx.Response(
        status,
        json=payload,
        request=httpx.Request('GET', 'https://identity.test/admin'),
    )


class NativeIdentityTransactionTest(AccountDatabaseTestCase):
    def command(self):
        return service.NativeIdentityLink(
            self.me.id,
            'keycloak-subject',
            VerifiedSocialIdentity('google', 'verified-provider-subject'),
        )

    async def test_local_mirror_and_remote_link_commit_once(self) -> None:
        remote = AsyncMock(side_effect=[response(200, []), response(204)])
        async with self.sessions() as db:
            with patch.object(
                service,
                '_keycloak_admin_request',
                remote,
            ), patch.object(
                db,
                'commit',
                AsyncMock(wraps=db.commit),
            ) as commit:
                self.assertEqual(await self.command().execute(db), 'linked')
                commit.assert_awaited_once()
        async with self.sessions() as db:
            identity = await db.scalar(
                select(UserIdentity).where(UserIdentity.provider == 'google'),
            )
            self.assertEqual(
                identity.provider_user_id,
                'verified-provider-subject',
            )

    async def test_flush_failure_never_mutates_remote_identity(self) -> None:
        remote = AsyncMock()
        async with self.sessions() as db:
            with patch.object(
                service,
                '_keycloak_admin_request',
                remote,
            ), patch.object(
                db,
                'flush',
                AsyncMock(
                    side_effect=IntegrityError(
                        'flush',
                        {},
                        RuntimeError('offline'),
                    ),
                ),
            ):
                with self.assertRaises(HTTPException):
                    await self.command().execute(db)
        remote.assert_not_awaited()

    async def test_commit_failure_preserves_link_and_retry_repairs_mirror(
        self,
    ) -> None:
        remote = AsyncMock(side_effect=[response(200, []), response(204)])
        async with self.sessions() as db:
            with patch.object(
                service,
                '_keycloak_admin_request',
                remote,
            ), patch.object(
                db,
                'commit',
                AsyncMock(side_effect=RuntimeError('offline')),
            ):
                with self.assertRaises(HTTPException) as raised:
                    await self.command().execute(db)
                self.assertEqual(
                    raised.exception.detail,
                    'account_reconciliation_required',
                )
        self.assertFalse(
            any(call.args[0] == 'DELETE' for call in remote.await_args_list),
        )
        async with self.sessions() as db:
            self.assertIsNone(
                await db.scalar(
                    select(UserIdentity).where(
                        UserIdentity.provider == 'google',
                    ),
                ),
            )
            linked = [
                {
                    'identityProvider': 'google',
                    'userId': 'verified-provider-subject',
                },
            ]
            with patch.object(
                service,
                '_keycloak_admin_request',
                AsyncMock(return_value=response(200, linked)),
            ):
                self.assertEqual(
                    await self.command().execute(db),
                    'already_linked',
                )
        async with self.sessions() as db:
            self.assertIsNotNone(
                await db.scalar(
                    select(UserIdentity).where(
                        UserIdentity.provider == 'google',
                    ),
                ),
            )

    async def test_uncertain_remote_timeout_reports_reconciliation(
        self,
    ) -> None:
        remote = AsyncMock(
            side_effect=[
                response(
                    200,
                    [],
                ),
                httpx.ReadTimeout('uncertain'),
            ],
        )
        async with self.sessions() as db:
            with patch.object(service, '_keycloak_admin_request', remote):
                with self.assertRaises(HTTPException) as raised:
                    await self.command().execute(db)
        self.assertEqual(
            raised.exception.detail,
            'account_reconciliation_required',
        )

    async def test_existing_link_is_never_removed_after_local_failure(
        self,
    ) -> None:
        linked = [
            {
                'identityProvider': 'google',
                'userId': 'verified-provider-subject',
            },
        ]
        remote = AsyncMock(return_value=response(200, linked))
        async with self.sessions() as db:
            with patch.object(
                service,
                '_keycloak_admin_request',
                remote,
            ), patch.object(
                db,
                'commit',
                AsyncMock(side_effect=RuntimeError('offline')),
            ):
                with self.assertRaises(RuntimeError):
                    await self.command().execute(db)
        self.assertEqual(remote.await_count, 1)
        call = remote.await_args
        assert call is not None
        self.assertEqual(call.args[0], 'GET')
