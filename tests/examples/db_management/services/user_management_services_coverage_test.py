from __future__ import annotations

import unittest
from datetime import datetime
from datetime import timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import patch
from uuid import UUID

from fastapi import HTTPException
from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.deployment_context import DeploymentBinding
from examples.auth.models import User
from examples.db_management.schemas.user import UserPage
from examples.db_management.schemas.user import UserRead
from examples.db_management.schemas.user import UserSignup
from examples.db_management.services import (
    user_management_services as services,
)


_TENANT_ID = UUID('11111111-1111-1111-1111-111111111111')
_OTHER_TENANT_ID = UUID('22222222-2222-2222-2222-222222222222')


def _request() -> Request:
    """Create a canonical request recognised by signup deployment resolution.

    Returns:
        HTTP request fixture with a secure public deployment origin.
    """
    return Request(
        {
            'type': 'http',
            'asgi': {'version': '3.0'},
            'http_version': '1.1',
            'method': 'POST',
            'scheme': 'https',
            'path': '/signup',
            'raw_path': b'/signup',
            'query_string': b'',
            'headers': [(b'host', b'api.example.test')],
            'client': ('192.0.2.1', 50000),
            'server': ('api.example.test', 443),
        },
    )


def _user_read() -> UserRead:
    """Return a minimal serialisable user for a paginated list assertion."""
    now = datetime(2026, 8, 24, tzinfo=timezone.utc)
    return UserRead(
        id=1,
        username='alice',
        role='user',
        status='active',
        group_id=3,
        group=None,
        profile=None,
        created_at=now,
        updated_at=now,
    )


class TestUserManagementCoverage(unittest.IsolatedAsyncioTestCase):
    """Verify remaining operator scope and deployment-resolution behaviour."""

    async def test_list_for_group_operator_forwards_group_and_tenant_scope(
        self,
    ) -> None:
        """A group administrator receives only its tenant-scoped keyset page.

        The downstream query receives both authorisation constraints.
        """
        operator = cast(
            User,
            SimpleNamespace(group_id=3, tenant_id=_TENANT_ID, role='admin'),
        )
        with (
            patch.object(services, 'is_super_admin', return_value=False),
            patch.object(services, 'ensure_admin_with_group'),
            patch.object(
                services,
                'list_users',
                new=AsyncMock(return_value=([_user_read()], 2)),
            ) as list_users,
        ):
            page = await services.list_users_for_operator(
                operator,
                cast(AsyncSession, SimpleNamespace()),
                cursor=1,
                page_size=25,
            )

        self.assertIsInstance(page, UserPage)
        self.assertEqual(page.next_cursor, 2)
        list_users_args = list_users.await_args
        assert list_users_args is not None
        self.assertEqual(list_users_args.kwargs['group_id'], 3)
        self.assertEqual(list_users_args.kwargs['tenant_id'], _TENANT_ID)

    async def test_list_all_users_reads_until_the_last_cursor(self) -> None:
        """The array endpoint combines bounded pages without losing rows."""
        operator = cast(User, SimpleNamespace())
        first = UserPage(items=[_user_read()], next_cursor=1)
        second = UserPage(items=[], next_cursor=None)

        with patch.object(
            services,
            'list_users_for_operator',
            new=AsyncMock(side_effect=[first, second]),
        ) as list_users_for_operator:
            result = await services.list_all_users_for_operator(
                operator,
                cast(AsyncSession, SimpleNamespace()),
            )

        self.assertEqual(result, first.items)
        self.assertEqual(list_users_for_operator.await_count, 2)
        self.assertEqual(
            list_users_for_operator.await_args_list[0].kwargs,
            {'cursor': None, 'page_size': 100},
        )
        self.assertEqual(
            list_users_for_operator.await_args_list[1].kwargs,
            {'cursor': 1, 'page_size': 100},
        )

    async def test_signup_requires_resolved_deployment_when_request_is_real(
        self,
    ) -> None:
        """A public signup cannot proceed if its deployment origin is unknown.

        The returned conflict asks the client to load a valid Registry entry.
        """
        payload = UserSignup(
            username='alice',
            password='strong-password',
            profile={
                'family_name': 'Example',
                'given_name': 'Alice',
                'email': 'alice@example.com',
            },
        )
        with patch.object(
            services,
            'resolve_request_deployment',
            new=AsyncMock(return_value=None),
        ):
            with self.assertRaises(HTTPException) as missing_deployment:
                await services.register_signup_user(
                    payload,
                    _request(),
                    cast(AsyncSession, SimpleNamespace()),
                    cast(services.Redis, SimpleNamespace()),
                )
        self.assertEqual(missing_deployment.exception.status_code, 409)

    async def test_signup_passes_resolved_tenant_to_account_creation(
        self,
    ) -> None:
        """Deployment-bound signup writes the tenant identifier.

        Tenant scope is supplied directly to account creation.
        """
        payload = UserSignup(
            username='alice',
            password='strong-password',
            profile={
                'family_name': 'Example',
                'given_name': 'Alice',
                'email': 'alice@example.com',
            },
        )
        deployment = DeploymentBinding(
            deployment_id=UUID('33333333-3333-3333-3333-333333333333'),
            tenant_id=_TENANT_ID,
            api_base_url='https://api.example.com/hazard/api',
            config_revision=1,
        )
        new_user = SimpleNamespace(id=5)
        user_read = _user_read()
        with (
            patch.object(
                services, 'validate_signup_consents',
                new=AsyncMock(),
            ),
            patch.object(
                services,
                'create_user',
                new=AsyncMock(return_value=new_user),
            ) as create_user,
            patch.object(services, 'record_user_consent', new=AsyncMock()),
            patch.object(
                services,
                'send_signup_verification_email',
                new=AsyncMock(),
            ),
            patch.object(
                services,
                'load_user_read',
                new=AsyncMock(return_value=user_read),
            ),
        ):
            result = await services.register_signup_user(
                payload,
                _request(),
                cast(AsyncSession, SimpleNamespace()),
                cast(services.Redis, SimpleNamespace()),
                deployment,
            )

        self.assertEqual(result, user_read)
        create_user_args = create_user.await_args
        assert create_user_args is not None
        self.assertEqual(
            create_user_args.kwargs['tenant_id'],
            _TENANT_ID,
        )

    def test_management_scope_rejects_cross_tenant_and_cross_group_users(
        self,
    ) -> None:
        """Group administrators cannot cross tenant or group boundaries."""
        operator = cast(
            User,
            SimpleNamespace(
                group_id=3,
                tenant_id=_TENANT_ID,
                role='admin',
            ),
        )
        with (
            patch.object(services, 'is_super_admin', return_value=False),
            patch.object(services, 'ensure_admin_with_group'),
        ):
            with self.assertRaises(HTTPException) as tenant_error:
                services.ensure_user_management_scope(
                    cast(
                        User,
                        SimpleNamespace(
                            group_id=3,
                            tenant_id=_OTHER_TENANT_ID,
                        ),
                    ),
                    operator,
                )
            with self.assertRaises(HTTPException) as group_error:
                services.ensure_user_management_scope(
                    cast(
                        User,
                        SimpleNamespace(group_id=4, tenant_id=_TENANT_ID),
                    ),
                    operator,
                )

        self.assertEqual(tenant_error.exception.status_code, 403)
        self.assertEqual(group_error.exception.status_code, 403)
