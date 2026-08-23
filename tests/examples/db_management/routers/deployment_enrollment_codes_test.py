from __future__ import annotations

import asyncio
import unittest
from datetime import timedelta
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch
from uuid import UUID

from fastapi import FastAPI
from fastapi.testclient import TestClient

from examples.auth.database import get_db
from examples.auth.models import DeploymentEnrollmentCode
from examples.auth.models import User
from examples.auth.models import utc_now
from examples.db_management.deps import require_tenant_deployment_administrator
from examples.db_management.deps import TenantDeploymentAdministrator
from examples.db_management.routers import deployment_enrollment_codes as router_module
from examples.db_management.routers.deployment_enrollment_codes import router
from examples.db_management.services import (
    deployment_enrollment_code_services as services,
)
from examples.db_management.services.deployment_enrollment_code_services import (
    ManagedEnrollmentCode,
)
from examples.db_management.services.deployment_enrollment_code_services import (
    ManagedEnrollmentCodeItem,
)
"""Security contracts for authenticated device-invitation management."""


_TENANT_ID = UUID('11111111-1111-1111-1111-111111111111')
_DEPLOYMENT_ID = UUID('22222222-2222-2222-2222-222222222222')
_CODE_ID = UUID('aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa')
_CODE_ID_TEXT = str(_CODE_ID)


def _administrator() -> TenantDeploymentAdministrator:
    """Perform administrator.

    Returns:
        The callable result.
    """
    return TenantDeploymentAdministrator(
        user=cast(User, SimpleNamespace(id=7, username='tenant-admin')),
        tenant_id=_TENANT_ID,
        deployment_id=_DEPLOYMENT_ID,
    )


class TestDeploymentEnrollmentCodeRouter(unittest.TestCase):
    """The management API must expose raw code material exactly once."""

    def setUp(self) -> None:
        """Perform setUp.
        """
        self.app = FastAPI()
        self.app.include_router(router)
        self.client = TestClient(self.app)
        self.db = MagicMock()

        async def override_get_db() -> Any:  # type: ignore[no-untyped-def]
            """Perform override get db.

            Returns:
                The callable result.
            """
            yield self.db

        self.app.dependency_overrides[get_db] = override_get_db
        self.app.dependency_overrides[
            require_tenant_deployment_administrator
        ] = _administrator

    def tearDown(self) -> None:
        """Perform tearDown.
        """
        self.client.close()

    def test_create_returns_the_raw_code_once_and_rejects_scope_fields(self) -> None:
        """Test create returns the raw code once and rejects scope fields.
        """
        expires_at = utc_now() + timedelta(minutes=30)
        with patch.object(
            router_module,
            'create_managed_enrollment_code',
            AsyncMock(
                return_value=ManagedEnrollmentCode(
                    id=_CODE_ID,
                    enrollment_code='a' * 43,
                    expires_at=expires_at,
                ),
            ),
        ) as create:
            response = self.client.post(
                '/deployment-enrollment-codes',
                json={'expires_in_minutes': 30},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['content-type'], 'application/json')
        self.assertEqual(response.headers['cache-control'], 'no-store')
        self.assertEqual(
            set(response.json()),
            {'id', 'enrollment_code', 'expires_at'},
        )
        self.assertEqual(response.json()['id'], _CODE_ID_TEXT)
        self.assertEqual(response.json()['enrollment_code'], 'a' * 43)
        create.assert_awaited_once()
        create_args = create.await_args
        assert create_args is not None
        self.assertEqual(
            create_args.kwargs['administrator'].tenant_id,
            _TENANT_ID,
        )
        self.assertEqual(
            create_args.kwargs['administrator'].deployment_id,
            _DEPLOYMENT_ID,
        )

        for forbidden in (
            'tenant_id',
            'deployment_id',
            'api_base_url',
            'registry_url',
            'private_key',
        ):
            with self.subTest(field=forbidden):
                denied = self.client.post(
                    '/deployment-enrollment-codes',
                    json={'expires_in_minutes': 30, forbidden: 'client-value'},
                )
                self.assertEqual(denied.status_code, 422)

    def test_create_enforces_one_minute_to_24_hour_ttl(self) -> None:
        """Test create enforces one minute to 24 hour ttl.
        """
        for value in (0, 1441):
            with self.subTest(value=value):
                response = self.client.post(
                    '/deployment-enrollment-codes',
                    json={'expires_in_minutes': value},
                )
                self.assertEqual(response.status_code, 422)

    def test_list_never_exposes_code_or_verifier(self) -> None:
        """Test list never exposes code or verifier.
        """
        expires_at = utc_now() + timedelta(minutes=30)
        with patch.object(
            router_module,
            'list_managed_enrollment_codes',
            AsyncMock(
                return_value=[
                    ManagedEnrollmentCodeItem(
                        id=_CODE_ID,
                        expires_at=expires_at,
                        status='active',
                    ),
                ],
            ),
        ):
            response = self.client.get('/deployment-enrollment-codes')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['cache-control'], 'no-store')
        self.assertEqual(
            response.json(),
            {
                'items': [{
                    'id': _CODE_ID_TEXT,
                    'expires_at': expires_at.isoformat().replace('+00:00', 'Z'),
                    'status': 'active',
                }],
            },
        )
        self.assertNotIn('enrollment_code', response.text)
        self.assertNotIn('verifier', response.text)
        self.assertNotIn('hash', response.text)

    def test_delete_is_idempotent_and_requires_canonical_lowercase_uuid(self) -> None:
        """Test delete is idempotent and requires canonical lowercase uuid.
        """
        with patch.object(
            router_module,
            'revoke_managed_enrollment_code',
            AsyncMock(),
        ) as revoke:
            response = self.client.delete(
                f'/deployment-enrollment-codes/{_CODE_ID_TEXT}',
            )

        self.assertEqual(response.status_code, 204)
        self.assertEqual(response.content, b'')
        self.assertEqual(response.headers['cache-control'], 'no-store')
        revoke.assert_awaited_once()
        revoke_args = revoke.await_args
        assert revoke_args is not None
        self.assertEqual(revoke_args.kwargs['public_id'], _CODE_ID)

        invalid = self.client.delete(
            f'/deployment-enrollment-codes/{_CODE_ID_TEXT.upper()}',
        )
        self.assertEqual(invalid.status_code, 422)


class TestDeploymentEnrollmentCodeServices(unittest.TestCase):
    """Exercise lifecycle status, tenant scope, and non-secret audit writes."""

    def test_status_precedence_is_safe_and_complete(self) -> None:
        """Test status precedence is safe and complete.
        """
        now = utc_now()
        base = {
            'expires_at': now + timedelta(minutes=1),
            'redeemed_at': None,
            'revoked_at': None,
        }
        self.assertEqual(
            services.enrollment_code_status(
                cast(DeploymentEnrollmentCode, SimpleNamespace(**base)),
                now=now,
            ),
            'active',
        )
        self.assertEqual(
            services.enrollment_code_status(
                cast(
                    DeploymentEnrollmentCode,
                    SimpleNamespace(**{**base, 'redeemed_at': now}),
                ),
                now=now,
            ),
            'redeemed',
        )
        self.assertEqual(
            services.enrollment_code_status(
                cast(
                    DeploymentEnrollmentCode,
                    SimpleNamespace(**{**base, 'expires_at': now}),
                ),
                now=now,
            ),
            'expired',
        )
        self.assertEqual(
            services.enrollment_code_status(
                cast(
                    DeploymentEnrollmentCode,
                    SimpleNamespace(
                        **{**base, 'redeemed_at': now, 'revoked_at': now},
                    ),
                ),
                now=now,
            ),
            'revoked',
        )

    def test_create_scopes_provisioning_and_writes_audit_without_verifier(self) -> None:
        """Test create scopes provisioning and writes audit without verifier.
        """
        enrollment = SimpleNamespace(
            id=10,
            public_id=_CODE_ID,
            expires_at=utc_now() + timedelta(minutes=30),
        )
        provisioned = SimpleNamespace(
            raw_code='b' * 43,
            enrollment=enrollment,
        )
        db = MagicMock()
        db.flush = AsyncMock()
        db.commit = AsyncMock()
        db.rollback = AsyncMock()
        administrator = _administrator()

        with patch.object(
            services,
            'provision_enrollment_code',
            AsyncMock(return_value=provisioned),
        ) as provision:
            result = asyncio.run(
                services.create_managed_enrollment_code(
                    db,
                    administrator=administrator,
                    expires_in_minutes=30,
                    pepper='p' * 32,
                ),
            )

        self.assertEqual(result.id, _CODE_ID)
        self.assertEqual(result.enrollment_code, 'b' * 43)
        self.assertNotEqual(result.enrollment_code, 'p' * 32)
        provision_args = provision.await_args
        assert provision_args is not None
        self.assertEqual(provision_args.kwargs['tenant_id'], _TENANT_ID)
        self.assertEqual(provision_args.kwargs['deployment_id'], _DEPLOYMENT_ID)
        db.flush.assert_awaited_once()
        db.commit.assert_awaited_once()
        audit = db.add.call_args.args[0]
        self.assertEqual(audit.action, 'created')
        self.assertEqual(audit.actor_user_id, 7)
        self.assertFalse(hasattr(audit, 'code_verifier_hash'))
        self.assertFalse(hasattr(audit, 'enrollment_code'))

    def test_revoke_locks_the_tenant_scoped_row_and_audits_once(self) -> None:
        """Test revoke locks the tenant scoped row and audits once.
        """
        enrollment = SimpleNamespace(id=10, redeemed_at=None, revoked_at=None)
        transaction = MagicMock()
        transaction.__aenter__ = AsyncMock(return_value=transaction)
        transaction.__aexit__ = AsyncMock(return_value=False)
        db = MagicMock()
        db.begin.return_value = transaction
        db.scalar = AsyncMock(return_value=enrollment)

        asyncio.run(
            services.revoke_managed_enrollment_code(
                db,
                administrator=_administrator(),
                public_id=_CODE_ID,
            ),
        )

        self.assertIsNotNone(enrollment.revoked_at)
        query = db.scalar.await_args.args[0]
        self.assertIsNotNone(query._for_update_arg)
        audit = db.add.call_args.args[0]
        self.assertEqual(audit.action, 'revoked')
        self.assertFalse(hasattr(audit, 'code_verifier_hash'))

    def test_public_id_must_be_canonical_lowercase_uuid(self) -> None:
        """Test public id must be canonical lowercase uuid.
        """
        self.assertEqual(
            services.parse_canonical_enrollment_code_id(_CODE_ID_TEXT),
            _CODE_ID,
        )
        for invalid in (_CODE_ID_TEXT.upper(), 'not-a-uuid'):
            with self.subTest(value=invalid):
                with self.assertRaises(ValueError):
                    services.parse_canonical_enrollment_code_id(invalid)
