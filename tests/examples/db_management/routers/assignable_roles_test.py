from __future__ import annotations

from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.testclient import TestClient

from examples.db_management.deps import SUPER_ADMIN_NAME
from examples.db_management.routers import users
from examples.db_management.services.role_policy import validate_role


class AssignableRolesTest(IsolatedAsyncioTestCase):
    def test_protected_and_peer_accounts_return_empty_but_cannot_be_mutated(
        self,
    ):
        app = FastAPI()
        app.include_router(users.router)
        operator = SimpleNamespace(
            id=2,
            username='group-admin',
            role='admin',
            group_id=1,
            tenant_id='t1',
        )
        target = SimpleNamespace(
            id=1,
            username=SUPER_ADMIN_NAME,
            role='admin',
            group_id=1,
            tenant_id='t1',
        )
        db = AsyncMock()
        db.get.return_value = target
        app.dependency_overrides[users.get_current_user] = lambda: operator
        app.dependency_overrides[users.get_db] = lambda: db
        with TestClient(app) as client:
            for name in (SUPER_ADMIN_NAME, 'peer-admin'):
                target.username = name
                response = client.get(
                    '/users/assignable-roles?target_user_id=1&locale=zh-TW',
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()['options'], [])
                self.assertIsNone(response.json()['default_id'])
                for role in ('admin', 'user', 'guest', 'super_admin'):
                    with self.assertRaises(HTTPException) as raised:
                        validate_role(operator, role, target)
                    self.assertEqual(raised.exception.status_code, 403)
            # The protected account cannot demote itself either.
            operator.username = SUPER_ADMIN_NAME
            target.username = SUPER_ADMIN_NAME
            self.assertEqual(
                client.get('/users/assignable-roles?target_user_id=1').json()[
                    'options'
                ],
                [],
            )
            with self.assertRaises(HTTPException):
                validate_role(operator, 'user', target)
            # It may still assign roles to other administrators.
            target.username = 'peer-admin'
            self.assertEqual(
                len(
                    client.get(
                        '/users/assignable-roles?target_user_id=1',
                    ).json()['options'],
                ),
                3,
            )

    def test_out_of_scope_targets_remain_forbidden(self):
        app = FastAPI()
        app.include_router(users.router)
        operator = SimpleNamespace(
            id=2,
            username='group-admin',
            role='admin',
            group_id=1,
            tenant_id='t1',
        )
        target = SimpleNamespace(
            id=1,
            username=SUPER_ADMIN_NAME,
            role='admin',
            group_id=1,
            tenant_id='t2',
        )
        db = AsyncMock()
        db.get.return_value = target
        app.dependency_overrides[users.get_current_user] = lambda: operator
        app.dependency_overrides[users.get_db] = lambda: db
        with TestClient(app) as client:
            self.assertEqual(
                client.get(
                    '/users/assignable-roles?target_user_id=1',
                ).status_code,
                403,
            )
            target.tenant_id = 't1'
            target.group_id = 2
            self.assertEqual(
                client.get(
                    '/users/assignable-roles?target_user_id=1',
                ).status_code,
                403,
            )
            target.group_id = 1
            operator.role = 'user'
            self.assertEqual(
                client.get(
                    '/users/assignable-roles?target_user_id=1',
                ).status_code,
                403,
            )
            operator.role = 'admin'
            db.get.return_value = None
            self.assertEqual(
                client.get(
                    '/users/assignable-roles?target_user_id=1',
                ).status_code,
                404,
            )
