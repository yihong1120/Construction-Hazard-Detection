from __future__ import annotations

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.testclient import TestClient
from jwt.exceptions import InvalidTokenError

from examples.auth.models import User
from examples.db_management import openwebui_deps as identity
from examples.db_management.routers import openwebui as routes
from examples.db_management.schemas import openwebui as schemas
from examples.db_management.services import openwebui_services as bridge


def _admin_route_enforces_oidc_client_callback(db, /):
    return db


class TestOpenWebUIAdminBridge(unittest.IsolatedAsyncioTestCase):

    async def test_verified_subject_resolves_active_visionnaire_admin(
        self,
    ) -> None:
        operator = SimpleNamespace(
            role='admin',
            status='active',
            username='manager',
        )
        db = AsyncMock()
        db.scalar.return_value = SimpleNamespace(user=operator)
        credentials = SimpleNamespace(
            scheme='Bearer',
            credentials='signed-token',
        )
        with patch.object(
            identity.verifier,
            'decode_access_token',
            AsyncMock(return_value={'sub': 'subject-1', 'azp': 'open-webui'}),
        ):
            result = await identity.openwebui_admin(credentials, db)
        self.assertIs(result, operator)

    async def test_other_oidc_client_cannot_use_bridge(self) -> None:
        credentials = SimpleNamespace(
            scheme='Bearer',
            credentials='signed-token',
        )
        with patch.object(
            identity.verifier,
            'decode_access_token',
            AsyncMock(
                return_value={
                    'sub': 'subject-1',
                    'azp': 'other-client',
                },
            ),
        ):
            with self.assertRaises(HTTPException) as raised:
                await identity.openwebui_admin(credentials, AsyncMock())
        self.assertEqual(raised.exception.status_code, 401)

    async def test_update_delegates_one_loaded_target_to_transaction(
        self,
    ) -> None:
        target = SimpleNamespace(id=29)
        payload = schemas.OpenWebUIUserUpdate(password='replacement-password')
        db, operator = AsyncMock(), User(id=1)
        with (
            patch.object(
                bridge,
                'target_from_subject',
                AsyncMock(return_value=target),
            ) as load,
            patch.object(bridge, 'OpenWebUIAccountUpdate') as mutation,
        ):
            mutation.return_value.execute = AsyncMock()
            await bridge.update_user_from_openwebui(
                'subject-29',
                payload,
                db,
                operator,
            )
        load.assert_awaited_once_with('subject-29', db)
        mutation.assert_called_once_with(target, payload, operator, db)
        mutation.return_value.execute.assert_awaited_once_with('subject-29')

    async def test_own_password_change_revokes_sessions(
        self,
    ) -> None:
        user = User(id=29)
        with (
            patch.object(
                bridge,
                'ensure_keycloak_subject_for_user',
                AsyncMock(return_value='subject-29'),
            ),
            patch.object(
                bridge,
                'reset_keycloak_password',
                AsyncMock(),
            ) as reset,
            patch.object(
                bridge,
                'logout_keycloak_user',
                AsyncMock(),
            ) as logout,
        ):
            result = await bridge.change_own_password_from_openwebui(
                schemas.OpenWebUIPasswordChange(
                    new_password='replacement-password',
                ),
                AsyncMock(),
                user,
            )
        reset.assert_awaited_once_with(
            'subject-29',
            password='replacement-password',
            temporary=False,
        )
        logout.assert_awaited_once_with('subject-29')
        self.assertIn('sessions revoked', result['message'])


class TestOpenWebUIHttpBoundary(unittest.TestCase):
    def test_admin_route_enforces_oidc_client_status_and_role(self) -> None:
        """HTTP requests retain their original identity and admin gates."""
        app = FastAPI()
        app.include_router(routes.router)
        db = AsyncMock()
        app.dependency_overrides[identity.get_db] = partial(
            _admin_route_enforces_oidc_client_callback,
            db,
        )
        admin = SimpleNamespace(
            id=1,
            role='admin',
            username='manager',
            status='active',
        )
        member = SimpleNamespace(
            id=2,
            role='user',
            username='member',
            status='active',
        )
        suspended = SimpleNamespace(
            id=3,
            role='admin',
            username='manager',
            status='suspended',
        )
        valid = {'sub': 'subject-1', 'azp': 'open-webui'}
        cases = [
            ('missing', None, valid, admin, 401),
            ('invalid', 'signed-token', InvalidTokenError(), admin, 401),
            (
                'other-client',
                'signed-token',
                {'sub': 'subject-1', 'azp': 'other-client'},
                admin,
                401,
            ),
            ('suspended', 'signed-token', valid, suspended, 403),
            ('member', 'signed-token', valid, member, 403),
            ('admin', 'signed-token', valid, admin, 200),
        ]
        with TestClient(app) as client:
            for name, token, claims, user, expected in cases:
                db.scalar.return_value = SimpleNamespace(user=user)
                decode = (
                    AsyncMock(side_effect=claims)
                    if isinstance(claims, Exception)
                    else AsyncMock(return_value=claims)
                )
                with (
                    self.subTest(name=name),
                    patch.object(
                        identity.verifier,
                        'decode_access_token',
                        decode,
                    ),
                    patch.object(
                        bridge,
                        'update_user_from_openwebui',
                        AsyncMock(return_value={'message': 'synchronized'}),
                    ) as update,
                ):
                    response = client.patch(
                        '/internal/openwebui/users/by-sub/subject-2',
                        json={'name': 'New Name'},
                        headers=(
                            {'Authorization': f'Bearer {token}'}
                            if token
                            else {}
                        ),
                    )
                    self.assertEqual(response.status_code, expected)
                    if expected == 200:
                        update.assert_awaited_once()
                        assert update.await_args is not None
                        args = update.await_args.args
                        self.assertEqual(args[0], 'subject-2')
                        self.assertEqual(args[1].name, 'New Name')
                        self.assertIs(args[2], db)
                        self.assertIs(args[3], admin)
                        self.assertEqual(
                            response.json(),
                            {'message': 'synchronized'},
                        )
                    else:
                        update.assert_not_awaited()
        self.assertEqual(app.openapi()['paths'], {})


if __name__ == '__main__':
    unittest.main()
