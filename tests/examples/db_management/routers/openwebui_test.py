from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch
from uuid import UUID

from fastapi import HTTPException

from examples.db_management.routers import openwebui as bridge
from examples.db_management.schemas.user import UserProfileBase


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
            scheme='Bearer', credentials='signed-token',
        )
        with patch.object(
            bridge.verifier,
            'decode_access_token',
            AsyncMock(return_value={'sub': 'subject-1', 'azp': 'open-webui'}),
        ):
            result = await bridge.openwebui_admin(credentials, db)
        self.assertIs(result, operator)

    async def test_other_oidc_client_cannot_use_bridge(self) -> None:
        credentials = SimpleNamespace(
            scheme='Bearer', credentials='signed-token',
        )
        with patch.object(
            bridge.verifier,
            'decode_access_token',
            AsyncMock(
                return_value={
                    'sub': 'subject-1',
                    'azp': 'other-client',
                },
            ),
        ):
            with self.assertRaises(HTTPException) as raised:
                await bridge.openwebui_admin(credentials, AsyncMock())
        self.assertEqual(raised.exception.status_code, 401)

    async def test_create_assigns_openwebui_role_after_canonical_user(
            self,
    ) -> None:
        payload = bridge.OpenWebUIUserCreate(
            username='new@example.com',
            password='initial-password',
            role='user',
            group_id=None,
            profile=UserProfileBase(
                family_name='', given_name='New User', email='new@example.com',
            ),
        )
        operator = SimpleNamespace(id=1, group_id=8)
        tenant_id = UUID('11111111-1111-4111-8111-111111111111')
        stored = SimpleNamespace(id=27, tenant_id=tenant_id)
        db = AsyncMock()
        db.get.return_value = stored
        created = SimpleNamespace(
            id=27,
            group_id=8,
            group_name='工程一組',
        )
        with (
            patch.object(
                bridge,
                'add_user',
                AsyncMock(return_value=created),
            ) as add,
            patch.object(
                bridge,
                'ensure_keycloak_subject_for_user',
                AsyncMock(return_value='subject-27'),
            ),
            patch.object(
                bridge,
                'assign_keycloak_client_role',
                AsyncMock(),
            ) as assign,
            patch.object(
                bridge,
                'list_site_ids_for_group',
                AsyncMock(return_value=[3, 7]),
            ),
        ):
            result = await bridge.create_user_from_openwebui(
                payload,
                db,
                operator,
            )
        self.assertEqual(result['keycloak_subject'], 'subject-27')
        self.assertEqual(result['scope'].tenant_id, tenant_id)
        self.assertEqual(result['scope'].group_id, 8)
        self.assertEqual(result['scope'].site_ids, [3, 7])
        assert add.await_args is not None
        self.assertEqual(
            add.await_args.kwargs['payload'].group_id,
            8,
        )
        assign.assert_awaited_once_with(
            'subject-27', client_id='open-webui', role_name='openwebui-user',
        )

    async def test_pending_openwebui_user_is_suspended_in_identity_source(
            self,
    ) -> None:
        payload = bridge.OpenWebUIUserCreate(
            username='pending@example.com',
            password='initial-password',
            role='user',
            openwebui_role='pending',
            group_id=None,
            profile=UserProfileBase(
                family_name='',
                given_name='Pending',
                email='pending@example.com',
            ),
        )
        stored = SimpleNamespace(
            id=28,
            tenant_id=UUID('11111111-1111-4111-8111-111111111111'),
        )
        created = SimpleNamespace(id=28, group_id=None, group_name=None)
        db = AsyncMock()
        db.get.return_value = stored
        with (
            patch.object(bridge, 'add_user', AsyncMock(return_value=created)),
            patch.object(
                bridge,
                'ensure_keycloak_subject_for_user',
                AsyncMock(return_value='subject-28'),
            ),
            patch.object(
                bridge,
                'assign_keycloak_client_role',
                AsyncMock(),
            ),
            patch.object(
                bridge,
                'update_user_status',
                AsyncMock(),
            ) as suspend,
        ):
            await bridge.create_user_from_openwebui(
                payload, db, SimpleNamespace(id=1, group_id=None),
            )
        suspend.assert_awaited_once()
        assert suspend.await_args is not None
        self.assertEqual(suspend.await_args.args[0].status, 'suspended')

    async def test_openwebui_password_update_is_not_temporary(self) -> None:
        """Open WebUI resets must not trigger UPDATE_PASSWORD on next login."""
        target = SimpleNamespace(id=29, status='active', role='user')
        payload = bridge.OpenWebUIUserUpdate(password='replacement-password')
        with patch.object(
            bridge, 'target_from_subject', AsyncMock(return_value=target),
        ), patch.object(
            bridge, 'admin_update_pwd_by_id', AsyncMock(),
        ) as update_password:
            await bridge.update_user_from_openwebui(
                'subject-29', payload, AsyncMock(), SimpleNamespace(id=1),
            )
        update_password.assert_awaited_once()
        assert update_password.await_args is not None
        self.assertFalse(update_password.await_args.kwargs['temporary'])

    async def test_own_password_change_revokes_sessions(
        self,
    ) -> None:
        user = SimpleNamespace(id=29)
        with patch.object(
            bridge,
            'ensure_keycloak_subject_for_user',
            AsyncMock(return_value='subject-29'),
        ), patch.object(
            bridge, 'reset_keycloak_password', AsyncMock(),
        ) as reset, patch.object(
            bridge, 'logout_keycloak_user', AsyncMock(),
        ) as logout:
            result = await bridge.change_own_password_from_openwebui(
                bridge.OpenWebUIPasswordChange(
                    new_password='replacement-password',
                ),
                AsyncMock(),
                user,
            )
        reset.assert_awaited_once_with(
            'subject-29', password='replacement-password', temporary=False,
        )
        logout.assert_awaited_once_with('subject-29')
        self.assertIn('sessions revoked', result['message'])


if __name__ == '__main__':
    unittest.main()
