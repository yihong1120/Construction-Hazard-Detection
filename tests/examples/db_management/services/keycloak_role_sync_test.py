from __future__ import annotations

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import HTTPException

from examples.db_management.services import (
    keycloak_user_management_services as module,
)


def _response_json(data, /):
    return data


def response(status, data=None):
    return SimpleNamespace(
        status_code=status,
        json=partial(_response_json, data),
    )


class KeycloakRoleSyncTest(unittest.IsolatedAsyncioTestCase):
    async def test_matching_roles_require_only_client_and_assignment_reads(
        self,
    ):
        with patch.object(
            module,
            'keycloak_admin_request',
            new=AsyncMock(
                side_effect=[
                    response(200, [{'id': 'client'}]),
                    response(200, [{'id': 'role', 'name': 'openwebui-user'}]),
                ],
            ),
        ) as request:
            await module.synchronize_openwebui_roles('subject', 'user')
        self.assertEqual(request.await_count, 2)
        self.assertEqual(
            [call.args[0] for call in request.await_args_list],
            ['GET', 'GET'],
        )

    async def test_role_switch_preserves_unmanaged_roles(
        self,
    ):
        old = {'id': 'old', 'name': 'openwebui-admin'}
        new = {'id': 'new', 'name': 'openwebui-user'}
        with patch.object(
            module,
            'keycloak_admin_request',
            new=AsyncMock(
                side_effect=[
                    response(200, [{'id': 'client'}]),
                    response(200, [old, {'id': 'other', 'name': 'auditor'}]),
                    response(200, new),
                    response(204),
                    response(204),
                ],
            ),
        ) as request:
            await module.synchronize_openwebui_roles('subject', 'user')
        self.assertEqual(request.await_count, 5)
        self.assertEqual(
            request.await_args_list[-1].kwargs['json_body'],
            [old],
        )
        self.assertEqual(
            request.await_args_list[-2].kwargs['json_body'],
            [new],
        )

    async def test_partial_remote_success_can_be_retried_without_reassigning(
        self,
    ):
        old = {'id': 'old', 'name': 'openwebui-admin'}
        new = {'id': 'new', 'name': 'openwebui-user'}
        with patch.object(
            module,
            'keycloak_admin_request',
            new=AsyncMock(
                side_effect=[
                    response(200, [{'id': 'client'}]),
                    response(200, [old, new]),
                    response(204),
                ],
            ),
        ) as request:
            await module.synchronize_openwebui_roles('subject', 'user')
        self.assertEqual(
            [call.args[0] for call in request.await_args_list],
            ['GET', 'GET', 'DELETE'],
        )

    async def test_provider_failures_do_not_mark_roles_synchronized(self):
        client = response(200, [{'id': 'client'}])
        assigned = response(200, [{'id': 'old', 'name': 'openwebui-admin'}])
        new = response(200, {'id': 'new', 'name': 'openwebui-user'})
        for replies in (
            [response(503)],
            [response(200, [])],
            [client, response(503)],
            [client, response(200, {})],
            [client, assigned, response(503)],
            [client, assigned, new, response(503)],
            [client, assigned, new, response(204), response(503)],
        ):
            with self.subTest(replies=len(replies)):
                with patch.object(
                    module,
                    'keycloak_admin_request',
                    new=AsyncMock(side_effect=replies),
                ):
                    with self.assertRaises(HTTPException):
                        await module.synchronize_openwebui_roles(
                            'subject',
                            'user',
                        )

    async def test_managed_scope_is_validated_before_any_provider_call(self):
        with patch.object(
            module,
            'keycloak_admin_request',
            new=AsyncMock(),
        ) as request:
            with self.assertRaises(ValueError):
                await module.synchronize_openwebui_roles(
                    'subject',
                    'super_admin',
                )
            with self.assertRaises(ValueError):
                await module.KeycloakClientRoles(
                    'open-webui',
                    frozenset({'openwebui-user'}),
                ).synchronize('subject', {'auditor'})
        request.assert_not_awaited()
