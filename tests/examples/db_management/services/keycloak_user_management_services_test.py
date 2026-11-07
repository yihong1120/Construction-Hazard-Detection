from __future__ import annotations

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import HTTPException

from examples.auth import keycloak_admin as transport
from examples.db_management.services import (
    keycloak_user_management_services as services,
)
from examples.db_management.services import (
    native_social_links as social,
)


def _response_json(payload, /):
    return payload


async def _asyncSetUp_handle(httpx, testcase, /, request):
    if request.url.path.endswith('/token'):
        testcase.token_calls += 1
        return httpx.Response(
            testcase.token_status,
            json={
                'access_token': f'service-token-{testcase.token_calls}',
                'expires_in': 300,
            },
        )
    testcase.admin_calls += 1
    if testcase.admin_calls <= testcase.reject_count:
        return httpx.Response(401)
    return httpx.Response(204)


def _response(
    status_code: int,
    *,
    headers: dict[str, str] | None = None,
    payload: object = None,
) -> SimpleNamespace:
    """Build the narrow HTTP response double required by these unit tests."""
    return SimpleNamespace(
        status_code=status_code,
        headers=headers or {},
        json=partial(_response_json, payload),
    )


class TestKeycloakUserManagementServices(unittest.IsolatedAsyncioTestCase):
    """Exercise server-only Keycloak account lifecycle calls."""

    async def test_remote_create_starts_disabled_without_local_credentials(
        self,
    ):
        account = services.KeycloakAccount(
            'alice',
            'test-password',
            'alice@example.test',
            'Alice',
            'Test',
            True,
        )
        with patch.object(
            services,
            'keycloak_admin_request',
            new=AsyncMock(
                return_value=_response(
                    201,
                    headers={'Location': 'https://id.example/users/sub-123'},
                ),
            ),
        ) as request:
            subject = await services.create_keycloak_user(account)
        self.assertEqual(subject, 'sub-123')
        payload = request.await_args.kwargs['json_body']
        self.assertFalse(payload['enabled'])
        self.assertNotIn('password', payload)
        self.assertNotIn('test-password', repr(account))

    async def test_creation_retry_does_not_adopt_existing_remote_accounts(
        self,
    ):
        account = services.KeycloakAccount(
            'alice',
            'test-password',
            'alice@example.test',
            'Alice',
            'Test',
            False,
        )
        with patch.object(
            services,
            'keycloak_admin_request',
            new=AsyncMock(return_value=_response(409)),
        ) as request:
            with self.assertRaises(HTTPException) as error:
                await services.create_keycloak_user(account)
        self.assertEqual(error.exception.status_code, 409)
        self.assertEqual(request.await_count, 1)

    async def test_find_requires_an_exact_canonical_username_match(
        self,
    ) -> None:
        """A fuzzy Keycloak response can never be linked to another user."""
        response = _response(
            200,
            payload=[
                {'id': 'wrong', 'username': 'alice-two'},
                {'id': 'sub-123', 'username': 'Alice'},
            ],
        )
        with patch.object(
            services,
            'keycloak_admin_request',
            AsyncMock(return_value=response),
        ) as request:
            subject = await services.find_keycloak_user_subject('alice')

        self.assertEqual(subject, 'sub-123')
        call = request.await_args
        assert call is not None
        self.assertEqual(
            call.args,
            ('GET', '/users?username=alice&exact=true'),
        )

    async def test_update_rejects_unexpected_keycloak_response(self) -> None:
        """Provider failures are reduced to a safe unavailable response."""
        with patch.object(
            services,
            'keycloak_admin_request',
            AsyncMock(return_value=_response(500)),
        ):
            with self.assertRaises(HTTPException) as error:
                await services.update_keycloak_user(
                    'sub-123',
                    enabled=False,
                )

        self.assertEqual(error.exception.status_code, 503)

    async def test_assign_application_role_uses_exact_client_and_role(
        self,
    ) -> None:
        """Provisioning grants the new identity access to Open WebUI."""
        client = _response(200, payload=[{'id': 'client-1'}])
        role = _response(
            200,
            payload={'id': 'role-1', 'name': 'openwebui-user'},
        )
        assigned = _response(204)
        with patch.object(
            services,
            'keycloak_admin_request',
            AsyncMock(side_effect=[client, role, assigned]),
        ) as request:
            await services.assign_keycloak_client_role(
                'subject-1',
                client_id='open-webui',
                role_name='openwebui-user',
            )

        self.assertEqual(request.await_count, 3)
        self.assertEqual(
            request.await_args_list[2].args,
            ('POST', '/users/subject-1/role-mappings/clients/client-1'),
        )
        self.assertEqual(
            request.await_args_list[2].kwargs['json_body'],
            [{'id': 'role-1', 'name': 'openwebui-user'}],
        )

    async def test_logout_revokes_all_keycloak_sessions(self) -> None:
        with patch.object(
            services,
            'keycloak_admin_request',
            AsyncMock(return_value=_response(204)),
        ) as request:
            await services.logout_keycloak_user('subject/unsafe')

        request.assert_awaited_once_with(
            'POST',
            '/users/subject%2Funsafe/logout',
        )


if __name__ == '__main__':
    unittest.main()


class KeycloakTransportTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        import httpx

        self.httpx = httpx
        self.token_calls = 0
        self.admin_calls = 0
        self.reject_count = 0
        self.token_status = 200

        handle = partial(_asyncSetUp_handle, httpx, self)

        self.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        self.settings = SimpleNamespace(
            oidc_enabled=True,
            keycloak_realm='realm',
            keycloak_user_linker_client_id='linker',
            keycloak_user_linker_client_secret='private-service-secret',
            resolved_keycloak_admin_base_url='https://id.example',
        )
        self.settings_patch = patch.object(
            transport.keycloak_admin,
            'settings',
            self.settings,
        )
        self.pool_patch = patch.object(
            transport,
            'get_application_http_client',
            AsyncMock(
                return_value=self.client,
            ),
        )
        self.settings_patch.start()
        self.pool_patch.start()
        self.addCleanup(self.settings_patch.stop)
        self.addCleanup(self.pool_patch.stop)

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def test_admin_operations_reuse_lifespan_client_and_token(
        self,
    ) -> None:
        import asyncio

        operations = (
            services.keycloak_admin_request,
            social._keycloak_admin_request,
        )
        responses = await asyncio.gather(
            *(operations[index % 2]('GET', '/users') for index in range(8)),
        )
        self.assertTrue(all(r.status_code == 204 for r in responses))
        self.assertEqual((self.token_calls, self.admin_calls), (1, 8))
        self.assertFalse(self.client.is_closed)

    async def test_401_refreshes_service_token_once(self) -> None:
        self.reject_count = 1
        response = await services.keycloak_admin_request('GET', '/users')
        self.assertEqual(response.status_code, 204)
        self.assertEqual((self.token_calls, self.admin_calls), (2, 2))

    async def test_persistent_401_does_not_retry_indefinitely(self) -> None:
        self.reject_count = 100
        response = await services.keycloak_admin_request('GET', '/users')
        self.assertEqual(response.status_code, 401)
        self.assertEqual((self.token_calls, self.admin_calls), (2, 2))

    async def test_provider_token_failure_returns_safe_error(self) -> None:
        self.token_status = 500
        with self.assertRaises(HTTPException) as error:
            await services.keycloak_admin_request('GET', '/users')
        self.assertEqual(error.exception.status_code, 503)
        self.assertEqual(
            error.exception.detail,
            'keycloak_identity_service_unavailable',
        )
        self.assertEqual(self.admin_calls, 0)

    async def test_missing_configuration_fails_before_network_io(self) -> None:
        self.settings.keycloak_user_linker_client_secret = ''
        with self.assertRaises(HTTPException) as error:
            await services.keycloak_admin_request('GET', '/users')
        self.assertEqual(error.exception.status_code, 503)
        self.assertEqual(self.token_calls, 0)

    async def test_disposable_client_closes_outside_application_lifespan(
        self,
    ) -> None:
        with (
            patch.object(
                transport,
                'get_application_http_client',
                AsyncMock(return_value=None),
            ),
            patch.object(
                transport.httpx,
                'AsyncClient',
                return_value=self.client,
            ),
        ):
            response = await services.keycloak_admin_request('GET', '/users')
        self.assertEqual(response.status_code, 204)
        self.assertTrue(self.client.is_closed)
