from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import HTTPException

from examples.db_management.services import (
    keycloak_user_management_services as services,
)


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
        json=lambda: payload,
    )


class TestKeycloakUserManagementServices(unittest.IsolatedAsyncioTestCase):
    """Exercise server-only Keycloak account lifecycle calls."""

    async def test_provision_creates_identity_then_sets_password(self) -> None:
        """A new account is not local-password backed."""
        created = _response(
            201,
            headers={'Location': 'https://id.example/realms/r/users/sub-123'},
        )
        password_set = _response(204)
        with patch.object(
            services,
            'keycloak_admin_request',
            AsyncMock(side_effect=[created, password_set]),
        ) as request:
            subject = await services.provision_keycloak_user(
                username='alice',
                password='safe-password',
                email='alice@example.com',
                given_name='Alice',
                family_name='Example',
                force_password_change=True,
            )

        self.assertEqual(subject, 'sub-123')
        self.assertEqual(request.await_count, 2)
        self.assertEqual(
            request.await_args_list[0].kwargs['json_body'],
            {
                'username': 'alice',
                'email': 'alice@example.com',
                'emailVerified': True,
                'firstName': 'Alice',
                'lastName': 'Example',
                'enabled': True,
            },
        )
        self.assertEqual(
            request.await_args_list[1].kwargs['json_body'],
            {
                'type': 'password',
                'value': 'safe-password',
                'temporary': True,
            },
        )

    async def test_provision_compensates_when_password_initialisation_fails(
        self,
    ) -> None:
        """A failed password step deletes the otherwise unusable identity."""
        created = _response(
            201,
            headers={'Location': 'https://id.example/users/sub-123'},
        )
        password_failed = _response(500)
        deleted = _response(204)
        with patch.object(
            services,
            'keycloak_admin_request',
            AsyncMock(side_effect=[created, password_failed, deleted]),
        ) as request:
            with self.assertRaises(HTTPException) as error:
                await services.provision_keycloak_user(
                    username='alice',
                    password='safe-password',
                    email='alice@example.com',
                    given_name='Alice',
                    family_name='Example',
                    force_password_change=False,
                )

        self.assertEqual(error.exception.status_code, 503)
        self.assertEqual(request.await_count, 3)
        self.assertEqual(
            request.await_args_list[2].args,
            ('DELETE', '/users/sub-123'),
        )

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


if __name__ == '__main__':
    unittest.main()
