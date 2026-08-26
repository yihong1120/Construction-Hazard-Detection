from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from examples.auth import identity_provider


class TestIdentityProviderBoundaries(unittest.TestCase):
    """Verify that password writes stop after the OIDC cutover flag is set."""

    def test_external_password_management_returns_account(self) -> None:
        """Visionnaire returns Keycloak's account route instead of a write."""
        with patch.object(
            identity_provider,
            'settings',
            SimpleNamespace(
                oidc_enabled=True,
                oidc_account_url='https://sso.example.com/realms/app/account',
                oidc_passwords_managed_externally=True,
            ),
        ):
            with self.assertRaises(HTTPException) as error:
                identity_provider.require_local_password_management()

        self.assertEqual(error.exception.status_code, 409)
        detail = error.exception.detail
        self.assertIsInstance(detail, dict)
        assert isinstance(detail, dict)
        self.assertEqual(
            detail['account_url'],
            'https://sso.example.com/realms/app/account',
        )

    def test_legacy_password_management_remains_available_during_migration(
        self,
    ) -> None:
        """The switch remains additive until all accounts have moved."""
        with patch.object(
            identity_provider,
            'settings',
            SimpleNamespace(oidc_passwords_managed_externally=False),
        ):
            identity_provider.require_local_password_management()

    def test_external_password_cutover_rejects_local_login(self) -> None:
        """A central-password cutover also disables stale local logins."""
        with patch.object(
            identity_provider,
            'settings',
            SimpleNamespace(oidc_passwords_managed_externally=True),
        ):
            with self.assertRaises(HTTPException) as error:
                identity_provider.require_local_login()

        self.assertEqual(error.exception.status_code, 409)
        detail = error.exception.detail
        self.assertIsInstance(detail, dict)
        assert isinstance(detail, dict)
        self.assertEqual(
            detail['code'],
            'login_managed_by_identity_provider',
        )

    def test_external_cutover_rejects_direct_social_identity_linking(
        self,
    ) -> None:
        """Google and Apple linking moves to the Keycloak account console."""
        with patch.object(
            identity_provider,
            'settings',
            SimpleNamespace(
                oidc_enabled=True,
                oidc_account_url='https://sso.example.com/realms/app/account',
                oidc_passwords_managed_externally=True,
            ),
        ):
            with self.assertRaises(HTTPException) as error:
                identity_provider.require_local_identity_management()

        self.assertEqual(error.exception.status_code, 409)
        detail = error.exception.detail
        self.assertIsInstance(detail, dict)
        assert isinstance(detail, dict)
        self.assertEqual(
            detail['code'],
            'social_identities_managed_by_identity_provider',
        )
        self.assertEqual(
            detail['account_url'],
            'https://sso.example.com/realms/app/account',
        )
