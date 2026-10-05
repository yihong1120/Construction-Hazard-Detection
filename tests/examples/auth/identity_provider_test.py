from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from examples.auth import identity_provider


def test_account_console_requires_configured_provider():
    with patch.object(
        identity_provider,
        'settings',
        SimpleNamespace(
            oidc_enabled=True, oidc_account_url='https://sso.example/account',
        ),
    ):
        assert (
            identity_provider.identity_provider_account_url()
            == 'https://sso.example/account'
        )
    with patch.object(
        identity_provider,
        'settings',
        SimpleNamespace(oidc_enabled=False, oidc_account_url=''),
    ):
        with pytest.raises(HTTPException) as error:
            identity_provider.identity_provider_account_url()
        assert error.value.status_code == 404
