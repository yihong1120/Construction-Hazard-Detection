from __future__ import annotations

import unittest
from types import SimpleNamespace
from uuid import UUID

from examples.db_management.services.auth_token_issuer import (
    deployment_claims,
)


class TestDeploymentClaims(unittest.TestCase):
    """Verify token claims are omitted safely without deployment context."""

    def test_deployment_claims_return_empty_mapping_for_no_deployment(
        self,
    ) -> None:
        """Anonymous contexts do not add empty deployment claim keys."""
        self.assertEqual(deployment_claims(None), {})

    def test_deployment_claims_serialise_the_bound_contract(self) -> None:
        """Deployment context is converted to stable string token claims."""
        deployment = SimpleNamespace(
            tenant_id=UUID('11111111-1111-1111-1111-111111111111'),
            deployment_id=UUID('22222222-2222-2222-2222-222222222222'),
            config_revision=4,
        )
        self.assertEqual(
            deployment_claims(deployment)['config_revision'],
            4,
        )
