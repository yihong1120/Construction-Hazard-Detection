from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import patch
from uuid import UUID

from fastapi import HTTPException
from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth import deployment_context
from examples.auth.deployment_context import canonical_api_base_url
from examples.auth.deployment_context import DeploymentBinding
from examples.auth.deployment_context import request_api_base_url
from examples.auth.deployment_context import require_deployment_match
from examples.auth.deployment_context import resolve_request_deployment
from examples.auth.deployment_context import (
    trusted_local_development_deployment_id,
)


class TestDeploymentContract(unittest.IsolatedAsyncioTestCase):
    """Exercise deployment-origin canonicalisation and token binding."""

    def setUp(self) -> None:
        """Perform setUp."""
        self.binding = DeploymentBinding(
            tenant_id=UUID('00000000-0000-0000-0000-000000000001'),
            deployment_id=UUID('00000000-0000-0000-0000-000000000002'),
            api_base_url='https://api.example.com/hazard/api',
            config_revision=3,
        )

    def test_canonical_api_root_requires_configured_path_without_query(
        self,
    ) -> None:
        """Test canonical api root requires configured path without query."""
        self.assertEqual(
            canonical_api_base_url('https://API.EXAMPLE.COM/hazard/api/'),
            'https://api.example.com/hazard/api',
        )
        self.assertEqual(
            canonical_api_base_url('https://api.example.com:443/hazard/api'),
            'https://api.example.com/hazard/api',
        )
        for value in (
            'http://api.example.com',
            'https://api.example.com',
            'https://api.example.com/db_management',
            'https://api.example.com?tenant_id=x',
            'https://user:pass@api.example.com',
            'https://api.example.com:not-a-port',
        ):
            with self.assertRaises(ValueError):
                canonical_api_base_url(value)

    @staticmethod
    def _request(host: str, client_host: str) -> Request:
        """Build a minimal direct-Uvicorn request for local-mode tests."""
        return Request(
            {
                'type': 'http',
                'asgi': {'version': '3.0'},
                'http_version': '1.1',
                'method': 'POST',
                'scheme': 'http',
                'path': '/login',
                'raw_path': b'/login',
                'query_string': b'',
                'headers': [(b'host', host.encode('ascii'))],
                'client': (client_host, 50000),
                'server': ('127.0.0.1', 8005),
            },
        )

    def test_local_development_uses_configured_deployment(self) -> None:
        """Test local development mode uses only server configured
        deployment."""
        with patch.dict(
            os.environ,
            {
                'LOCAL_DEVELOPMENT_AUTH_ENABLED': 'true',
                'LOCAL_DEVELOPMENT_DEPLOYMENT_ID': str(
                    self.binding.deployment_id,
                ),
            },
            clear=False,
        ):
            deployment_id = trusted_local_development_deployment_id(
                self._request('127.0.0.1:8005', '127.0.0.1'),
            )
            remote_attempt = trusted_local_development_deployment_id(
                self._request('127.0.0.1:8005', '192.0.2.10'),
            )

        self.assertEqual(deployment_id, self.binding.deployment_id)
        self.assertIsNone(remote_attempt)

    def test_binding_and_local_mode_reject_invalid_configuration(self) -> None:
        """Bindings expose stable claims and local mode rejects invalid IDs."""
        self.assertEqual(self.binding.issuer, self.binding.api_base_url)
        self.assertIn(str(self.binding.deployment_id), self.binding.audience)
        self.assertEqual(
            self.binding.as_response()['config_revision'],
            self.binding.config_revision,
        )
        self.assertTrue(deployment_context._is_loopback_host('::1'))
        self.assertTrue(deployment_context._is_loopback_host('localhost'))
        self.assertFalse(deployment_context._is_loopback_host(None))
        self.assertFalse(deployment_context._is_loopback_host('example.com'))

        with patch.dict(
            os.environ,
            {
                'LOCAL_DEVELOPMENT_AUTH_ENABLED': '1',
                'LOCAL_DEVELOPMENT_DEPLOYMENT_ID': 'not-a-uuid',
            },
            clear=False,
        ):
            with self.assertRaises(HTTPException) as invalid_configuration:
                trusted_local_development_deployment_id(
                    self._request('localhost:8005', '127.0.0.1'),
                )
        self.assertEqual(invalid_configuration.exception.status_code, 503)

    def test_request_origin_handles_disabled_local_mode_and_ipv6_urls(
        self,
    ) -> None:
        """Origin helpers build configured URLs and retain safe defaults.

        Explicitly disabled local mode never uses ambient deployment values.
        """
        with patch.dict(
            os.environ,
            {'LOCAL_DEVELOPMENT_AUTH_ENABLED': 'false'},
            clear=False,
        ):
            self.assertIsNone(
                trusted_local_development_deployment_id(
                    self._request('localhost:8005', '127.0.0.1'),
                ),
            )
        self.assertEqual(
            canonical_api_base_url('https://[::1]/hazard/api'),
            'https://[::1]/hazard/api',
        )
        request = self._request('api.example.com:443', '192.0.2.1')
        request.scope['scheme'] = 'https'
        with patch.dict(
            os.environ,
            {'DEPLOYMENT_API_BASE_PATH': '/public/api'},
            clear=False,
        ):
            self.assertEqual(
                request_api_base_url(request),
                'https://api.example.com/public/api',
            )

    async def test_resolve_and_compare_deployment_states(self) -> None:
        """Resolution distinguishes unknown, disabled, revoked, and mismatch.

        Every state returns a stable conflict response.
        """
        request = self._request('api.example.com:8005', '192.0.2.1')
        db = SimpleNamespace(scalar=AsyncMock())

        with patch.object(
            deployment_context,
            'request_api_base_url',
            return_value=self.binding.api_base_url,
        ):
            db.scalar.return_value = None
            with self.assertRaises(HTTPException) as unknown:
                await resolve_request_deployment(
                    cast(Request, request),
                    cast(AsyncSession, db),
                )
            self.assertEqual(unknown.exception.status_code, 409)

            db.scalar.return_value = SimpleNamespace(
                tenant=SimpleNamespace(status='disabled'),
                status='active',
            )
            with self.assertRaises(HTTPException) as disabled:
                await resolve_request_deployment(
                    cast(Request, request),
                    cast(AsyncSession, db),
                )
            self.assertEqual(
                disabled.exception.detail['code'], 'tenant_disabled',
            )

            db.scalar.return_value = SimpleNamespace(
                tenant=SimpleNamespace(status='active'),
                status='revoked',
            )
            with self.assertRaises(HTTPException) as revoked:
                await resolve_request_deployment(
                    cast(Request, request),
                    cast(AsyncSession, db),
                )
            self.assertEqual(
                revoked.exception.detail['code'], 'deployment_revoked',
            )

            db.scalar.return_value = SimpleNamespace(
                id=self.binding.deployment_id,
                tenant_id=self.binding.tenant_id,
                api_base_url=self.binding.api_base_url,
                config_revision=self.binding.config_revision,
                tenant=SimpleNamespace(status='active'),
                status='active',
            )
            resolved = await require_deployment_match(
                cast(Request, request),
                cast(AsyncSession, db),
                tenant_id=str(self.binding.tenant_id),
                deployment_id=str(self.binding.deployment_id),
                config_revision=self.binding.config_revision,
            )
            self.assertEqual(resolved, self.binding)

            with self.assertRaises(HTTPException) as mismatch:
                await require_deployment_match(
                    cast(Request, request),
                    cast(AsyncSession, db),
                    tenant_id='wrong-tenant',
                    deployment_id=str(self.binding.deployment_id),
                    config_revision=self.binding.config_revision,
                )
        self.assertEqual(
            mismatch.exception.detail['code'],
            'deployment_configuration_changed',
        )


if __name__ == '__main__':
    unittest.main()
