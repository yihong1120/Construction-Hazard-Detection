from __future__ import annotations

import base64
import json
import unittest
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch
from uuid import UUID

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.hazmat.primitives.serialization import NoEncryption
from cryptography.hazmat.primitives.serialization import PrivateFormat
from cryptography.hazmat.primitives.serialization import PublicFormat
from fastapi import FastAPI
from fastapi.testclient import TestClient

from examples.auth.database import get_db
from examples.auth.models import Deployment
from examples.auth.models import DEPLOYMENT_STATUS_ACTIVE
from examples.auth.models import DEPLOYMENT_STATUS_REVOKED
from examples.auth.models import TENANT_STATUS_ACTIVE
from examples.auth.models import TENANT_STATUS_DISABLED
from examples.deployment_registry import router as registry
from examples.deployment_registry import service as registry_service
from examples.deployment_registry.schemas import (
    DeploymentRegistryDocument,
)
from examples.deployment_registry.schemas import (
    MAX_REGISTRY_TTL_SECONDS,
)
from examples.deployment_registry.signing import (
    build_registry_document,
)
"""Contract tests for the anonymous signed deployment registry."""


_TENANT_ID = UUID('11111111-1111-1111-1111-111111111111')
_DEPLOYMENT_ID = UUID('22222222-2222-2222-2222-222222222222')
_PATH = f'/deployment-registry/v1/deployments/{_DEPLOYMENT_ID}'


def _base64url_decode(value: str) -> bytes:
    """Decode the unpadded base64url representation used by the registry."""
    return base64.urlsafe_b64decode(value + '=' * (-len(value) % 4))


class TestDeploymentRegistryRouter(unittest.TestCase):
    """Verify the public route never becomes an authentication endpoint."""

    def setUp(self) -> None:
        """Perform setUp.
        """
        self.app = FastAPI()
        self.app.include_router(registry.router)
        self.client = TestClient(self.app, base_url='https://api.example.com')
        self.deployment = SimpleNamespace(
            id=_DEPLOYMENT_ID,
            tenant_id=_TENANT_ID,
            tenant=SimpleNamespace(status=TENANT_STATUS_ACTIVE),
            api_base_url='https://api.example.com/hazard/api',
            config_revision=7,
            status=DEPLOYMENT_STATUS_ACTIVE,
        )
        self.db = MagicMock()
        self.db.scalar = AsyncMock(return_value=self.deployment)

        async def override_get_db() -> Any:  # type: ignore[no-untyped-def]
            """Perform override get db.

            Returns:
                The callable result.
            """
            yield self.db

        self.app.dependency_overrides[get_db] = override_get_db
        self.private_key = Ed25519PrivateKey.generate()
        self.private_pem = self.private_key.private_bytes(
            Encoding.PEM,
            PrivateFormat.PKCS8,
            NoEncryption(),
        ).decode('ascii')
        self.settings_patch = patch.multiple(
            registry_service.settings,
            deployment_registry_ed25519_private_key=self.private_pem,
            deployment_registry_key_id='registry-ed25519-2026-01',
            deployment_registry_ttl_seconds=600,
        )
        self.settings_patch.start()

    def tearDown(self) -> None:
        """Perform tearDown.
        """
        self.settings_patch.stop()
        self.client.close()

    def test_returns_exact_signed_nine_field_document(self) -> None:
        """Test returns exact signed nine field document.
        """
        response = self.client.get(_PATH)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['content-type'], 'application/json')
        document = response.json()
        self.assertEqual(
            set(document),
            {
                'schema_version',
                'deployment_id',
                'tenant_id',
                'api_base_url',
                'config_revision',
                'issued_at',
                'expires_at',
                'key_id',
                'signature',
            },
        )
        self.assertEqual(document['deployment_id'], str(_DEPLOYMENT_ID))
        self.assertEqual(document['tenant_id'], str(_TENANT_ID))
        self.assertEqual(document['api_base_url'], 'https://api.example.com/hazard/api')
        self.assertEqual(document['config_revision'], 7)
        self.assertEqual(document['expires_at'] - document['issued_at'], 600)
        public_key = self.private_key.public_key()
        signed_document = document.copy()
        del signed_document['key_id']
        del signed_document['signature']
        public_key.verify(
            _base64url_decode(document['signature']),
            json.dumps(
                signed_document,
                ensure_ascii=False,
                separators=(',', ':'),
                sort_keys=True,
            ).encode('utf-8'),
        )

    def test_unknown_deployment_returns_not_found(self) -> None:
        """An unknown deployment returns the service-level not-found result."""
        self.db.scalar.return_value = None
        unknown = self.client.get(_PATH, follow_redirects=False)

        self.assertEqual(unknown.status_code, 404)

    def test_disabled_tenant_or_revoked_deployment_is_not_discoverable(self) -> None:
        """Test disabled tenant or revoked deployment is not discoverable.
        """
        for tenant_status, deployment_status in (
            (TENANT_STATUS_DISABLED, DEPLOYMENT_STATUS_ACTIVE),
            (TENANT_STATUS_ACTIVE, DEPLOYMENT_STATUS_REVOKED),
        ):
            with self.subTest(
                tenant_status=tenant_status,
                deployment_status=deployment_status,
            ):
                self.deployment.tenant.status = tenant_status
                self.deployment.status = deployment_status
                response = self.client.get(_PATH)
                self.assertEqual(response.status_code, 410)
                self.assertEqual(
                    response.json()['detail']['code'],
                    'deployment_revoked',
                )


class TestDeploymentRegistryService(unittest.TestCase):
    """Test signing invariants independently from HTTP plumbing."""

    def setUp(self) -> None:
        """Perform setUp.
        """
        self.private_key = Ed25519PrivateKey.generate()
        self.private_pem = self.private_key.private_bytes(
            Encoding.PEM,
            PrivateFormat.PKCS8,
            NoEncryption(),
        ).decode('ascii')
        self.deployment = cast(
            Deployment,
            SimpleNamespace(
                id=_DEPLOYMENT_ID,
                tenant_id=_TENANT_ID,
                api_base_url='https://api.example.com/hazard/api',
                config_revision=1,
            ),
        )

    def test_ttl_cannot_exceed_24_hours(self) -> None:
        """Test ttl cannot exceed 24 hours.
        """
        with self.assertRaises(ValueError):
            build_registry_document(
                self.deployment,
                private_key_pem=self.private_pem,
                key_id='registry-ed25519-2026-01',
                ttl_seconds=MAX_REGISTRY_TTL_SECONDS + 1,
                issued_at=1,
            )

    def test_schema_rejects_noncanonical_signature_document(self) -> None:
        """Test schema rejects noncanonical signature document."""
        document = build_registry_document(
            self.deployment,
            private_key_pem=self.private_pem,
            key_id='registry-ed25519-2026-01',
            ttl_seconds=60,
            issued_at=100,
        )
        document['signature'] = 'not-a-real-ed25519-signature'
        with self.assertRaises(ValueError):
            DeploymentRegistryDocument.model_validate(document)

    def test_service_refuses_a_host_root_or_service_specific_api_url(self) -> None:
        """Test service refuses a host root or service specific api url.
        """
        for api_base_url in (
            'https://api.example.com',
            'https://api.example.com/hazard/api/db_management',
        ):
            with self.subTest(api_base_url=api_base_url):
                self.deployment.api_base_url = api_base_url
                with self.assertRaises(ValueError):
                    build_registry_document(
                        self.deployment,
                        private_key_pem=self.private_pem,
                        key_id='registry-ed25519-2026-01',
                        ttl_seconds=60,
                        issued_at=100,
                    )

    def test_public_key_pem_is_exportable_but_not_part_of_document(self) -> None:
        """Pinning/export is an operator concern; the endpoint returns no key."""
        public_pem = self.private_key.public_key().public_bytes(
            Encoding.PEM,
            PublicFormat.SubjectPublicKeyInfo,
        )
        document = build_registry_document(
            self.deployment,
            private_key_pem=self.private_pem,
            key_id='registry-ed25519-2026-01',
            ttl_seconds=60,
            issued_at=100,
        )
        self.assertTrue(public_pem.startswith(b'-----BEGIN PUBLIC KEY-----'))
        self.assertNotIn('public_key', document)
        self.assertNotIn('private_key', document)
