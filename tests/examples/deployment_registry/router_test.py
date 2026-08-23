from __future__ import annotations

import base64
import json
import unittest
from datetime import timedelta
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
from fastapi import FastAPI
from fastapi.testclient import TestClient

from examples.auth.database import get_db
from examples.auth.models import Deployment
from examples.auth.models import utc_now
from examples.auth.redis_pool import get_redis_pool
from examples.deployment_registry import service as registry_service
from examples.deployment_registry.enrollments import (
    enrollment_code_verifier_hash,
)
from examples.deployment_registry.enrollments import EnrollmentExchangeResult
from examples.deployment_registry.enrollments import redeem_enrollment_code
from examples.deployment_registry.router import router
from examples.deployment_registry.signing import build_registry_document

"""Contract tests for the independent signed Deployment Registry."""


_TENANT_ID = UUID('11111111-1111-1111-1111-111111111111')
_DEPLOYMENT_ID = UUID('22222222-2222-2222-2222-222222222222')
_GET_PATH = f"/deployment-registry/v1/deployments/{_DEPLOYMENT_ID}"
_EXCHANGE_PATH = '/deployment-registry/v1/enrollments/exchange'
_CODE = 'a' * 32
_REQUEST_HEADERS = {
    'Content-Type': 'application/json',
}


def _base64url_decode(value: str) -> bytes:
    """Perform base64url decode.

    Args:
        value: Value used by this callable.

    Returns:
        The callable result.
    """
    return base64.urlsafe_b64decode(value + '=' * (-len(value) % 4))


class TestDeploymentEnrollmentExchangeRouter(unittest.TestCase):
    """Ensure exchange never turns into an authentication or profile API."""

    def setUp(self) -> None:
        """Perform setUp."""
        self.app = FastAPI()
        self.app.include_router(router)
        self.client = TestClient(self.app, base_url='https://api.example.com')
        self.db = MagicMock()
        self.redis = AsyncMock()

        async def override_get_db() -> Any:  # type: ignore[no-untyped-def]
            """Perform override get db.

            Returns:
                The callable result.
            """
            yield self.db

        async def override_get_redis() -> Any:  # type: ignore[no-untyped-def]
            """Perform override get redis.

            Returns:
                The callable result.
            """
            yield self.redis

        self.app.dependency_overrides[get_db] = override_get_db
        self.app.dependency_overrides[get_redis_pool] = override_get_redis
        self.settings_patch = patch.multiple(
            registry_service.settings,
            deployment_enrollment_code_pepper='p' * 32,
            deployment_enrollment_rate_limit_max=5,
            deployment_enrollment_rate_limit_window_seconds=300,
        )
        self.settings_patch.start()

    def tearDown(self) -> None:
        """Perform tearDown."""
        self.settings_patch.stop()
        self.client.close()

    def test_success_response_has_only_deployment_id(self) -> None:
        """Test success response has only deployment id."""
        with (
            patch.object(
                registry_service,
                'enforce_enrollment_exchange_rate_limit',
                AsyncMock(return_value=None),
            ),
            patch.object(
                registry_service,
                'redeem_enrollment_code',
                AsyncMock(
                    return_value=EnrollmentExchangeResult(
                        status='redeemed',
                        deployment_id=_DEPLOYMENT_ID,
                    ),
                ),
            ),
        ):
            response = self.client.post(
                _EXCHANGE_PATH,
                headers=_REQUEST_HEADERS,
                json={'enrollment_code': _CODE},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['content-type'], 'application/json')
        self.assertEqual(
            response.json(),
            {
                'deployment_id': str(_DEPLOYMENT_ID),
            },
        )

    def test_invalid_request_body_uses_standard_validation(self) -> None:
        """Invalid enrollment payloads use FastAPI's standard 422 response."""
        for payload in (
            {'enrollment_code': 'not valid'},
            {'enrollment_code': _CODE, 'unexpected': True},
        ):
            with self.subTest(payload=payload):
                response = self.client.post(
                    _EXCHANGE_PATH,
                    headers=_REQUEST_HEADERS,
                    json=payload,
                )
                self.assertEqual(response.status_code, 422)

    def test_invalid_terminal_and_rate_limited_codes_are_non_secret(
        self,
    ) -> None:
        """Test invalid terminal and rate limited codes are non secret."""
        with (
            patch.object(
                registry_service,
                'enforce_enrollment_exchange_rate_limit',
                AsyncMock(return_value=None),
            ),
            patch.object(
                registry_service,
                'redeem_enrollment_code',
                AsyncMock(
                    return_value=EnrollmentExchangeResult(status='invalid'),
                ),
            ),
        ):
            invalid = self.client.post(
                _EXCHANGE_PATH,
                headers=_REQUEST_HEADERS,
                json={'enrollment_code': _CODE},
            )
        self.assertEqual(invalid.status_code, 403)

        with (
            patch.object(
                registry_service,
                'enforce_enrollment_exchange_rate_limit',
                AsyncMock(return_value=None),
            ),
            patch.object(
                registry_service,
                'redeem_enrollment_code',
                AsyncMock(
                    return_value=EnrollmentExchangeResult(status='terminal'),
                ),
            ),
        ):
            terminal = self.client.post(
                _EXCHANGE_PATH,
                headers=_REQUEST_HEADERS,
                json={'enrollment_code': _CODE},
            )
        self.assertEqual(terminal.status_code, 410)

        with patch.object(
            registry_service,
            'enforce_enrollment_exchange_rate_limit',
            AsyncMock(return_value=61),
        ):
            limited = self.client.post(
                _EXCHANGE_PATH,
                headers=_REQUEST_HEADERS,
                json={'enrollment_code': _CODE},
            )
        self.assertEqual(limited.status_code, 429)
        self.assertEqual(limited.headers['retry-after'], '61')


class TestEnrollmentService(unittest.IsolatedAsyncioTestCase):
    """Verify verifier secrecy and the database-lock redemption contract."""

    async def test_redeem_marks_the_locked_row_once(self) -> None:
        """Test redeem marks the locked row once."""
        enrollment = SimpleNamespace(
            redeemed_at=None,
            revoked_at=None,
            expires_at=utc_now() + timedelta(minutes=1),
            deployment_id=_DEPLOYMENT_ID,
        )
        deployment = _DEPLOYMENT_ID
        transaction = MagicMock()
        transaction.__aenter__ = AsyncMock(return_value=transaction)
        transaction.__aexit__ = AsyncMock(return_value=False)
        db = MagicMock()
        db.begin.return_value = transaction
        db.scalar = AsyncMock(side_effect=[enrollment, deployment])

        result = await redeem_enrollment_code(
            db,
            verifier_hash='x' * 64,
        )

        self.assertEqual(result.status, 'redeemed')
        self.assertEqual(result.deployment_id, _DEPLOYMENT_ID)
        self.assertIsNotNone(enrollment.redeemed_at)
        self.assertEqual(db.scalar.await_count, 2)
        first_query = db.scalar.await_args_list[0].args[0]
        second_query = db.scalar.await_args_list[1].args[0]
        self.assertIsNotNone(first_query._for_update_arg)
        rendered_second_query = str(second_query)
        self.assertEqual(
            {table.name for table in second_query._for_update_arg.of},
            {'deployments', 'tenants'},
        )
        self.assertNotIn('LEFT OUTER JOIN', rendered_second_query)

    async def test_verifier_is_keyed_and_raw_code_is_not_the_database_value(
        self,
    ) -> None:
        """Test verifier is keyed and raw code is not the database value."""
        first = enrollment_code_verifier_hash(_CODE, 'p' * 32)
        second = enrollment_code_verifier_hash(_CODE, 'q' * 32)
        self.assertNotEqual(first, _CODE)
        self.assertNotEqual(first, second)
        self.assertEqual(len(first), 64)


class TestRegistrySigningContract(unittest.TestCase):
    """Ensure the signature bytes exactly match the mobile contract."""

    def test_signs_fixed_seven_fields_not_key_id_or_http_body(self) -> None:
        """Test signs fixed seven fields not key id or http body."""
        private_key = Ed25519PrivateKey.generate()
        pem = private_key.private_bytes(
            Encoding.PEM,
            PrivateFormat.PKCS8,
            NoEncryption(),
        ).decode('ascii')
        deployment = cast(
            Deployment,
            SimpleNamespace(
                id=_DEPLOYMENT_ID,
                tenant_id=_TENANT_ID,
                api_base_url='https://api.example.com/hazard/api',
                config_revision=1,
            ),
        )
        document = build_registry_document(
            deployment,
            private_key_pem=pem,
            key_id='registry-ed25519-2026-01',
            ttl_seconds=60,
            issued_at=123,
        )

        expected = (
            b'{"api_base_url":"https://api.example.com/hazard/api",'
            b'"config_revision":1,'
            b'"deployment_id":"22222222-2222-2222-2222-222222222222",'
            b'"expires_at":183,"issued_at":123,"schema_version":1,'
            b'"tenant_id":"11111111-1111-1111-1111-111111111111"}'
        )
        signed_document = document.copy()
        del signed_document['key_id']
        del signed_document['signature']
        signed = json.dumps(
            signed_document,
            ensure_ascii=False,
            separators=(',', ':'),
            sort_keys=True,
        ).encode('utf-8')
        self.assertEqual(signed, expected)
        self.assertNotIn(b'key_id', signed)
        private_key.public_key().verify(
            _base64url_decode(str(document['signature'])),
            signed,
        )
