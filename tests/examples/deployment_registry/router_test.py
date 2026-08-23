from __future__ import annotations

import base64
import json
import unittest
from datetime import datetime
from datetime import timedelta
from datetime import timezone
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
from examples.deployment_registry import signing
from examples.deployment_registry.enrollments import (
    enforce_enrollment_exchange_rate_limit,
)
from examples.deployment_registry.enrollments import (
    enrollment_code_verifier_hash,
)
from examples.deployment_registry.enrollments import EnrollmentExchangeResult
from examples.deployment_registry.enrollments import provision_enrollment_code
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

    async def test_rate_limit_batches_counters_and_returns_retry_delay(
        self,
    ) -> None:
        """Rate limiting batches anonymised counters in one pipeline call.

        The result retains the retry duration of the exceeded counter.
        """
        pipeline = MagicMock()
        pipeline.eval = MagicMock()
        pipeline.execute = AsyncMock(return_value=[(2, 300), (6, 0)])
        pipeline.__aenter__ = AsyncMock(return_value=pipeline)
        pipeline.__aexit__ = AsyncMock(return_value=False)
        redis = MagicMock()
        redis.pipeline.return_value = pipeline

        retry_after = await enforce_enrollment_exchange_rate_limit(
            redis,
            '192.0.2.10',
            'a' * 64,
            maximum=5,
            window_seconds=300,
        )

        self.assertEqual(retry_after, 1)
        self.assertEqual(pipeline.eval.call_count, 2)
        pipeline.execute.assert_awaited_once()
        with self.assertRaises(ValueError):
            await enforce_enrollment_exchange_rate_limit(
                redis,
                None,
                'a' * 64,
                maximum=0,
                window_seconds=300,
            )

    async def test_provision_validates_inputs_and_stages_verifier_only(
        self,
    ) -> None:
        """Provisioning checks active deployments and stages no raw secret.

        Only the HMAC verifier is written to the database row.
        """
        now = datetime(2026, 8, 23, tzinfo=timezone.utc)
        db = MagicMock()
        db.scalar = AsyncMock(return_value=SimpleNamespace(id=_DEPLOYMENT_ID))

        with (
            patch(
                'examples.deployment_registry.enrollments.utc_now',
                return_value=now,
            ),
            patch(
                (
                    'examples.deployment_registry.enrollments.secrets.'
                    'token_urlsafe'
                ),
                return_value='raw-enrollment-code',
            ),
            patch(
                'examples.deployment_registry.enrollments.secrets.token_bytes',
                return_value=b'\x00' * 16,
            ),
        ):
            provisioned = await provision_enrollment_code(
                db,
                _DEPLOYMENT_ID,
                now + timedelta(minutes=5),
                'operator',
                'p' * 32,
                _TENANT_ID,
            )

        self.assertEqual(provisioned.raw_code, 'raw-enrollment-code')
        self.assertNotEqual(
            provisioned.enrollment.code_verifier_hash,
            provisioned.raw_code,
        )
        db.add.assert_called_once_with(provisioned.enrollment)

        with self.assertRaises(ValueError):
            await provision_enrollment_code(
                db,
                _DEPLOYMENT_ID,
                now + timedelta(minutes=5),
                '',
                'p' * 32,
            )

        with (
            patch(
                'examples.deployment_registry.enrollments.utc_now',
                return_value=now,
            ),
            patch(
                (
                    'examples.deployment_registry.enrollments.secrets.'
                    'token_urlsafe'
                ),
                return_value='raw-enrollment-code',
            ),
            patch(
                'examples.deployment_registry.enrollments.secrets.token_bytes',
                return_value=b'\x00' * 16,
            ),
        ):
            db.scalar = AsyncMock(return_value=None)
            with self.assertRaises(ValueError):
                await provision_enrollment_code(
                    db,
                    _DEPLOYMENT_ID,
                    now + timedelta(minutes=5),
                    'operator',
                    'p' * 32,
                )

    async def test_enrollment_service_rejects_terminal_and_invalid_states(
        self,
    ) -> None:
        """Enrollment service handles bad peppers, quotas, and terminal rows.

        Invalid, expired, and unsafe states never redeem a deployment.
        """
        with self.assertRaises(ValueError):
            enrollment_code_verifier_hash('raw-code', 'short-pepper')

        pipeline = MagicMock()
        pipeline.eval = MagicMock()
        pipeline.execute = AsyncMock(return_value=[(1, 300), (2, 300)])
        pipeline.__aenter__ = AsyncMock(return_value=pipeline)
        pipeline.__aexit__ = AsyncMock(return_value=False)
        redis = MagicMock()
        redis.pipeline.return_value = pipeline
        self.assertIsNone(
            await enforce_enrollment_exchange_rate_limit(
                redis,
                None,
                'a' * 64,
                maximum=5,
                window_seconds=300,
            ),
        )

        transaction = MagicMock()
        transaction.__aenter__ = AsyncMock(return_value=transaction)
        transaction.__aexit__ = AsyncMock(return_value=False)
        db = MagicMock()
        db.begin.return_value = transaction
        db.scalar = AsyncMock(return_value=None)
        invalid = await redeem_enrollment_code(db, verifier_hash='a' * 64)
        self.assertEqual(invalid.status, 'invalid')

        expired = SimpleNamespace(
            redeemed_at=None,
            revoked_at=None,
            expires_at=utc_now() - timedelta(seconds=1),
            deployment_id=_DEPLOYMENT_ID,
        )
        db.scalar = AsyncMock(return_value=expired)
        terminal = await redeem_enrollment_code(db, verifier_hash='a' * 64)
        self.assertEqual(terminal.status, 'terminal')

        active = SimpleNamespace(
            redeemed_at=None,
            revoked_at=None,
            expires_at=utc_now() + timedelta(minutes=1),
            deployment_id=_DEPLOYMENT_ID,
        )
        db.scalar = AsyncMock(side_effect=[active, None])
        inactive_deployment = await redeem_enrollment_code(
            db,
            verifier_hash='a' * 64,
        )
        self.assertEqual(inactive_deployment.status, 'terminal')

        current = utc_now()
        with self.assertRaises(ValueError):
            await provision_enrollment_code(
                db,
                _DEPLOYMENT_ID,
                current,
                'operator',
                'p' * 32,
            )


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

    def test_signing_rejects_invalid_lifetime_revision_and_key_material(
        self,
    ) -> None:
        """Registry signing refuses invalid public configuration or key input.

        Invalid lifetimes, timestamps, and key material never reach signing.
        """
        deployment = cast(
            Deployment,
            SimpleNamespace(
                id=_DEPLOYMENT_ID,
                tenant_id=_TENANT_ID,
                api_base_url='https://api.example.com/hazard/api',
                config_revision=1,
            ),
        )
        for private_key_pem, ttl_seconds, issued_at in (
            ('key', 0, 1),
            ('key', 60, -1),
            ('', 60, 1),
        ):
            with self.subTest(
                private_key_pem=private_key_pem,
                ttl_seconds=ttl_seconds,
                issued_at=issued_at,
            ):
                with self.assertRaises(ValueError):
                    build_registry_document(
                        deployment,
                        private_key_pem=private_key_pem,
                        key_id='key-1',
                        ttl_seconds=ttl_seconds,
                        issued_at=issued_at,
                    )

        deployment.config_revision = 0
        with self.assertRaises(ValueError):
            build_registry_document(
                deployment,
                private_key_pem='key',
                key_id='key-1',
                ttl_seconds=60,
                issued_at=1,
            )

    def test_signing_rejects_non_ed25519_signature_lengths(self) -> None:
        """The Registry accepts only private keys with Ed25519 signatures.

        Other signature sizes cannot be exposed to native clients.
        """
        deployment = cast(
            Deployment,
            SimpleNamespace(
                id=_DEPLOYMENT_ID,
                tenant_id=_TENANT_ID,
                api_base_url='https://api.example.com/hazard/api',
                config_revision=1,
            ),
        )
        private_key = MagicMock()
        private_key.sign.return_value = b'not-ed25519'
        with patch.object(
            signing,
            'load_pem_private_key',
            return_value=private_key,
        ):
            with self.assertRaises(ValueError):
                build_registry_document(
                    deployment,
                    private_key_pem='private-key',
                    key_id='key-1',
                    ttl_seconds=60,
                    issued_at=1,
                )
