from __future__ import annotations

import json
import time
import unittest

from fastapi import HTTPException
from starlette.requests import Request

from examples.auth.models import User
from examples.auth.models import USER_STATUS_ACTIVE
from examples.db_management.services import (
    legacy_password_migration_services as svc,
)
from tests.examples.auth.session_store_test import FakeRedis


def _request(
    body: bytes,
    headers: dict[str, str],
    host: str = '127.0.0.1',
) -> Request:
    """Create the loopback request made by the Keycloak custom provider."""
    sent = False

    async def receive() -> dict[str, object]:
        nonlocal sent
        if sent:
            return {'type': 'http.request', 'body': b'', 'more_body': False}
        sent = True
        return {'type': 'http.request', 'body': body, 'more_body': False}

    return Request(
        {
            'type': 'http',
            'method': 'POST',
            'path': '/auth/legacy-password/keycloak/verify',
            'scheme': 'http',
            'server': ('127.0.0.1', 8005),
            'client': (host, 43210),
            'headers': [
                (key.lower().encode('ascii'), value.encode('ascii'))
                for key, value in headers.items()
            ],
        },
        receive,
    )


class _Database:
    """Minimal user lookup/commit test double for the private protocol."""

    def __init__(self, user: User | None) -> None:
        self.user = user
        self.commit_count = 0

    async def scalar(self, _statement: object) -> User | None:
        return self.user

    async def commit(self) -> None:
        self.commit_count += 1


class LegacyPasswordMigrationServicesTest(unittest.IsolatedAsyncioTestCase):
    """Verify one-use migration without retaining a plaintext password."""

    def setUp(self) -> None:
        self.redis = FakeRedis()
        self.user = User(
            id=1,
            username='alice',
            password_hash='placeholder',
            status=USER_STATUS_ACTIVE,
        )
        self.user.set_password('legacy password to migrate')
        self.db = _Database(self.user)
        self._settings = {
            name: getattr(svc.settings, name)
            for name in (
                'legacy_password_migration_enabled',
                'legacy_password_migration_ttl_seconds',
                'native_social_exchange_shared_secret',
            )
        }
        svc.settings.legacy_password_migration_enabled = True
        svc.settings.legacy_password_migration_ttl_seconds = 30
        svc.settings.native_social_exchange_shared_secret = 'a' * 64

    def tearDown(self) -> None:
        for name, value in self._settings.items():
            setattr(svc.settings, name, value)

    def _signed_request(self, payload: dict[str, str]) -> Request:
        body = json.dumps(payload, separators=(',', ':')).encode('utf-8')
        timestamp = str(int(time.time()))
        return _request(
            body,
            {
                'X-Visionnaire-Timestamp': timestamp,
                'X-Visionnaire-Legacy-Signature': (
                    svc.legacy_password_migration_signature(timestamp, body)
                ),
            },
        )

    async def test_successful_proof_is_one_use_then_disables_legacy_hash(
        self,
    ) -> None:
        subject = 'keycloak-subject-1'
        verified = await svc.verify_legacy_password(
            self._signed_request(
                {
                    'keycloak_subject': subject,
                    'password': 'legacy password to migrate',
                },
            ),
            self.db,  # type: ignore[arg-type]
            self.redis,  # type: ignore[arg-type]
        )
        token = verified['migration_token']
        if not isinstance(token, str):
            self.fail('Migration acknowledgement was not a token')
        self.assertNotIn('password', verified)

        completed = await svc.complete_legacy_password_migration(
            self._signed_request(
                {
                    'keycloak_subject': subject,
                    'migration_token': token,
                },
            ),
            self.db,  # type: ignore[arg-type]
            self.redis,  # type: ignore[arg-type]
        )
        self.assertEqual(completed, {'status': 'completed'})
        self.assertEqual(self.user.password_hash, svc._DISABLED_PASSWORD_HASH)
        self.assertEqual(self.db.commit_count, 1)

        with self.assertRaises(HTTPException) as ctx:
            await svc.complete_legacy_password_migration(
                self._signed_request(
                    {
                        'keycloak_subject': subject,
                        'migration_token': token,
                    },
                ),
                self.db,  # type: ignore[arg-type]
                self.redis,  # type: ignore[arg-type]
            )
        self.assertEqual(ctx.exception.status_code, 401)

    async def test_invalid_password_and_non_loopback_never_issue_token(
        self,
    ) -> None:
        payload = {
            'keycloak_subject': 'keycloak-subject-1',
            'password': 'incorrect password',
        }
        with self.assertRaises(HTTPException) as ctx:
            await svc.verify_legacy_password(
                self._signed_request(payload),
                self.db,  # type: ignore[arg-type]
                self.redis,  # type: ignore[arg-type]
            )
        self.assertEqual(ctx.exception.status_code, 401)

        body = json.dumps(payload, separators=(',', ':')).encode('utf-8')
        timestamp = str(int(time.time()))
        request = _request(
            body,
            {
                'X-Visionnaire-Timestamp': timestamp,
                'X-Visionnaire-Legacy-Signature': (
                    svc.legacy_password_migration_signature(timestamp, body)
                ),
            },
            host='203.0.113.1',
        )
        with self.assertRaises(HTTPException) as ctx:
            await svc.verify_legacy_password(
                request,
                self.db,  # type: ignore[arg-type]
                self.redis,  # type: ignore[arg-type]
            )
        self.assertEqual(ctx.exception.status_code, 403)
