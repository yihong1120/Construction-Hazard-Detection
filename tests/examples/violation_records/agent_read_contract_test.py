from __future__ import annotations

import unittest
from datetime import datetime
from datetime import timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.testclient import TestClient

from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.db_management.schemas.auth import AccessTokenSubject
from examples.violation_records import violation_query_service
from examples.violation_records import violation_review_service
from examples.violation_records.routers import router
from examples.violation_records.schemas import SiteOut
from examples.violation_records.schemas import ViolationList


def _database() -> SimpleNamespace:
    return SimpleNamespace()


def _credentials() -> JwtAuthorizationCredentials:
    return JwtAuthorizationCredentials(
        subject=cast(
            AccessTokenSubject,
            {
                'username': 'gateway-reader',
                'user_id': 7,
                'role': 'user',
                'jti': 'oidc:test:reader',
                'features': [],
                'tenant_id': '11111111-1111-1111-1111-111111111111',
            },
        ),
        payload={'sub': 'reader-oidc-subject'},
        token='synthetic-user-oidc-token',
    )


class AgentReadContractTest(unittest.TestCase):
    """Protect the existing per-user API used by the gateway adapter."""

    def setUp(self) -> None:
        self.app = FastAPI()
        self.app.include_router(router)
        self.app.dependency_overrides[get_db] = _database
        self.app.dependency_overrides[jwt_access] = _credentials
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)

    def test_cursor_page_preserves_scope_and_does_not_invent_total(self):
        with patch.object(
            violation_query_service,
            'get_violations',
            AsyncMock(
                return_value=ViolationList(
                    items=[], next_cursor='opaque-next-page', has_more=True,
                ),
            ),
        ) as query:
            response = self.client.get(
                '/violations',
                params={
                    'site_id': 4,
                    'keyword': 'hardhat',
                    'limit': 2,
                    'cursor': 'opaque-current-page',
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                'items': [],
                'next_cursor': 'opaque-next-page',
                'has_more': True,
            },
        )
        args = query.await_args.args
        self.assertEqual(args[2].token, 'synthetic-user-oidc-token')
        self.assertEqual(args[2].payload['sub'], 'reader-oidc-subject')
        self.assertEqual(query.await_args.kwargs['filters'].site_id, 4)
        self.assertEqual(
            query.await_args.kwargs['filters'].cursor, 'opaque-current-page',
        )
        self.assertEqual(query.await_args.kwargs['limit'], 2)

    def test_sites_use_the_same_verified_user_identity(self):
        timestamp = datetime(2026, 11, 1, tzinfo=timezone.utc)
        with patch.object(
            violation_query_service,
            'get_my_sites',
            AsyncMock(
                return_value=[
                    SiteOut(
                        id=4,
                        name='Authorized Site',
                        created_at=timestamp,
                        updated_at=timestamp,
                    ),
                ],
            ),
        ) as query:
            response = self.client.get('/my_sites')

        self.assertEqual(response.status_code, 200)
        self.assertEqual([item['id'] for item in response.json()], [4])
        self.assertEqual(query.await_args.args[1].subject['user_id'], 7)

    def test_anonymous_requests_do_not_reach_domain_queries(self):
        del self.app.dependency_overrides[jwt_access]
        with patch.object(
            violation_query_service, 'get_my_sites', AsyncMock(),
        ) as query:
            response = self.client.get('/my_sites')

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers['www-authenticate'], 'Bearer')
        query.assert_not_awaited()

    def test_forbidden_detail_is_not_retried_under_another_identity(self):
        with patch.object(
            violation_review_service,
            'get_single_violation',
            AsyncMock(
                side_effect=HTTPException(403, 'No access to this violation'),
            ),
        ) as query:
            response = self.client.get('/violations/42')

        self.assertEqual(response.status_code, 403)
        self.assertEqual(
            response.json(), {'detail': 'No access to this violation'},
        )
        query.assert_awaited_once()
        self.assertEqual(query.await_args.args[3].subject['user_id'], 7)
