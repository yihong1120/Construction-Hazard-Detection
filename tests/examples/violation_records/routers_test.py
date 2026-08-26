from __future__ import annotations

import unittest
from datetime import datetime
from datetime import timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.db_management.schemas.auth import AccessTokenSubject
from examples.violation_records import violation_query_service
from examples.violation_records import violation_review_service
from examples.violation_records.routers import router
from examples.violation_records.schemas import ViolationList
from examples.violation_records.schemas import ViolationListItem


def _credentials() -> JwtAuthorizationCredentials:
    """Return valid credentials for a router delegation test."""
    return JwtAuthorizationCredentials(
        subject=cast(
            AccessTokenSubject,
            {
                'username': 'reviewer',
                'user_id': 1,
                'role': 'admin',
                'jti': 'router-test',
                'features': [],
            },
        ),
    )


class TestViolationRoutes(unittest.TestCase):
    """Verify canonical violation routes delegate to their service modules."""

    def setUp(self) -> None:
        """Create an application with deterministic authentication.

        Dependencies:
            Overrides for database and token validation.
        """
        app = FastAPI()
        app.include_router(router, prefix='/api')
        app.dependency_overrides[get_db] = lambda: SimpleNamespace()
        app.dependency_overrides[jwt_access] = _credentials
        self.client = TestClient(app)
        self.paths = {route.path for route in router.routes}

    def test_filter_options_has_a_frontend_compatibility_route(self) -> None:
        """The deployed Flutter artifact retains the legacy filter path."""
        self.assertIn('/violations/filter-options', self.paths)
        self.assertIn('/filter-options', self.paths)
        self.assertIn('/violations/analytics', self.paths)
        self.assertNotIn('/analytics', self.paths)
        self.assertNotIn(
            '/hazard/api/detection/violations/analytics',
            self.paths,
        )

    def test_list_route_delegates_to_the_compact_query_service(self) -> None:
        """A list response contains compact list items only."""
        timestamp = datetime(2026, 8, 23, tzinfo=timezone.utc)
        response = ViolationList(
            items=[
                ViolationListItem(
                    id=1,
                    site_name='Site A',
                    stream_name='Camera A',
                    detection_time=timestamp,
                    thumbnail_url='/get_violation_thumbnail?image_path=a.jpg',
                ),
            ],
        )
        with patch.object(
            violation_query_service,
            'get_violations',
            new=AsyncMock(return_value=response),
        ) as get_violations:
            result = self.client.get('/api/violations')

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()['items'][0]['id'], 1)
        self.assertNotIn('detections', result.json()['items'][0])
        get_violations.assert_awaited_once()

    def test_compatibility_filter_route_allows_unselected_site(self) -> None:
        """Initial page load can populate filters before a site is selected."""
        with patch.object(
            violation_query_service,
            'get_violation_filter_options',
            new=AsyncMock(
                return_value={'cameras': [], 'violation_types': []},
            ),
        ) as get_filter_options:
            result = self.client.get('/api/filter-options')

        self.assertEqual(result.status_code, 200)
        get_filter_options.assert_awaited_once()
        filter_call = get_filter_options.await_args
        self.assertIsNotNone(filter_call)
        assert filter_call is not None
        self.assertIsNone(filter_call.args[0])

    def test_review_route_delegates_to_review_service(self) -> None:
        """Review routes call the dedicated workflow service."""
        with patch.object(
            violation_review_service,
            'get_violation_review_audit_log',
            new=AsyncMock(return_value=[]),
        ) as get_audit_log:
            result = self.client.get('/api/violations/1/audit-log')

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json(), [])
        get_audit_log.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
