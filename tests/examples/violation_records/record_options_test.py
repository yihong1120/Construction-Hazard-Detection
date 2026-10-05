from __future__ import annotations

from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.testclient import TestClient

from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.violation_records import record_policy
from examples.violation_records import routers


class RecordOptionsTest(IsolatedAsyncioTestCase):
    def setUp(self):
        self.user = SimpleNamespace(
            id=1, tenant_id='t1', role='admin', status='active',
        )
        self.credentials = SimpleNamespace(
            subject={'user_id': 1, 'tenant_id': 't1'},
        )
        self.db = AsyncMock()
        self.db.get.return_value = self.user
        self.record = SimpleNamespace(
            model_id='original',
            model_version='v1',
            model_classes=[
                {
                    'id': 7,
                    'code': 'vehicle',
                    'display_name': '車輛',
                    'color': '#ab47bc',
                },
            ],
            is_flagged=True,
            review_status='pending',
        )

    def test_native_routes_return_contract_and_private_cache(self):
        app = FastAPI()
        app.include_router(routers.router)
        app.dependency_overrides[get_db] = lambda: self.db
        app.dependency_overrides[jwt_access] = lambda: self.credentials
        with patch.object(
            record_policy,
            'require_violation',
            AsyncMock(return_value=self.record),
        ) as access:
            with TestClient(app) as client:
                feedback = client.get(
                    '/violations/9/feedback-options?locale=zh-TW',
                )
                review = client.get('/violations/9/review-options?locale=en')
        self.assertEqual(feedback.status_code, 200)
        self.assertEqual(
            feedback.headers['cache-control'], 'private, no-store',
        )
        self.assertEqual(feedback.json()['schema_version'], 1)
        self.assertIn(
            {'id': 'vehicle', 'label': '車輛', 'color': '#AB47BC'},
            feedback.json()['options'],
        )
        self.assertNotIn(
            {'id': 'person', 'label': '人員'}, feedback.json()['options'],
        )
        self.assertIsNone(feedback.json()['default_id'])
        self.assertEqual(review.status_code, 200)
        self.assertEqual(
            [x['id'] for x in review.json()['options']],
            ['resolved', 'dismissed'],
        )
        access.assert_awaited_with(self.db, self.user, 9)

    async def test_missing_snapshot_disables_feedback_and_review(self):
        self.record.model_classes = None
        self.user.role = 'user'
        with patch.object(
            record_policy,
            'require_violation',
            AsyncMock(return_value=self.record),
        ):
            for kind in ('feedback', 'review'):
                result = await record_policy.authenticated_record_options(
                    self.db, self.credentials, 9, kind, 'zh-TW',
                )
                self.assertEqual(result['options'], [])
                self.assertIsNone(result['default_id'])

    async def test_tenant_and_resource_denials_are_preserved(self):
        self.user.tenant_id = 't2'
        with self.assertRaises(HTTPException) as raised:
            await record_policy.authenticated_record_options(
                self.db, self.credentials, 9, 'feedback', 'zh-TW',
            )
        self.assertEqual(raised.exception.status_code, 403)
        self.user.tenant_id = 't1'
        for code in (403, 404, 503):
            with patch.object(
                record_policy,
                'require_violation',
                AsyncMock(side_effect=HTTPException(code)),
            ):
                with self.assertRaises(HTTPException) as raised:
                    await record_policy.authenticated_record_options(
                        self.db, self.credentials, 9, 'feedback', 'zh-TW',
                    )
                self.assertEqual(raised.exception.status_code, code)
