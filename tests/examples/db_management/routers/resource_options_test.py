from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException
from fastapi import Response

from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.db_management.routers import (
    deployment_enrollment_codes as invitations,
)
from examples.db_management.routers import streams as route
from examples.db_management.routers import users
from examples.db_management.services import invitation_policy
from examples.db_management.services import role_policy
from examples.db_management.services import site_access
from examples.violation_records import record_policy as resources
from examples.YOLO_server_api import model_registry as registry


class ContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        (self.path / 'ppe.pt').write_bytes(b'immutable-model')
        self.model = {
            'id': 'ppe-stream-v1',
            'display_name': 'PPE',
            'version': 'v1',
            'artifact': str(self.path / 'ppe.pt'),
            'sha256': hashlib.sha256(b'immutable-model').hexdigest(),
            'capabilities': ['image', 'stream'],
            'tenant_ids': ['t1'],
            'site_ids': [123],
            'classes': [
                {
                    'id': 7,
                    'code': 'helmet',
                    'display_name': '安全帽',
                    'color': '#4CAF50',
                },
            ],
        }
        self.data = {
            'revision': 'r1',
            'default_id': self.model['id'],
            'models': [self.model],
        }
        self.write_registry()

        def rows():
            data = json.loads((self.path / 'models.json').read_text())
            return [
                {
                    'model_key': m['id'],
                    'definition': m,
                    'is_default': m['id'] == data['default_id'],
                }
                for m in data['models']
            ]

        patcher = patch.object(
            registry, '_read_catalog_rows', side_effect=rows,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.me = SimpleNamespace(
            id=8, username='admin', role='admin', group_id=2, tenant_id='t1',
        )
        self.credentials = JwtAuthorizationCredentials(
            subject={'role': 'admin', 'tenant_id': 't1'}, token='access',
        )

    def write_registry(self):
        (self.path / 'models.json').write_text(json.dumps(self.data))

    async def options(self, scope, target=None):
        if scope == 'stream_models':
            return await route.get_stream_models(
                target, Response(), self.me, AsyncMock(),
            )
        return await invitations.get_invitation_expiry(
            MagicMock(),
            Response(),
            self.me,
            AsyncMock(),
        )

    async def test_stream_catalog_scoped_and_revalidated(self):
        with patch.object(route, 'require_site', AsyncMock()):
            first = await self.options('stream_models', 123)
            self.assertEqual(first.default_id, 'ppe-stream-v1')
            self.assertEqual(first.options[0].id, 'ppe-stream-v1')
            self.model['enabled'] = False
            self.write_registry()
            second = await self.options('stream_models', 123)
            self.assertEqual(second.options, [])
            self.assertIsNone(second.default_id)
            self.assertNotEqual(first.revision, second.revision)
        with self.assertRaises(HTTPException) as error:
            registry.require_model(
                'ppe-stream-v1', 'v1', 't1', 'admin', 'stream', 123,
            )
        self.assertEqual(error.exception.status_code, 409)

    async def test_model_versions_and_tenants_are_enforced(self):
        for version, tenant, status in [('v0', 't1', 409), ('v1', 't2', 403)]:
            with self.subTest(version=version, tenant=tenant):
                with self.assertRaises(HTTPException) as error:
                    registry.require_model(
                        'ppe-stream-v1', version, tenant, 'admin', 'image',
                    )
                self.assertEqual(error.exception.status_code, status)
        (self.path / 'ppe.pt').write_bytes(b'replaced')
        with self.assertRaises(HTTPException) as error:
            registry.require_model(
                'ppe-stream-v1', 'v1', 't1', 'admin', 'image',
            )
        self.assertEqual(error.exception.status_code, 409)

    async def test_registry_outage_is_not_empty_success(self):
        (self.path / 'models.json').write_text('{broken')
        with patch.object(route, 'require_site', AsyncMock()):
            with self.assertRaises(HTTPException) as error:
                await self.options('stream_models', 123)
        self.assertEqual(error.exception.status_code, 503)

    async def test_registry_rejects_duplicate_and_numeric_codes(self):
        for classes in [
            [self.model['classes'][0]] * 2,
            [{'id': 0, 'code': '123', 'display_name': 'x'}],
        ]:
            with self.subTest(classes=classes):
                self.model['classes'] = classes
                self.write_registry()
                with self.assertRaises(HTTPException) as error:
                    registry.load_registry()
                self.assertEqual(error.exception.status_code, 503)

    async def test_roles_cannot_be_forged_or_cross_tenant(self):
        target = SimpleNamespace(
            username='member', role='user', group_id=2, tenant_id='t1',
        )
        self.assertEqual(
            role_policy.assignable_roles(self.me, target), ['user', 'guest'],
        )
        for role in ('admin', 'super_admin', 'invented'):
            with self.assertRaises(HTTPException):
                role_policy.validate_role(self.me, role, target)
        target.tenant_id = 't2'
        with self.assertRaises(HTTPException):
            role_policy.assignable_roles(self.me, target)

    async def test_review_transitions_depend_on_current_state(self):
        record = SimpleNamespace(is_flagged=True, review_status='pending')
        self.assertEqual(
            resources.review_actions(self.me, record),
            ['resolved', 'dismissed'],
        )
        record.review_status = 'resolved'
        self.assertEqual(
            resources.review_actions(self.me, record), ['pending'],
        )
        self.me.role = 'user'
        self.assertEqual(resources.review_actions(self.me, record), [])

    async def test_invitation_policy_changes_are_enforced(self):
        path = self.path / 'invitations.json'
        path.write_text(
            json.dumps(
                {
                    't1': {
                        'max_minutes': 120,
                        'minutes': [45, 90],
                        'default_minutes': 45,
                    },
                },
            ),
        )
        with patch.dict(os.environ, {'INVITATION_POLICY_PATH': str(path)}):
            invitation_policy.validate_invitation('t1', 90)
            with self.assertRaises(HTTPException):
                invitation_policy.validate_invitation('t1', 60)
            with patch.object(
                invitations,
                'require_tenant_deployment_administrator',
                AsyncMock(),
            ):
                result = await self.options('invitation_expiry')
            self.assertEqual(result.default_id, '45')
            self.assertEqual(result.policy, {'max_minutes': 120})

    async def test_target_membership_rechecked(self):
        db = AsyncMock()
        db.get.return_value = SimpleNamespace(id=123)
        db.scalar.return_value = None
        with self.assertRaises(HTTPException) as error:
            await site_access.require_site(db, self.me, 123)
        self.assertEqual(error.exception.status_code, 403)
        db.get.return_value = None
        with self.assertRaises(HTTPException) as error:
            await site_access.require_site(db, self.me, 123)
        self.assertEqual(error.exception.status_code, 404)

    async def test_detection_catalog_and_pinned_inference(self):
        from io import BytesIO
        from fastapi import UploadFile
        from examples.YOLO_server_api import routers
        from examples.YOLO_server_api.schemas import DetectionRequest

        response = routers.list_detection_models(Response(), self.credentials)
        self.assertEqual(response.models[0].classes[0].id, 7)
        self.assertEqual(response.models[0].classes[0].color, '#4CAF50')
        self.assertEqual(response.models[0].version, 'v1')
        inference = AsyncMock(
            return_value=([[1, 2, 3, 4, 0.8, 7]], {'inference': 0, 'post': 0}),
        )
        loaded = object()
        with (
            patch.object(
                routers.model_loader,
                'get_registry_model',
                return_value=loaded,
            ) as loader,
            patch.object(routers, 'run_detection_from_bytes', inference),
        ):
            result = await routers.detect(
                DetectionRequest(
                    model='ppe-stream-v1',
                    model_version='v1',
                    image=UploadFile(
                        filename='image.jpg', file=BytesIO(b'image'),
                    ),
                ),
                self.credentials,
                3,
            )
            self.assertEqual(result[0][-1], 7)
            self.assertIs(inference.await_args.args[1], loaded)
            loader.assert_called_once()
            with self.assertRaises(HTTPException) as error:
                await routers.detect(
                    DetectionRequest(
                        model='ppe-stream-v1',
                        model_version='old',
                        image=UploadFile(
                            filename='image.jpg', file=BytesIO(b'image'),
                        ),
                    ),
                    self.credentials,
                    3,
                )
            self.assertEqual(error.exception.status_code, 409)
            self.assertEqual(loader.call_count, 1)

    async def test_stream_mutation_rejects_disabled_selection(self):
        from examples.db_management.routers import streams
        from examples.db_management.schemas.stream_config import (
            StreamConfigCreate,
        )

        self.model['enabled'] = False
        self.write_registry()
        site = SimpleNamespace(id=123, groups=[SimpleNamespace(id=2)])
        db = AsyncMock()
        with (
            patch.object(
                streams, '_get_site_or_404', AsyncMock(return_value=site),
            ),
            patch.object(
                streams, 'require_site', AsyncMock(return_value=site),
            ),
            patch.object(
                streams,
                'get_group_stream_limit',
                AsyncMock(return_value=(0, 10)),
            ),
        ):
            with self.assertRaises(HTTPException) as error:
                await streams.endpoint_create_stream_config(
                    StreamConfigCreate(
                        site_id=123,
                        stream_name='camera',
                        video_url='rtsp://camera',
                        model_key='ppe-stream-v1',
                    ),
                    db,
                    self.me,
                )
        self.assertEqual(error.exception.status_code, 409)
        db.commit.assert_not_awaited()

    async def test_feedback_rejects_unknown_annotation_category(self):
        from examples.violation_records import (
            violation_review_service as service,
        )
        from examples.violation_records.schemas import ViolationFeedbackCreate

        record = SimpleNamespace(
            id=1,
            site='site',
            model_id='old',
            model_version='old',
            model_classes=[
                {'id': 0, 'code': 'vehicle', 'display_name': 'Vehicle'},
            ],
        )
        db = AsyncMock()
        db.scalar.return_value = record
        db.get.return_value = self.me
        credentials = JwtAuthorizationCredentials(
            subject={
                'username': 'admin',
                'user_id': 8,
                'tenant_id': 't1',
            },
        )
        with (
            patch.object(
                service.user_service,
                'get_effective_site_names',
                AsyncMock(return_value=['site']),
            ),
            patch.object(service, 'authorize_violation', AsyncMock()),
        ):
            with self.assertRaises(HTTPException) as error:
                await service.submit_violation_feedback(
                    1,
                    ViolationFeedbackCreate(
                        type='false_negative',
                        corrected_label='forged-category',
                        corrected_bbox=[0, 0, 1, 1],
                    ),
                    db,
                    credentials,
                )
        self.assertEqual(error.exception.status_code, 422)
        db.commit.assert_not_awaited()

    async def test_resource_routes_and_removed_gateway(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from examples.db_management.app import safe_http_exception

        app = FastAPI()
        for router in (route.router, users.router, invitations.router):
            app.include_router(router)
        app.add_exception_handler(HTTPException, safe_http_exception)
        app.dependency_overrides[route.get_current_user] = lambda: self.me
        app.dependency_overrides[route.get_db] = lambda: AsyncMock()
        with TestClient(app) as client:
            result = client.get('/users/assignable-roles')
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json()['schema_version'], 1)
            self.assertEqual(
                result.headers['Cache-Control'], 'private, no-store',
            )
            for scope in (
                'stream_models',
                'assignable_roles',
                'invitation_expiry',
                'upload_chat',
                'violation_feedback_labels',
                'violation_review_actions',
            ):
                self.assertEqual(
                    client.get(
                        '/client-options', params={'scope': scope},
                    ).status_code,
                    404,
                )
            with patch.object(route, 'require_site', AsyncMock()):
                result = client.get('/sites/123/stream-models')
                self.assertEqual(result.status_code, 200)
                self.assertEqual(
                    result.json()['options'][0]['id'], 'ppe-stream-v1',
                )
                (self.path / 'models.json').write_text('{broken')
                self.assertEqual(
                    client.get('/sites/123/stream-models').status_code, 503,
                )
            with patch.object(
                invitations,
                'require_tenant_deployment_administrator',
                AsyncMock(),
            ):
                result = client.get(
                    '/deployment-enrollment-codes/expiry-options',
                )
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.json()['policy']['max_minutes'], 1440)

    async def test_violation_upload_validates_and_saves_model_snapshot(self):
        from io import BytesIO
        from fastapi import UploadFile
        from examples.violation_records import (
            violation_upload_service as service,
        )

        credentials = JwtAuthorizationCredentials(
            subject={
                'username': 'admin',
                'user_id': 8,
                'tenant_id': 't1',
                'role': 'admin',
            },
        )
        db = AsyncMock()
        db.get.return_value = self.me
        db.scalar.return_value = 123
        with (
            patch.object(
                service.user_service,
                'get_effective_site_names',
                AsyncMock(return_value=['site']),
            ),
            patch.object(service, 'require_site', AsyncMock()),
            patch.object(
                service.violation_manager,
                'save_violation',
                AsyncMock(return_value=9),
            ) as save,
        ):
            result = await service.upload_violation(
                'site',
                'camera',
                None,
                None,
                '[[0,0,1,1,0.9,7,123]]',
                None,
                None,
                UploadFile(file=BytesIO(b'image')),
                db,
                credentials,
                model_id='ppe-stream-v1',
                model_version='v1',
            )
            self.assertEqual(result.violation_id, 9)
            self.assertEqual(save.await_args.kwargs['model_version'], 'v1')
            self.assertEqual(
                save.await_args.kwargs['model_classes'], self.model['classes'],
            )
            with self.assertRaises(HTTPException) as error:
                await service.upload_violation(
                    'site',
                    'camera',
                    None,
                    None,
                    '[[0,0,1,1,0.9,0,123]]',
                    None,
                    None,
                    UploadFile(file=BytesIO(b'image')),
                    db,
                    credentials,
                    model_id='ppe-stream-v1',
                    model_version='v1',
                )
            self.assertEqual(error.exception.status_code, 422)
            self.assertEqual(save.await_count, 1)
