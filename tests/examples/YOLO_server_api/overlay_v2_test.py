from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import AsyncMock
from unittest.mock import patch

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.testclient import TestClient

from examples.YOLO_server_api import routers
from examples.YOLO_server_api.model_registry import ModelClass
from examples.YOLO_server_api.overlay_response import build_overlay_response
from examples.YOLO_server_api.overlay_response import calculate_regions


class OverlayV2Test(TestCase):
    def setUp(self):
        self.entry = SimpleNamespace(
            id='custom',
            version='v1',
            classes=[
                ModelClass(id=42, code='safety_cone', display_name='交通錐'),
                ModelClass(id=3, code='utility_pole', display_name='電桿'),
            ],
        )
        self.image = cv2.imencode(
            '.png', np.zeros((200, 400, 3), dtype=np.uint8),
        )[1].tobytes()
        self.rows = [[100, 50, 200, 150, 0.9, 42]]
        self.app = FastAPI()
        self.app.include_router(routers.detection_router)
        self.app.dependency_overrides[routers.jwt_access] = lambda: (
            SimpleNamespace(subject={'tenant_id': 't1', 'role': 'user'})
        )
        self.app.dependency_overrides[routers.rate_limiter_service] = lambda: (
            99
        )

    def test_explicit_opt_out_keeps_cones_visible(self):
        with patch(
            'examples.YOLO_server_api.overlay_response.calculate_regions',
        ) as cluster:
            result = build_overlay_response(
                self.rows, self.entry, self.image, False, 'objects',
            )
        cluster.assert_not_called()
        box = result.overlay_objects[0]
        self.assertEqual(
            box.bbox.model_dump(), {'x': 0.25, 'y': 0.25, 'w': 0.25, 'h': 0.5},
        )
        self.assertTrue(box.show_box)
        self.assertEqual(box.display_name, '交通錐')
        self.assertEqual(
            result.region_status,
            {'cone': 'not_computed', 'pole': 'not_computed'},
        )

    def test_multiple_saved_region_shapes_and_region_view(self):
        rings = [
            [[0, 0], [100, 0], [0, 100]],
            [[200, 0], [300, 0], [300, 100]],
        ]
        with patch(
            'examples.YOLO_server_api.overlay_response.calculate_regions',
            return_value=(rings, []),
        ):
            result = build_overlay_response(
                self.rows, self.entry, self.image, True, 'regions',
            )
        self.assertEqual(len(result.overlay_regions), 2)
        self.assertEqual(result.overlay_regions[1].id, 'cone_1')
        self.assertEqual(
            result.region_status, {'cone': 'available', 'pole': 'empty'},
        )
        self.assertFalse(result.overlay_objects[0].show_box)

    def test_clustering_uses_codes_not_model_numeric_ids(self):
        # One pole produces its existing geometry buffer even with model ID 3.
        cones, poles = calculate_regions(
            [[10, 10, 20, 40, 0.9, 3]], self.entry.classes,
        )
        self.assertEqual(cones, [])
        self.assertTrue(poles)
        cones, poles = calculate_regions(
            [[10, 10, 20, 40, 0.9, 9]], self.entry.classes,
        )
        self.assertEqual((cones, poles), ([], []))

    def test_http_contract_v1_preserved_and_v2_opt_in(self):
        with (
            patch.object(
                routers,
                '_run_detection_request',
                AsyncMock(return_value=(self.rows, self.entry, self.image)),
            ) as run,
            TestClient(self.app) as client,
        ):
            files = {'image': ('image.png', self.image, 'image/png')}
            old = client.post('/detect', data={'model': 'custom'}, files=files)
            self.assertEqual(old.json(), self.rows)
            new = client.post(
                '/v2/detect',
                data={'model': 'custom', 'model_version': 'v1'},
                files=files,
            )
            self.assertEqual(new.status_code, 200)
            self.assertEqual(
                new.json()['region_status'], {
                    'cone': 'empty', 'pole': 'empty',
                },
            )
            self.assertFalse(new.json()['overlay_objects'][0]['show_box'])
            self.assertEqual(
                new.json()['overlay_coordinates']['bbox_format'], 'xywh',
            )
            self.assertEqual(new.json()['model_version'], 'v1')
            self.assertEqual(run.await_args.args[0].model_version, 'v1')
            regions = client.post(
                '/v2/detect',
                data={
                    'model': 'custom',
                    'compute_regions': 'true',
                    'view': 'regions',
                },
                files=files,
            )
            self.assertEqual(regions.status_code, 200)
            self.assertEqual(
                regions.json()['region_status'],
                {'cone': 'empty', 'pole': 'empty'},
            )
            self.assertFalse(regions.json()['overlay_objects'][0]['show_box'])
            before = run.await_count
            bad = client.post(
                '/v2/detect',
                data={
                    'model': 'custom',
                    'view': 'regions',
                    'compute_regions': 'false',
                },
                files=files,
            )
            self.assertEqual(bad.status_code, 422)
            self.assertEqual(run.await_count, before)

    def test_auth_denial_prevents_inference(self):
        def deny():
            raise HTTPException(401)

        self.app.dependency_overrides[routers.jwt_access] = deny
        with (
            patch.object(
                routers, '_run_detection_request', AsyncMock(),
            ) as run,
            TestClient(self.app) as client,
        ):
            response = client.post(
                '/v2/detect',
                data={'model': 'custom'},
                files={'image': ('x.png', self.image)},
            )
        self.assertEqual(response.status_code, 401)
        run.assert_not_awaited()

    def test_rate_limit_prevents_inference(self):
        def deny():
            raise HTTPException(429)

        self.app.dependency_overrides[routers.rate_limiter_service] = deny
        with (
            patch.object(
                routers, '_run_detection_request', AsyncMock(),
            ) as run,
            TestClient(self.app) as client,
        ):
            response = client.post(
                '/v2/detect',
                data={'model': 'custom'},
                files={'image': ('x.png', self.image)},
            )
        self.assertEqual(response.status_code, 429)
        run.assert_not_awaited()

    def test_overlay_fields_share_exact_contract_with_violations(self):
        from examples.shared.overlay_schemas import OverlayPayload
        from examples.YOLO_server_api.schemas import DetectionOverlayResponse
        from examples.violation_records.schemas import ViolationItem

        for name, field in OverlayPayload.model_fields.items():
            self.assertEqual(
                DetectionOverlayResponse.model_fields[name].annotation,
                ViolationItem.model_fields[name].annotation,
            )
            self.assertEqual(
                DetectionOverlayResponse.model_fields[name].default,
                ViolationItem.model_fields[name].default,
            )
