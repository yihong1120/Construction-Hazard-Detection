from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest import TestCase
from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import HTTPException

from examples.shared.class_colors import overlay_style
from examples.violation_records import violation_upload_service as uploads
from examples.violation_records.overlay_geometry import decode_polygons
from examples.violation_records.overlay_geometry import stored_regions
from examples.violation_records.schemas import OverlayRegion


class OverlayGeometryTest(TestCase):
    def test_saved_pixels_are_normalized_without_reclustering(self):
        raw = '[[[100,50],[300,50],[300,150],[100,50]]]'
        record = SimpleNamespace(cone_polygon_json=raw, pole_polygon_json='[]')
        regions, status = stored_regions(record, (400, 200), overlay_style())
        self.assertEqual(status, {'cone': 'available', 'pole': 'empty'})
        self.assertEqual(
            regions[0]['points'],
            [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.25]],
        )
        self.assertEqual(regions[0]['color'], '#FF4081')
        self.assertEqual(record.cone_polygon_json, raw)
        self.assertEqual(OverlayRegion.model_validate(regions[0]).kind, 'cone')

    def test_missing_invalid_and_unreadable_are_distinct(self):
        for raw, size, expected in (
            (None, (640, 360), 'not_recorded'),
            ('null', (640, 360), 'not_recorded'),
            ('[]', None, 'empty'),
            ('{bad', (640, 360), 'invalid'),
            ('[[[0,0],[1,0],[0,1]]]', None, 'image_unavailable'),
        ):
            record = SimpleNamespace(
                cone_polygon_json=raw, pole_polygon_json=None,
            )
            regions, status = stored_regions(record, size, overlay_style())
            self.assertEqual(status['cone'], expected)
            self.assertEqual(regions, [])

    def test_rejects_malformed_upload_geometry(self):
        for value in (
            {},
            [[1, 2]],
            [[[0, 0], [1, 1], [2, 2]]],
            [[[0, 0], [1, float('nan')], [2, 0]]],
            [[[0, 0], [True, 1], [2, 0]]],
        ):
            with self.assertRaises(ValueError):
                decode_polygons(json.dumps(value))

    def test_boundary_outside_image_is_preserved_for_canvas_clipping(self):
        record = SimpleNamespace(
            cone_polygon_json='[[[-10,0],[20,0],[0,20]]]',
            pole_polygon_json=None,
        )
        regions, _ = stored_regions(record, (100, 100), overlay_style())
        self.assertEqual(regions[0]['points'][0], [-0.1, 0])


class ClusterUploadTest(IsolatedAsyncioTestCase):
    async def test_invalid_boundary_is_rejected_before_saving_evidence(self):
        with (
            patch.object(
                uploads.user_service,
                'get_effective_site_names',
                AsyncMock(return_value=['Site']),
            ),
            patch.object(
                uploads.violation_manager, 'save_violation', AsyncMock(),
            ) as save,
        ):
            with self.assertRaises(HTTPException) as error:
                await uploads.upload_violation(
                    'Site',
                    'Cam',
                    None,
                    None,
                    None,
                    '[[[0,0],[1,1],[2,2]]]',
                    '[]',
                    None,
                    AsyncMock(),
                    SimpleNamespace(subject={'username': 'operator'}),
                )
        self.assertEqual(error.exception.status_code, 422)
        save.assert_not_awaited()
