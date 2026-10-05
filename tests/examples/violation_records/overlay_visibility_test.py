from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from examples.streaming_web import overlay_renderer as renderer
from examples.violation_records.feedback_policy import validate_feedback_label
from examples.violation_records.record_policy import historical_classes
from examples.violation_records.schemas import FeedbackDetectionItem
from examples.violation_records.schemas import ViolationFeedbackCreate
from examples.violation_records.violation_services import (
    _overlay_objects_from_feedback,
)
from tests.model_fixtures import CLASSES


class OverlayVisibilityTest(TestCase):
    def test_cones_remain_in_response_and_feedback_but_hidden_in_normal_view(
        self,
    ):
        record = SimpleNamespace(model_classes=CLASSES)
        metadata = [c.model_dump() for c in historical_classes(record)]
        detections = [
            FeedbackDetectionItem(
                id=f'det_{i}', label=code, bbox=[0, 0, 0.5, 0.5],
            )
            for i, code in enumerate(('safety_cone', 'person', 'class-99'))
        ]
        objects = _overlay_objects_from_feedback(
            detections, [], None, metadata,
        )
        self.assertEqual(len(objects), 3)
        cone = objects[0]
        self.assertEqual(cone.display_name, '交通錐')
        self.assertEqual(cone.color, '#FF5722')
        self.assertFalse(cone.show_box)
        self.assertFalse(cone.show_label)
        self.assertTrue(cone.feedback_show_box)
        self.assertTrue(cone.feedback_show_label)
        self.assertTrue(objects[1].show_box)
        self.assertTrue(objects[2].show_box)
        validate_feedback_label(
            ViolationFeedbackCreate(
                type='false_negative',
                corrected_label='safety_cone',
                corrected_bbox=[0, 0, 10, 10],
            ),
            record,
        )

    def test_live_renderer_hides_canonical_metadata_cones(self):
        for metadata in (
            [
                {
                    'id': 6,
                    'code': 'safety_cone',
                    'display_name': '交通錐',
                    'color': '#FF5722',
                },
            ],
        ):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            with patch.object(renderer, '_draw_label') as draw_label:
                renderer.render_overlay_array(
                    frame,
                    [[10, 10, 40, 70, 0.9, 6, 1]],
                    {},
                    [],
                    [],
                    class_metadata=metadata,
                )
            self.assertFalse(frame.any())
            draw_label.assert_not_called()

    def test_regions_still_render_without_individual_cone_boxes(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        renderer.render_overlay_array(
            frame,
            [[10, 10, 40, 70, 0.9, 6, 1]],
            {},
            [[[10, 10], [80, 10], [80, 80], [10, 10]]],
            [],
        )
        self.assertTrue(frame.any())
