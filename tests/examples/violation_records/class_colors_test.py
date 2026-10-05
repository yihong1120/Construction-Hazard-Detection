from __future__ import annotations

from datetime import datetime
from datetime import timezone
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

from examples.streaming_web import overlay_labels
from examples.streaming_web.overlay_renderer import render_overlay_array
from examples.violation_records import violation_services as services
from examples.violation_records.record_policy import historical_classes
from examples.YOLO_server_api.model_registry import ModelClass


class ClassColorsTest(TestCase):
    def test_invalid_optional_colors_do_not_invalidate_classes(self):
        for color in (None, '', '#ABC', '#FFAB47BC', 'red', 123, '#12345Z'):
            item = ModelClass(
                id=0, code='forklift', display_name='堆高機', color=color,
            )
            self.assertEqual(item.color, '#AB47BC')
            self.assertEqual(item.code, 'forklift')
        self.assertEqual(
            ModelClass(
                id=0, code='forklift', display_name='堆高機', color='#ab47bc',
            ).color,
            '#AB47BC',
        )

    def test_list_and_detail_share_original_snapshot(self):
        snapshot = [
            {
                'id': 0,
                'code': 'forklift',
                'display_name': '堆高機',
                'color': '#AB47BC',
            },
        ]
        now = datetime.now(timezone.utc)
        record = SimpleNamespace(
            id=4,
            model_id='old',
            model_version='v1',
            model_classes=snapshot,
            site='Site',
            stream_name='Cam',
            detection_time=now,
            created_at=now,
            image_path='image.jpg',
            detections_json='[[1,2,3,4,0.9,0,1]]',
            warnings_json=None,
            cone_polygon_json=None,
            pole_polygon_json=None,
            is_flagged=False,
            flag_reason=None,
            flagged_by=None,
            flagged_at=None,
            review_note=None,
            reviewed_by=None,
            reviewed_at=None,
        )
        with patch.object(
            services, '_media_endpoint_url', return_value='media',
        ):
            detail = services._violation_to_detail_item(record, MagicMock())
            listing = services._violation_to_list_item(
                (
                    4,
                    'Site',
                    'Cam',
                    now,
                    'image.jpg',
                    None,
                    False,
                    None,
                    None,
                    'v1',
                    snapshot,
                ),
                MagicMock(),
            )
        self.assertEqual(detail.class_metadata, snapshot)
        self.assertEqual(listing.class_metadata, snapshot)
        self.assertEqual(listing.model_version, 'v1')
        self.assertEqual(historical_classes(record)[0].code, 'forklift')

    def test_renderer_uses_snapshot_numeric_mapping_color_and_name(self):
        from examples.streaming_web import overlay_renderer as renderer

        frame = np.zeros((160, 160, 3), dtype=np.uint8)
        with patch.object(renderer, '_draw_label') as label:
            render_overlay_array(
                frame,
                [[20, 50, 120, 120, 0.9, 0, 1]],
                {},
                [],
                [],
                class_metadata=[
                    {
                        'id': 0,
                        'code': 'forklift',
                        'display_name': '堆高機',
                        'color': '#AB47BC',
                    },
                ],
            )
        detection = label.call_args.args[1]
        self.assertEqual(detection.class_name, 'forklift')
        self.assertEqual(
            overlay_labels._format_label(
                detection, 'zh-TW',
            ), '堆高機',
        )
        self.assertEqual(tuple(frame[90, 20]), (188, 71, 171))

    def test_warning_overlay_preserves_class_color(self):
        frame = np.zeros((160, 160, 3), dtype=np.uint8)
        render_overlay_array(
            frame,
            [[20, 50, 120, 120, 0.9, 5, 1]],
            {
                'warning_close_to_vehicle': {
                    'count': 1,
                    'person_bboxes': [[20, 50, 120, 120]],
                },
            },
            [],
            [],
            class_metadata=[
                {
                    'id': 5,
                    'code': 'person',
                    'display_name': '人員',
                    'color': '#FF9800',
                },
            ],
        )
        self.assertEqual(tuple(frame[90, 20]), (0, 152, 255))

    def test_backend_owns_unknown_and_flagged_box_colors(self):
        from examples.violation_records.schemas import FeedbackDetectionItem

        detections = [
            FeedbackDetectionItem(
                id='det_0',
                label='forklift',
                confidence=0.9,
                bbox=[0, 0, 0.5, 0.5],
            ),
            FeedbackDetectionItem(
                id='det_1',
                label='class-99',
                confidence=0.9,
                bbox=[0.5, 0.5, 1, 1],
            ),
        ]
        objects = services._overlay_objects_from_feedback(
            detections,
            [],
            None,
            class_metadata=[
                {
                    'id': 0,
                    'code': 'forklift',
                    'display_name': '堆高機',
                    'color': '#123456',
                },
            ],
        )
        self.assertEqual(objects[0].color, '#123456')
        self.assertEqual(objects[0].code, 'forklift')
        self.assertEqual(objects[0].display_name, '堆高機')
        self.assertEqual(objects[1].color, '#9E9E9E')

    def test_backend_style_preserves_snapshot_color(self):
        from examples.shared.class_colors import overlay_style

        style = overlay_style([{'code': 'person', 'color': '#123456'}])
        self.assertEqual(style['class_colors']['person'], '#123456')
        self.assertEqual(style['cone_polygon_color'], '#FF4081')
        self.assertEqual(overlay_style()['class_colors']['person'], '#FF9800')
