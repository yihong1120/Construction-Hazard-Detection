from __future__ import annotations

import json
import unittest
from typing import cast
from unittest import mock

import cv2
import numpy as np

from examples.streaming_web import overlay_renderer as renderer
from examples.streaming_web import overlay_text
from examples.streaming_web.overlay_renderer import (
    has_warning,
)
from examples.streaming_web.overlay_renderer import (
    normalise_label_language,
)
from examples.streaming_web.overlay_renderer import (
    normalise_overlay_mode,
)
from examples.streaming_web.overlay_renderer import (
    render_overlay_array,
)


class OverlayRendererTest(unittest.TestCase):
    """Tests for OverlayRendererTest."""

    def test_normalise_overlay_mode(self) -> None:
        """Exercise this test."""
        self.assertEqual(normalise_overlay_mode('backend'), 'backend')
        self.assertEqual(normalise_overlay_mode('true'), 'backend')
        self.assertEqual(normalise_overlay_mode('none'), 'none')
        self.assertEqual(normalise_overlay_mode(None), 'none')

    def test_normalise_label_language(self) -> None:
        """Exercise this test."""
        self.assertEqual(normalise_label_language('zh_TW'), 'zh-TW')
        self.assertEqual(normalise_label_language('en-GB'), 'en')
        self.assertEqual(normalise_label_language('ja-JP'), 'ja')
        self.assertEqual(normalise_label_language('th-TH'), 'th')
        self.assertEqual(normalise_label_language('vi-XX'), 'vi')
        self.assertEqual(normalise_label_language('zh'), 'zh-TW')
        self.assertEqual(normalise_label_language('missing'), 'en')

    def test_has_warning(self) -> None:
        """Exercise this test."""
        self.assertTrue(
            has_warning(
                {'warning_no_hardhat': {'count': 2}},
            ),
        )
        self.assertTrue(
            has_warning({
                'warning_no_hardhat': {'count': True},
            }),
        )
        self.assertFalse(
            has_warning({'warning_no_hardhat': {'count': 0}}),
        )
        self.assertFalse(
            has_warning({
                'warning_no_hardhat': {'count': False},
            }),
        )

    def test_render_overlay_array_none_mode_and_empty_frame_return_original(
        self,
    ) -> None:
        """Exercise this test."""
        frame = np.zeros((0, 0, 3), dtype=np.uint8)

        self.assertIs(
            render_overlay_array(
                frame,
                detection_items=[],
                warnings={},
                cone_polygons=[],
                pole_polygons=[],
                overlay_mode='none',
            ),
            frame,
        )
        self.assertIs(
            render_overlay_array(
                frame,
                detection_items=[],
                warnings={},
                cone_polygons=[],
                pole_polygons=[],
                overlay_mode='backend',
            ),
            frame,
        )

    def test_render_overlay_array_draws_without_json_roundtrip(self) -> None:
        """Exercise this test."""
        frame = np.full((240, 320, 3), 245, dtype=np.uint8)
        rendered = render_overlay_array(
            frame.copy(),
            detection_items=[[40, 40, 180, 210, 0.93, 5, 12]],
            warnings={
                'warning_people_in_controlled_area': {'count': 1},
            },
            cone_polygons=[],
            pole_polygons=[],
            overlay_mode='backend',
            label_language='zh-TW',
        )
        self.assertEqual(rendered.shape[:2], (240, 320))
        self.assertFalse(np.array_equal(rendered, frame))

    def test_render_overlay_array_accepts_strict_decoded_data(self) -> None:
        """The array renderer consumes the detector's decoded payload."""
        frame = np.full((240, 320, 3), 245, dtype=np.uint8)
        rendered = render_overlay_array(
            frame.copy(),
            detection_items=[[40, 40, 180, 210, 0.93, 5, 12, 0]],
            warnings={
                'warning_people_in_controlled_area': {'count': 1},
            },
            cone_polygons=[],
            pole_polygons=[],
            overlay_mode='backend',
            label_language='zh-TW',
        )

        self.assertEqual(rendered.shape[:2], (240, 320))
        self.assertFalse(np.array_equal(rendered, frame))

    def test_render_overlay_array_draws_warning_panel_bottom_right(
        self,
    ) -> None:
        """Exercise this test."""
        frame = np.full((240, 320, 3), 245, dtype=np.uint8)
        rendered = render_overlay_array(
            frame.copy(),
            detection_items=[[40, 40, 180, 210, 0.93, 5, 12]],
            warnings={'warning_close_to_vehicle': {'count': 2}},
            cone_polygons=[],
            pole_polygons=[],
            overlay_mode='backend',
            label_language='en',
        )

        bottom_right = rendered[170:238, 180:318]
        original_bottom_right = frame[170:238, 180:318]
        self.assertFalse(np.array_equal(bottom_right, original_bottom_right))

    def test_render_overlay_array_does_not_infer_proximity_person(
        self,
    ) -> None:
        """Exercise this test."""
        frame = np.full((240, 320, 3), 245, dtype=np.uint8)
        rendered = render_overlay_array(
            frame.copy(),
            detection_items=[
                [100, 100, 105, 110, 0.95, 5, -1, 0],
                [107, 105, 200, 200, 0.85, 10, 1, 1],
            ],
            warnings={'warning_close_to_vehicle': {'count': 1}},
            cone_polygons=[],
            pole_polygons=[],
            overlay_mode='backend',
            label_language='en',
            box_thickness=3,
        )

        person_border_bgr = rendered[100, 100]
        self.assertGreater(person_border_bgr[1], 120)
        vehicle_border_bgr = rendered[105, 150]
        self.assertGreater(vehicle_border_bgr[1], 200)

    def test_render_overlay_array_uses_explicit_warning_person_bbox(
        self,
    ) -> None:
        """Exercise this test."""
        frame = np.full((240, 320, 3), 245, dtype=np.uint8)
        rendered = render_overlay_array(
            frame.copy(),
            detection_items=[
                [40, 40, 80, 120, 0.95, 5, 42, 0],
                [180, 40, 220, 140, 0.85, 10, 1, 1],
            ],
            warnings={
                'warning_close_to_vehicle': {
                    'count': 1,
                    'person_bboxes': [[40, 40, 80, 120]],
                    'person_track_ids': ['42'],
                },
            },
            cone_polygons=[],
            pole_polygons=[],
            overlay_mode='backend',
            label_language='en',
            box_thickness=3,
        )

        person_border_bgr = rendered[40, 40]
        self.assertGreater(person_border_bgr[2], 200)
        self.assertLess(person_border_bgr[0], 100)
        vehicle_border_bgr = rendered[40, 180]
        self.assertGreater(vehicle_border_bgr[1], 200)

    def test_render_overlay_array_draws_warning_person_bbox_directly(
        self,
    ) -> None:
        """Exercise this test."""
        frame = np.full((240, 320, 3), 245, dtype=np.uint8)
        rendered = render_overlay_array(
            frame.copy(),
            detection_items=[
                [180, 40, 220, 140, 0.85, 10, 1, 1],
            ],
            warnings={
                'warning_close_to_vehicle': {
                    'count': 1,
                    'person_bboxes': [[40, 40, 80, 120]],
                },
            },
            cone_polygons=[],
            pole_polygons=[],
            overlay_mode='backend',
            label_language='en',
            box_thickness=3,
        )

        target_border_bgr = rendered[40, 40]
        self.assertGreater(target_border_bgr[2], 200)
        self.assertLess(target_border_bgr[0], 100)

    def test_render_overlay_array_infers_warning_panel_from_violation_box(
        self,
    ) -> None:
        """Exercise this test."""
        frame = np.full((240, 320, 3), 245, dtype=np.uint8)
        rendered = render_overlay_array(
            frame.copy(),
            detection_items=[[40, 40, 80, 180, 0.93, 4, -1, 0]],
            warnings={},
            cone_polygons=[],
            pole_polygons=[],
            overlay_mode='backend',
            label_language='en',
        )

        bottom_right = rendered[170:238, 180:318]
        original_bottom_right = frame[170:238, 180:318]
        self.assertFalse(np.array_equal(bottom_right, original_bottom_right))

    def test_render_overlay_array_draws_all_detection_items(self) -> None:
        """Exercise this test."""
        frame = np.full((240, 320, 3), 245, dtype=np.uint8)
        rendered = render_overlay_array(
            frame.copy(),
            detection_items=[
                [10, 20, 50, 90, 0.93, 5, -1, 0],
                [80, 20, 120, 90, 0.91, 5, -1, 0],
                [150, 20, 190, 90, 0.89, 5, -1, 0],
            ],
            warnings={},
            cone_polygons=[],
            pole_polygons=[],
            overlay_mode='backend',
            label_language='en',
            box_thickness=3,
        )

        self.assertFalse(np.array_equal(rendered[20, 10], frame[20, 10]))
        self.assertFalse(np.array_equal(rendered[20, 80], frame[20, 80]))
        self.assertFalse(np.array_equal(rendered[20, 150], frame[20, 150]))

    def test_render_overlay_array_polygon_fill_keeps_unrelated_roi(
        self,
    ) -> None:
        """ROI polygon fill is pixel-equivalent to a full-frame blend."""
        frame = np.arange(120 * 160 * 3, dtype=np.uint8).reshape(120, 160, 3)
        polygon = [[10, 10], [40, 10], [40, 40], [10, 40]]
        rendered = render_overlay_array(
            frame.copy(),
            detection_items=[],
            warnings={},
            cone_polygons=[cast(renderer.PolygonCoordinates, polygon)],
            pole_polygons=[],
            overlay_mode='backend',
            label_language='en',
        )

        points = np.array(polygon, dtype=np.int32)
        reference = frame.copy()
        overlay = reference.copy()
        cv2.fillPoly(overlay, [points], (129, 64, 255))
        cv2.addWeighted(overlay, 0.4, reference, 0.6, 0, reference)
        cv2.polylines(
            reference,
            [points],
            isClosed=True,
            color=(99, 30, 233),
            thickness=3,
        )

        np.testing.assert_array_equal(rendered, reference)

    def test_detection_parser_skips_low_confidence_and_degenerate_boxes(
        self,
    ) -> None:
        """Exercise this test."""
        parsed = tuple(
            renderer._iter_detections_from_data(
                [
                    [0.1, 0.1, 0.2, 0.2, 0.2, 5, -1, 0],
                    [10, 10, 10, 20, 0.9, 5, -1, 0],
                    [0.2, 0.2, 0.3, 0.4, 0.95, 5, -1, 0],
                ],
                frame_width=200,
                frame_height=100,
                warning_classes=set(),
                warning_targets=set(),
                min_confidence=0.4,
            ),
        )

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].bbox, (40, 20, 60, 40))
        self.assertIsNone(parsed[0].track_id)

    def test_warning_targets_and_counts_use_the_detector_schema(self) -> None:
        """Warnings are dictionaries with an integer count field."""
        warnings: renderer.WarningPayload = {
            'warning_no_hardhat': {'count': 1},
            'warning_close_to_vehicle': {
                'count': 2,
                'person_bboxes': [[0.1, 0.1, 0.2, 0.3]],
            },
            'warning_no_safety_vest': {'count': 1},
            'warning_people_in_utility_pole_controlled_area': {'count': 1},
            'detect_machinery_close_to_pole': {'count': 1},
        }

        self.assertEqual(
            renderer._warning_targets(warnings, 200, 100),
            {(20, 10, 40, 30)},
        )
        self.assertEqual(
            renderer._warning_classes(warnings),
            {
                'no-hardhat',
                'no-safety-vest',
                'person',
                'machinery',
                'vehicle',
            },
        )
        self.assertEqual(renderer._warning_count({'count': 3}), 3)
        self.assertEqual(
            renderer._bbox_iou((0, 0, 1, 1), (2, 2, 3, 3)),
            0.0,
        )

    def test_warning_summary_lines_limit_and_inferred_counts(self) -> None:
        """Exercise this test."""
        detections = (
            renderer.DetectionOverlay('no-hardhat', 0.9, (1, 1, 5, 5)),
            renderer.DetectionOverlay('no-hardhat', 0.8, (6, 1, 9, 5)),
        )

        lines = renderer._warning_summary_lines(
            {},
            'en',
            detections=detections,
        )

        self.assertEqual(lines, ['No hardhat x2'])
        self.assertEqual(
            renderer._format_warning_summary_line('unknown_key', 2, {}),
            'unknown_key x2',
        )
        self.assertEqual(
            renderer._warning_summary_lines(
                {'warning_no_hardhat': {'count': 1}},
                'en',
                detections=detections,
            ),
            ['No hardhat'],
        )
        with mock.patch.object(
            renderer,
            '_overlay_max_warning_summary_items',
            1,
        ):
            self.assertEqual(
                renderer._warning_summary_lines(
                    {
                        'warning_no_hardhat': {'count': 0},
                        'warning_no_safety_vest': {'count': 2},
                        'warning_close_to_vehicle': {'count': 1},
                    },
                    'en',
                ),
                ['No safety vest x2'],
            )
            self.assertEqual(
                renderer._warning_summary_lines(
                    {},
                    'en',
                    detections=(
                        renderer.DetectionOverlay(
                            'no-hardhat',
                            0.9,
                            (1, 1, 2, 2),
                        ),
                        renderer.DetectionOverlay(
                            'no-safety-vest',
                            0.9,
                            (3, 3, 4, 4),
                        ),
                    ),
                ),
                ['No hardhat'],
            )

    def test_draw_warning_summary_status_and_empty_roi(self) -> None:
        """Exercise this test."""
        frame = np.zeros((1, 1, 3), dtype=np.uint8)
        with mock.patch.object(
            renderer,
            '_overlay_draw_warning_status',
            True,
        ):
            renderer._draw_warning_summary(frame, {}, 'en')

        self.assertEqual(frame.shape, (1, 1, 3))
        renderer._draw_warning_summary(
            np.zeros((0, 0, 3), dtype=np.uint8), {
                'warning_no_hardhat': {'count': 1},
            }, 'en',
        )

    def test_label_and_colour_helpers_cover_branching(self) -> None:
        """Exercise this test."""
        frame = np.full((80, 80, 3), 245, dtype=np.uint8)
        detection = renderer.DetectionOverlay(
            class_name='person',
            confidence=0.91,
            bbox=(70, 5, 79, 60),
            track_id='7',
        )

        renderer._draw_detection(frame, detection, 'en', 2)
        renderer._draw_detection(
            frame,
            renderer.DetectionOverlay('cone', 0.9, (1, 1, 10, 10)),
            'en',
            2,
        )

        self.assertFalse(renderer._should_draw_label(detection, 99_999))
        with mock.patch.object(
            renderer,
            '_overlay_draw_labels',
            False,
        ):
            self.assertFalse(renderer._should_draw_label(detection, 0))
        with mock.patch.object(
            renderer,
            '_overlay_label_warnings_only',
            True,
        ):
            self.assertFalse(renderer._should_draw_label(detection, 0))
        self.assertEqual(renderer._color_for_class('person'), (255, 152, 0))
        self.assertEqual(len(renderer._color_for_class('unknown-class')), 3)
        self.assertTrue(renderer._is_bright((255, 255, 255)))

    def test_polygon_helpers_normalise_detector_coordinates(self) -> None:
        """Polygon payloads use the detector's list-of-coordinate format."""
        polygons = renderer._normalised_polygons_for_overlay(
            json.dumps([[
                [0.1, 0.1],
                [0.3, 0.1],
                [0.3, 0.3],
            ]]),
            100,
            50,
        )

        self.assertEqual(len(polygons), 1)
        np.testing.assert_array_equal(
            polygons[0],
            np.array([[10, 5], [30, 5], [30, 15]], dtype=np.int32),
        )
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        renderer._draw_polygons(
            frame,
            (
                (
                    json.dumps([[[1, 1], [10, 1], [10, 10]]]),
                    (255, 0, 0),
                    (255, 0, 0),
                    0.4,
                ),
            ),
        )
        self.assertTrue(frame.any())
        np.testing.assert_array_equal(
            renderer._normalise_polygon(
                [[1, 2], [3, 4], [4, 5]],
                10,
                10,
            ),
            np.array([[1, 2], [3, 4], [4, 5]], dtype=np.int32),
        )

    def test_roi_blend_helpers_cover_noop_edges(self) -> None:
        """Exercise this test."""
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        points = np.array([[1, 1], [4, 1], [4, 4]], dtype=np.int32)

        renderer._blend_polygon_fill_roi(frame, points, (0, 0, 255), 0)
        renderer._blend_polygon_fill_roi(
            frame,
            np.array([[-5, -5], [-4, -5], [-4, -4]], dtype=np.int32),
            (0, 0, 255),
            0.5,
        )
        overlay_text._blend_bgra_roi(
            frame,
            np.zeros((2, 2, 4), dtype=np.uint8),
            20,
            20,
        )
        overlay_text._blend_bgra_roi(
            frame,
            np.zeros((2, 2, 4), dtype=np.uint8),
            1,
            1,
        )
        renderer._blend_polygon_fill_roi(
            frame[:, :0],
            points,
            (0, 0, 255),
            0.5,
        )
        renderer._blend_polygon_fill_roi(
            np.zeros((10, 10, 0), dtype=np.uint8),
            points,
            (0, 0, 255),
            0.5,
        )

        self.assertFalse(frame.any())

    def test_pillow_text_helpers_cover_missing_font_and_clipped_area(
        self,
    ) -> None:
        """Exercise this test."""
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        with mock.patch(
            'examples.streaming_web.overlay_text.'
            '_load_overlay_font',
            return_value=None,
        ):
            overlay_text._render_pillow_text_bitmap.cache_clear()
            self.assertIsNone(
                overlay_text._render_pillow_text_bitmap(
                    '中文',
                    16,
                    (255, 255, 255),
                ),
            )
            overlay_text._draw_pillow_text(
                frame,
                '中文',
                (5, 5),
                (255, 255, 255),
                (0, 0, 10, 10),
            )

        overlay_text._load_overlay_font.cache_clear()
        with (
            mock.patch(
                'examples.streaming_web.overlay_text.'
                'ImageFont.truetype',
                side_effect=RuntimeError,
            ),
            mock.patch(
                'examples.streaming_web.overlay_text.'
                'ImageFont.load_default',
                side_effect=RuntimeError,
            ),
        ):
            self.assertIsNone(overlay_text._load_overlay_font(13))

        overlay_text._render_pillow_text_bitmap.cache_clear()
        rendered = overlay_text._render_pillow_text_bitmap(
            '中文',
            16,
            (255, 255, 255),
        )
        self.assertIsNotNone(rendered)
        overlay_text._draw_pillow_text(
            frame,
            '中文',
            (30, 30),
            (255, 255, 255),
            (0, 0, 19, 19),
        )
