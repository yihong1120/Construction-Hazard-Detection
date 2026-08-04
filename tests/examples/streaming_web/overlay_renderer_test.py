from __future__ import annotations

import json
import unittest
from unittest import mock

import cv2
import numpy as np

from examples.streaming_web import overlay_renderer as renderer
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
from examples.streaming_web.overlay_renderer import (
    render_overlay_frame,
)


class OverlayRendererTest(unittest.TestCase):
    """Tests for OverlayRendererTest."""

    def _jpeg_frame(self) -> bytes:
        """Support _jpeg_frame."""
        frame = np.full((240, 320, 3), 245, dtype=np.uint8)
        ok, encoded = cv2.imencode('.jpg', frame)
        self.assertTrue(ok)
        return encoded.tobytes()

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

    def test_env_helpers_parse_configured_values(self) -> None:
        """Exercise this test."""
        with mock.patch.dict(
            'os.environ',
            {
                'STREAMING_OVERLAY_TEST_BOOL': 'yes',
                'STREAMING_OVERLAY_TEST_INT': 'bad',
            },
        ):
            self.assertTrue(
                renderer._env_bool(
                    'STREAMING_OVERLAY_TEST_BOOL',
                    False,
                ),
            )
            self.assertEqual(
                renderer._env_int('STREAMING_OVERLAY_TEST_INT', 7),
                7,
            )
        self.assertTrue(renderer._env_bool('MISSING_OVERLAY_BOOL', True))
        self.assertEqual(renderer._env_int('MISSING_OVERLAY_INT', 7), 7)

    def test_has_warning(self) -> None:
        """Exercise this test."""
        self.assertTrue(
            has_warning(
                json.dumps({
                    'warning_no_hardhat': {'count': 2},
                }),
            ),
        )
        self.assertTrue(
            has_warning({
                'warning_no_hardhat': {'count': True},
            }),
        )
        self.assertFalse(
            has_warning(
                json.dumps({
                    'warning_no_hardhat': {'count': 0},
                }),
            ),
        )
        self.assertFalse(
            has_warning({
                'warning_no_hardhat': {'count': False},
            }),
        )
        self.assertFalse(has_warning(''))

    def test_render_overlay_frame_bad_input_returns_original(self) -> None:
        """Exercise this test."""
        frame_bytes = b'not-a-jpeg'

        rendered = render_overlay_frame(frame_bytes, overlay_mode='backend')

        self.assertEqual(rendered, frame_bytes)

    def test_render_overlay_frame_encode_failure_returns_original(
        self,
    ) -> None:
        """Exercise this test."""
        frame_bytes = self._jpeg_frame()
        with mock.patch(
            'examples.streaming_web.overlay_renderer.cv2.imencode',
            return_value=(False, np.array([], dtype=np.uint8)),
        ):
            rendered = render_overlay_frame(
                frame_bytes,
                overlay_mode='backend',
            )

        self.assertEqual(rendered, frame_bytes)

    def test_render_overlay_array_none_mode_and_empty_frame_return_original(
        self,
    ) -> None:
        """Exercise this test."""
        frame = np.zeros((0, 0, 3), dtype=np.uint8)

        self.assertIs(
            render_overlay_array(frame, overlay_mode='none'),
            frame,
        )
        self.assertIs(
            render_overlay_array(frame, overlay_mode='backend'),
            frame,
        )

    def test_parse_json_returns_none_for_bad_json(self) -> None:
        """Exercise this test."""
        self.assertIsNone(renderer._parse_json('{bad json'))

    def test_render_overlay_frame_none_returns_original(self) -> None:
        """Exercise this test."""
        frame_bytes = self._jpeg_frame()
        rendered = render_overlay_frame(
            frame_bytes,
            detection_items_json=json.dumps([[40, 40, 180, 210, 0.93, 5]]),
            overlay_mode='none',
        )
        self.assertEqual(rendered, frame_bytes)

    def test_render_overlay_frame_with_unicode_label(self) -> None:
        """Exercise this test."""
        frame_bytes = self._jpeg_frame()
        rendered = render_overlay_frame(
            frame_bytes,
            detection_items_json=json.dumps([[40, 40, 180, 210, 0.93, 5, 12]]),
            warnings_json=json.dumps({
                'warning_people_in_controlled_area': {'count': 1},
            }),
            overlay_mode='backend',
            label_language='zh-TW',
        )
        decoded = cv2.imdecode(
            np.frombuffer(rendered, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(decoded)
        assert decoded is not None
        self.assertEqual(decoded.shape[:2], (240, 320))
        self.assertNotEqual(rendered, frame_bytes)

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

    def test_render_overlay_array_accepts_json_strings(self) -> None:
        """Exercise this test."""
        frame = np.full((240, 320, 3), 245, dtype=np.uint8)
        rendered = render_overlay_array(
            frame.copy(),
            detection_items=json.dumps([[40, 40, 180, 210, 0.93, 5, 12]]),
            warnings=json.dumps({
                'warning_people_in_controlled_area': {'count': 1},
            }),
            cone_polygons='[]',
            pole_polygons='[]',
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
            detection_items=[[40, 40, 80, 180, 0.93, 4]],
            warnings={},
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
                [10, 20, 50, 90, 0.93, 5],
                [80, 20, 120, 90, 0.91, 5],
                [150, 20, 190, 90, 0.89, 5],
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
            cone_polygons=[polygon],
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

    def test_cached_detection_parser_handles_json_and_invalid_data(
        self,
    ) -> None:
        """Exercise this test."""
        detections = renderer._detections_for_overlay(
            json.dumps([{
                'class_name': 'person',
                'confidence': 0.9,
                'bbox': {'x': 0.1, 'y': 0.1, 'w': 0.2, 'h': 0.3},
                'track_id': 'abc',
            }]),
            200,
            100,
            0.4,
            json.dumps({
                'warning_people_in_controlled_area': {'count': 1},
            }),
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].bbox, (20, 10, 60, 40))
        self.assertTrue(detections[0].is_warning)
        self.assertEqual(
            renderer._detections_for_overlay(
                '{"bad": true}', 200, 100, 0.4, '',
            ),
            (),
        )
        self.assertEqual(
            renderer._detections_from_data({'bad': True}, 200, 100, 0.4, {}),
            (),
        )

    def test_detections_from_data_adds_explicit_warning_target_overlay(
        self,
    ) -> None:
        """Exercise this test."""
        detections = renderer._detections_from_data(
            [],
            200,
            100,
            0.4,
            {
                'warning_close_to_vehicle': {
                    'count': 1,
                    'person_bboxes': [[20, 10, 40, 30]],
                },
            },
        )

        self.assertEqual(len(detections), 1)
        self.assertTrue(detections[0].is_warning)

    def test_detection_parser_skips_low_confidence_and_invalid_boxes(
        self,
    ) -> None:
        """Exercise this test."""
        parsed = tuple(
            renderer._iter_detections_from_data(
                [
                    [0.1, 0.1, 0.2, 0.2, 0.2, 5],
                    [10, 10, 10, 20, 0.9, 5],
                    'bad',
                    {'bbox': [], 'confidence': 0.9},
                    [0.2, 0.2, 0.3, 0.4, 0.95, 5, -1],
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
        self.assertIsNone(
            renderer._parse_dict_detection(
                {'bbox': [1], 'confidence': 0.9},
                frame_width=200,
                frame_height=100,
                warning_classes=set(),
                warning_targets=set(),
            ),
        )

    def test_warning_targets_and_counts_cover_edge_formats(self) -> None:
        """Exercise this test."""
        warnings = {
            'warning_no_hardhat': {'count': 1},
            'warning_close_to_vehicle': {
                'count': '2',
                'person_bboxes': [[0.1, 0.1, 0.2, 0.3], 'bad'],
            },
            'warning_no_safety_vest': {'count': object()},
            'warning_people_in_utility_pole_controlled_area': 1,
            'detect_machinery_close_to_pole': True,
        }

        self.assertEqual(
            renderer._warning_targets(warnings, 200, 100),
            {(20, 10, 40, 30)},
        )
        self.assertEqual(
            renderer._explicit_person_warning_bboxes('bad', 200, 100),
            set(),
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
        self.assertEqual(renderer._warning_count({'count': b'3'}), 3)
        self.assertEqual(renderer._warning_count({'count': object()}), 1)
        self.assertEqual(renderer._warning_count({'count': 'bad'}), 1)
        self.assertEqual(renderer._warning_count(False), 0)
        self.assertEqual(renderer._warning_count(2.8), 2)
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

    def test_polygon_helpers_cover_invalid_and_normalised_data(self) -> None:
        """Exercise this test."""
        polygons = renderer._normalised_polygons_for_overlay(
            json.dumps([[
                {'x': 0.1, 'y': 0.1},
                {'x': 0.3, 'y': 0.1},
                {'x': 0.3, 'y': 0.3},
            ]]),
            100,
            50,
        )

        self.assertEqual(len(polygons), 1)
        np.testing.assert_array_equal(
            polygons[0],
            np.array([[10, 5], [30, 5], [30, 15]], dtype=np.int32),
        )
        self.assertEqual(
            renderer._normalised_polygons_for_overlay(
                '{"bad": true}', 100, 50,
            ),
            (),
        )
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        renderer._draw_polygons(
            frame,
            (
                ('{"bad": true}', (255, 0, 0), (255, 0, 0), 0.4),
                (
                    json.dumps([[[1, 1], [10, 1], [10, 10]]]),
                    (255, 0, 0),
                    (255, 0, 0),
                    0.4,
                ),
            ),
        )
        self.assertTrue(frame.any())
        self.assertIsNone(
            renderer._normalise_polygon(
                [[1, 2], [3, 4]], 10, 10,
            ),
        )
        self.assertIsNone(
            renderer._normalise_polygon([[1, 2], [3, 4], ['bad', 5]], 10, 10),
        )
        self.assertIsNone(
            renderer._normalise_polygon(
                [
                    {'x': 1, 'y': 2},
                    {'x': 3},
                    {'x': 4, 'y': 5},
                ],
                10,
                10,
            ),
        )
        self.assertIsNone(
            renderer._normalise_polygon([[1, 2], object(), [4, 5]], 10, 10),
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
        renderer._blend_bgra_roi(
            frame,
            np.zeros((2, 2, 4), dtype=np.uint8),
            20,
            20,
        )
        renderer._blend_bgra_roi(
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
            'examples.streaming_web.overlay_renderer.'
            '_load_overlay_font',
            return_value=None,
        ):
            renderer._render_pillow_text_bitmap.cache_clear()
            self.assertIsNone(
                renderer._render_pillow_text_bitmap(
                    '中文',
                    16,
                    (255, 255, 255),
                ),
            )
            renderer._draw_pillow_text(
                frame,
                '中文',
                (5, 5),
                (255, 255, 255),
                (0, 0, 10, 10),
            )

        renderer._load_overlay_font.cache_clear()
        with (
            mock.patch(
                'examples.streaming_web.overlay_renderer.'
                'ImageFont.truetype',
                side_effect=RuntimeError,
            ),
            mock.patch(
                'examples.streaming_web.overlay_renderer.'
                'ImageFont.load_default',
                side_effect=RuntimeError,
            ),
        ):
            self.assertIsNone(renderer._load_overlay_font(13))

        renderer._render_pillow_text_bitmap.cache_clear()
        rendered = renderer._render_pillow_text_bitmap(
            '中文',
            16,
            (255, 255, 255),
        )
        self.assertIsNotNone(rendered)
        renderer._draw_pillow_text(
            frame,
            '中文',
            (30, 30),
            (255, 255, 255),
            (0, 0, 19, 19),
        )
