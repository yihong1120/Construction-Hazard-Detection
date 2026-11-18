from __future__ import annotations

import importlib.util
import itertools
import math
import os
import random
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

import src.stream_processor as processor
from src.danger_detector import DangerDetector
from src.native_postprocess import load_postprocess_backend
from src.native_postprocess import RustDangerDetector
from src.native_postprocess import RustYoloDetector
from src.settings.postprocess import PostprocessSettings
from src.yolo_detector import YoloDetector

AVAILABLE = importlib.util.find_spec('construction_postprocess') is not None
FLAGS = (
    'detect_no_safety_vest_or_helmet',
    'detect_near_machinery_or_vehicle',
    'detect_in_restricted_area',
    'detect_in_utility_pole_restricted_area',
    'detect_machinery_close_to_pole',
)


class BackendSelectionTest(unittest.TestCase):
    def test_backend_settings_reject_invalid_names(self):
        with patch.dict(os.environ, {'POSTPROCESS_BACKEND': 'fastest'}):
            with self.assertRaises(ValidationError):
                PostprocessSettings()
        self.assertEqual(PostprocessSettings(backend='rust').backend, 'rust')

    def test_python_mode_never_loads_an_extension(self):
        with patch('src.native_postprocess.importlib.import_module') as load:
            self.assertIsNone(load_postprocess_backend('python'))
        load.assert_not_called()

    def test_absence_is_optional_only_for_auto(self):
        error = ModuleNotFoundError(
            'not installed', name='construction_postprocess',
        )
        with patch(
            'src.native_postprocess.importlib.import_module',
            side_effect=error,
        ):
            self.assertIsNone(load_postprocess_backend('auto'))
            with self.assertRaisesRegex(RuntimeError, 'install the'):
                load_postprocess_backend('rust')

    def test_broken_installation_does_not_silently_switch_backends(self):
        for error in (
            ImportError('unresolved native symbol'),
            ModuleNotFoundError('missing dependency', name='another_module'),
        ):
            with self.subTest(error=error), patch(
                'src.native_postprocess.importlib.import_module',
                side_effect=error,
            ):
                with self.assertRaises((ImportError, RuntimeError)):
                    load_postprocess_backend('auto')
        with patch(
            'src.native_postprocess.importlib.import_module',
            return_value=SimpleNamespace(ABI_VERSION=2),
        ):
            with self.assertRaisesRegex(RuntimeError, 'ABI'):
                load_postprocess_backend('auto')
        with patch(
            'src.native_postprocess.importlib.import_module',
            return_value=SimpleNamespace(ABI_VERSION=1),
        ):
            with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
                load_postprocess_backend('auto')


@unittest.skipUnless(AVAILABLE, 'Requires the native/postprocess wheel')
class NativePostprocessTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.extension = importlib.import_module('construction_postprocess')

    def test_tracker_algorithm_is_fixed_for_the_native_stream(self):
        with self.assertRaisesRegex(ValueError, 'centroid'):
            RustYoloDetector(
                native_extension=self.extension, remote_tracker='hungarian',
            )
        native = RustYoloDetector(native_extension=self.extension)
        native.remote_tracker = 'hungarian'
        with self.assertRaisesRegex(RuntimeError, 'mid-stream'):
            native.track_detections([])

    def test_inactive_rules_do_not_read_detection_rows(self):
        self.assertEqual(
            self.extension.safety_warnings([[math.nan]], False, False), {},
        )

    def test_centroid_outputs_match_through_ties_expiry_and_parameter_changes(
        self,
    ):
        rng = random.Random(20261116)
        for movement, keep in itertools.product((0.0, 2.0, 40.0), (0, 1, 10)):
            reference = YoloDetector(movement_thr=movement, max_id_keep=keep)
            native = RustYoloDetector(
                native_extension=self.extension,
                movement_thr=movement, max_id_keep=keep,
            )
            for frame in range(65):
                rows = []
                if frame % 7:
                    for index in range(25):
                        x = index * 8 + rng.choice((-2, 0, 2, 40, 80))
                        y = index * 9 + rng.choice((-2, 0, 2))
                        rows.append([x, y, x + 5, y + 10, 0.9, index % 11])
                    rows.append(list(rows[0]))
                if frame == 35:
                    reference.movement_thr_sq = native.movement_thr_sq = 100
                    reference.max_id_keep = native.max_id_keep = 3
                self.assertEqual(
                    reference.track_detections(rows),
                    native.track_detections(rows),
                    (movement, keep, frame),
                )

    def test_full_danger_output_preserves_all_feature_flags_and_polygons(self):
        scenes = [
            [],
            [
                [0, 0, 4, 4, 0.9, 6], [100, 0, 104, 4, 0.9, 6],
                [0, 100, 4, 104, 0.9, 6], [100, 100, 104, 104, 0.9, 6],
                [40, 40, 45, 50, 0.9, 5, 42, 0],
                [45, 45, 200, 200, 0.8, 10, 7, 1],
                [150, 30, 160, 200, 0.9, 9],
                [10, 10, 20, 20, 0.9, 2], [10, 10, 20, 20, 0.9, 4],
            ],
            [
                [105, 110, 100, 100, 0.9, 5, 42.0, 0],
                [107, 105, 200, 200, 0.8, 8, 1, 1],
                [108, 105, 200, 200, 0.8, 10, -1, 0],
            ],
            [
                [100, 100, 105, 110, 0.9, 5],
                [107, 105, 200, 200, 0.8, 10],
                [108, 105, 220, 210, 0.8, 10],
            ],
        ]
        for bits in itertools.product((False, True), repeat=len(FLAGS)):
            flags = dict(zip(FLAGS, bits))
            reference = DangerDetector(flags)
            native = RustDangerDetector(
                flags, native_extension=self.extension,
            )
            for scene in scenes:
                self.assertEqual(
                    reference.detect_danger(
                        scene,
                    ), native.detect_danger(scene),
                    (flags, scene),
                )
        self.assertEqual(
            DangerDetector().detect_danger(scenes[1]),
            RustDangerDetector(
                native_extension=self.extension,
            ).detect_danger(scenes[1]),
        )

    def test_person_identifiers_preserve_python_string_semantics(self):
        for identifier in (42, 42.0, 'camera:42', None, '', -1):
            scene = [
                [100, 100, 105, 110, 0.9, 5, identifier, 0],
                [107, 105, 200, 200, 0.8, 10, 1, 1],
            ]
            self.assertEqual(
                DangerDetector().detect_danger(scene),
                RustDangerDetector(
                    native_extension=self.extension,
                ).detect_danger(scene),
            )

    def test_native_rejection_preserves_existing_tracker_state(self):
        reference = YoloDetector()
        native = RustYoloDetector(native_extension=self.extension)
        rows = [[0, 0, 10, 20, 0.9, 5]]
        self.assertEqual(
            reference.track_detections(rows), native.track_detections(rows),
        )
        with self.assertRaises(ValueError):
            native.track_detections([[math.nan, 0, 10, 20, 0.9, 5]])
        reference.frame_count += 1
        self.assertEqual(
            reference.track_detections(rows), native.track_detections(rows),
        )
        with self.assertRaises(ValueError):
            self.extension.CentroidTracker().track(rows, 1, math.inf, 10)
        with self.assertRaises(ValueError):
            self.extension.safety_warnings(
                [[math.inf, 0, 1, 1, 0.9, 5]], True, True,
            )

    def test_large_spatial_extent_has_bounded_native_work(self):
        result = self.extension.safety_warnings(
            [
                [-1e9, -1e9, 1e9, 1e9, 0.9, 10],
                [1e9 + 5, 1e9 + 5, 1e9 + 6, 1e9 + 15, 0.9, 5, 42],
            ],
            False, True,
        )
        self.assertEqual(result['warning_close_to_vehicle']['count'], 1)
        self.assertEqual(
            result['warning_close_to_vehicle']['person_track_ids'], ['42'],
        )

    async def test_stream_entrypoint_selects_native_and_closes_resources(self):
        cfg = {
            'video_url': 'rtsp://example.test/camera',
            'model_key': 'yolo26n', 'site': 'test', 'stream_name': 'camera',
            'recognition_enabled': True, 'detection_items': {},
        }
        capture = AsyncMock()
        worker = AsyncMock()
        redis = SimpleNamespace(
            redis=SimpleNamespace(get=AsyncMock(return_value=None)),
            delete=AsyncMock(),
        )
        with patch.dict(os.environ, {'POSTPROCESS_BACKEND': 'rust'}), patch(
            'src.stream_processor.StreamCapture', return_value=capture,
        ), patch(
            'src.stream_processor.YoloWorkerClient', return_value=worker,
        ), patch(
            'src.stream_processor.RedisManager', return_value=redis,
        ), patch('src.stream_processor.ViolationQueue') as delivery, patch(
            'src.stream_processor._run_decoupled_media_server_loop',
            new_callable=AsyncMock,
        ) as loop:
            await processor._run_single_stream(cfg, object(), object())
        runtime = loop.await_args.args[0]
        self.assertIsInstance(runtime.detector, RustYoloDetector)
        self.assertIsInstance(runtime.danger, RustDangerDetector)
        delivery.return_value.close.assert_called_once()
        worker.close.assert_awaited_once()
        capture.release_resources.assert_awaited_once()
        redis.delete.assert_awaited_once()

    async def test_worker_errors_keep_frame_age_metadata_and_cleanup(self):
        detectors = [
            YoloDetector(worker_client=AsyncMock()),
            RustYoloDetector(
                native_extension=self.extension, worker_client=AsyncMock(),
            ),
        ]
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        rows = [[0, 0, 10, 20, 0.9, 5]]
        for detector in detectors:
            detector.worker_client.last_model_metadata = {'version': 'v1'}
            detector.worker_client.detect.side_effect = [
                rows, RuntimeError('inference unavailable'), rows,
            ]
        first = [
            await detector.generate_detections(frame)
            for detector in detectors
        ]
        self.assertEqual(first[0], first[1])
        for detector in detectors:
            with self.assertRaisesRegex(RuntimeError, 'inference unavailable'):
                await detector.generate_detections(frame)
        last = [
            await detector.generate_detections(frame)
            for detector in detectors
        ]
        self.assertEqual(last[0], last[1])
        for detector in detectors:
            self.assertEqual(detector.frame_count, 3)
            self.assertEqual(detector.model_metadata, {'version': 'v1'})
            await detector.close()
            detector.worker_client.close.assert_awaited_once()
        independent = RustYoloDetector(native_extension=self.extension)
        self.assertEqual(independent.track_detections(rows)[0][6], 0)
