from __future__ import annotations

import runpy
import sys
import unittest
from types import ModuleType
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

import src.local_yolo_detector as local_module
from src.local_yolo_detector import LocalYoloDetector


class TestLocalYoloDetector(unittest.IsolatedAsyncioTestCase):
    """Verify optional local inference independently of the stream worker."""

    def setUp(self) -> None:
        """Perform setUp."""
        patcher = patch.object(local_module, 'YOLO', MagicMock())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.detector = LocalYoloDetector(model_key='yolo11n')

    def test_initialisation_loads_local_model(self) -> None:
        """Ultralytics is loaded only when a local detector is requested."""
        self.assertEqual(self.detector.model_key, 'yolo11n')
        self.assertTrue(self.detector.use_ultralytics)
        self.assertTrue(hasattr(self.detector, 'ultralytics_model'))

    def test_lazy_optional_imports(self) -> None:
        """SAHI and Ultralytics remain deferred until local use."""
        factory_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        class DetectionFactory:
            """Provide DetectionFactory."""

            @staticmethod
            def from_pretrained(*args: object, **kwargs: object) -> str:
                """Perform from pretrained.

                Args:
                    *args: Value used by this callable.
                    **kwargs: Value used by this callable.

                Returns:
                    The callable result.
                """
                factory_calls.append((args, kwargs))
                return 'sahi-model'

        sahi_module = ModuleType('sahi')
        setattr(sahi_module, 'AutoDetectionModel', DetectionFactory)
        predict_module = ModuleType('sahi.predict')
        setattr(predict_module, 'get_sliced_prediction', lambda *args: args)
        ultralytics_module = ModuleType('ultralytics')

        class FakeYolo:
            """Provide FakeYolo."""

            pass

        setattr(ultralytics_module, 'YOLO', FakeYolo)
        with (
            patch.dict(
                sys.modules,
                {
                    'sahi': sahi_module,
                    'sahi.predict': predict_module,
                    'ultralytics': ultralytics_module,
                },
            ),
            patch.object(local_module, 'YOLO', None),
        ):
            self.assertEqual(
                local_module._LazyAutoDetectionModel.from_pretrained(
                    'model-path',
                    device='cpu',
                ),
                'sahi-model',
            )
            self.assertEqual(
                factory_calls,
                [(('model-path',), {'device': 'cpu'})],
            )
            self.assertEqual(
                local_module.get_sliced_prediction('frame', 'model'),
                ('frame', 'model'),
            )
            self.assertIs(local_module._yolo_class(), FakeYolo)

    async def test_generate_ultralytics_detections(self) -> None:
        """Ultralytics tracking rows are normalised for interactive callers."""
        boxes = MagicMock()
        boxes.__len__.return_value = 1
        boxes.xyxy.tolist.return_value = [[10, 10, 50, 50]]
        boxes.conf.tolist.return_value = [0.9]
        boxes.cls.tolist.return_value = [0]
        boxes.id = None
        result = MagicMock()
        result.boxes = boxes
        self.detector.ultralytics_model.track.return_value = [result]

        detections, tracked = await self.detector.generate_detections(
            np.zeros((4, 4, 3), dtype=np.uint8),
        )

        self.assertEqual(detections, [[10, 10, 50, 50, 0.9, 0]])
        self.assertEqual(tracked, [[10, 10, 50, 50, 0.9, 0, -1, 0]])

    async def test_detect_local_normalises_raw_ultralytics_boxes(self) -> None:
        """The one-frame local API preserves box, score, and class values."""
        boxes = MagicMock()
        boxes.__len__.return_value = 1
        boxes.xyxy[0].tolist.return_value = [1, 2, 3, 4]
        boxes.conf[0].item.return_value = 0.75
        boxes.cls[0].item.return_value = 5
        result = MagicMock()
        result.boxes = boxes
        self.detector.ultralytics_model.return_value = [result]

        detections = await self.detector._detect_local(
            np.zeros((2, 2, 3), dtype=np.uint8),
        )

        self.assertEqual(detections, [[1.0, 2.0, 3.0, 4.0, 0.75, 5]])

    async def test_generate_ultralytics_tracks_data_rows_and_motion(
        self,
    ) -> None:
        """Native tracker rows retain IDs and report motion above threshold."""
        boxes = MagicMock()
        boxes.__len__.return_value = 1
        boxes.data = np.array([[2, 2, 6, 6, 4, 0.8, 1]], dtype=float)
        result = MagicMock()
        result.boxes = boxes
        self.detector.prev_centers = {4: (0.0, 0.0)}
        self.detector.movement_thr_sq = 1.0
        self.detector.ultralytics_model.track.return_value = [result]

        detections, tracked = await self.detector.generate_detections(
            np.zeros((2, 2, 3), dtype=np.uint8),
        )

        self.assertEqual(detections, [[2.0, 2.0, 6.0, 6.0, 0.8, 1]])
        self.assertEqual(
            tracked,
            [[2.0, 2.0, 6.0, 6.0, 0.8, 1, 4, 1]],
        )

    async def test_generate_ultralytics_accepts_tensor_box_data(self) -> None:
        """GPU-like tensor box storage is moved to CPU before normalising."""
        boxes = MagicMock()
        boxes.__len__.return_value = 1
        boxes.data = local_module.torch.tensor(
            [[1, 1, 3, 3, 0.9, 2]],
        )
        result = MagicMock()
        result.boxes = boxes
        self.detector.ultralytics_model.track.return_value = [result]

        detections, _tracked = await self.detector.generate_detections(
            np.zeros((2, 2, 3), dtype=np.uint8),
        )

        self.assertEqual(detections[0][:4], [1.0, 1.0, 3.0, 3.0])
        self.assertAlmostEqual(detections[0][4], 0.9)
        self.assertEqual(detections[0][5], 2)

    async def test_local_preview_main_builds_detector_from_cli(self) -> None:
        """The preview CLI forwards parsed options to its detector."""
        detector = MagicMock()
        detector.run_detection = AsyncMock()
        with (
            patch(
                'src.local_yolo_detector.argparse.ArgumentParser.parse_args',
                return_value=MagicMock(
                    url='rtsp://camera',
                    model_key='yolo26n',
                    use_ultralytics=True,
                ),
            ),
            patch(
                'src.local_yolo_detector.LocalYoloDetector',
                return_value=detector,
            ) as detector_class,
        ):
            await local_module.main()

        detector_class.assert_called_once_with(
            model_key='yolo26n',
            use_ultralytics=True,
        )
        detector.run_detection.assert_awaited_once_with('rtsp://camera')

    async def test_generate_ultralytics_handles_empty_boxes(self) -> None:
        """An empty tracker result clears stale state without an error."""
        result = MagicMock()
        result.boxes = None
        self.detector.ultralytics_model.track.return_value = [result]
        with patch.object(self.detector, '_cleanup_prev_centers') as cleanup:
            assert await self.detector.generate_detections(
                np.zeros((2, 2, 3), dtype=np.uint8),
            ) == ([], [])
        cleanup.assert_called_once()

    async def test_sahi_detection_returns_untracked_rows(self) -> None:
        """SAHI works for MCP single-frame inference without tracker IDs."""
        with patch.object(
            local_module.AutoDetectionModel,
            'from_pretrained',
            return_value=MagicMock(),
        ):
            detector = LocalYoloDetector(
                model_key='yolo11n',
                use_ultralytics=False,
            )

        prediction = MagicMock()
        prediction.object_prediction_list = [
            MagicMock(
                category=MagicMock(id=5),
                bbox=MagicMock(to_voc_bbox=lambda: [1, 2, 3, 4]),
                score=MagicMock(value=0.77),
            ),
        ]
        with patch.object(
            local_module,
            'get_sliced_prediction',
            return_value=prediction,
        ):
            detections, tracked = await detector.generate_detections(
                np.zeros((4, 4, 3), dtype=np.uint8),
            )

        self.assertEqual(detections, [[1, 2, 3, 4, 0.77, 5]])
        self.assertEqual(tracked, [[1, 2, 3, 4, 0.77, 5, -1, 0]])

    async def test_cuda_oom_releases_model_and_returns_empty_results(
        self,
    ) -> None:
        """A local CUDA OOM does not leak resources or crash the caller."""
        self.detector.ultralytics_model.track.side_effect = RuntimeError(
            'CUDA out of memory',
        )
        with (
            patch.object(self.detector, '_release_local_model') as release,
            patch.object(self.detector, '_cleanup_prev_centers') as cleanup,
        ):
            detections, tracked = await self.detector.generate_detections(
                np.zeros((2, 2, 3), dtype=np.uint8),
            )

        self.assertEqual((detections, tracked), ([], []))
        release.assert_called_once()
        cleanup.assert_called_once()

    async def test_non_cuda_errors_are_propagated(self) -> None:
        """Only CUDA memory pressure is handled as a recoverable failure."""
        self.detector.ultralytics_model.track.side_effect = RuntimeError(
            'model file is corrupt',
        )
        with self.assertRaisesRegex(RuntimeError, 'model file is corrupt'):
            await self.detector.generate_detections(
                np.zeros((2, 2, 3), dtype=np.uint8),
            )

    def test_release_local_model_clears_gpu_resources(self) -> None:
        """Explicit cleanup removes model references and CUDA cache entries."""
        self.detector.model = object()
        with (
            patch(
                'src.local_yolo_detector.torch.cuda.is_available',
                return_value=True,
            ),
            patch(
                'src.local_yolo_detector.torch.cuda.empty_cache',
            ) as empty_cache,
            patch(
                'src.local_yolo_detector.torch.cuda.ipc_collect',
            ) as ipc_collect,
        ):
            self.detector._release_local_model()

        self.assertFalse(hasattr(self.detector, 'ultralytics_model'))
        self.assertFalse(hasattr(self.detector, 'model'))
        empty_cache.assert_called_once()
        ipc_collect.assert_called_once()

    def test_cleanup_removes_stale_local_tracks(self) -> None:
        """Unused local tracking centres are bounded by ``max_id_keep``."""
        self.detector.prev_centers = {1: (100, 100), 2: (200, 200)}
        self.detector.prev_centers_last_seen = {1: 1, 2: 2}
        self.detector.frame_count = 50
        self.detector.max_id_keep = 5

        self.detector._cleanup_prev_centers()

        self.assertEqual(self.detector.prev_centers, {})
        self.assertEqual(self.detector.prev_centers_last_seen, {})

    async def test_run_detection_releases_capture_when_opening_fails(
        self,
    ) -> None:
        """The optional desktop preview reports an unavailable source."""
        capture = MagicMock()
        capture.isOpened.return_value = False
        with patch(
            'src.local_yolo_detector.cv2.VideoCapture',
            return_value=capture,
        ):
            with self.assertRaisesRegex(ValueError, 'Failed to open stream'):
                await self.detector.run_detection('invalid')

    async def test_run_detection_draws_rows_and_releases_capture(self) -> None:
        """The preview loop handles a read failure and exits cleanly."""
        capture = MagicMock()
        capture.isOpened.return_value = True
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        capture.read.side_effect = [(False, None), (True, frame)]
        with (
            patch(
                'src.local_yolo_detector.cv2.VideoCapture',
                return_value=capture,
            ),
            patch.object(
                self.detector,
                'generate_detections',
                return_value=([], [[1, 1, 4, 4, 0.8, 1, 3, 1]]),
            ),
            patch(
                'src.local_yolo_detector.asyncio.sleep',
                new=AsyncMock(),
            ),
            patch(
                'src.local_yolo_detector.cv2.waitKey',
                return_value=ord('q'),
            ),
            patch('src.local_yolo_detector.cv2.imshow'),
            patch('src.local_yolo_detector.cv2.rectangle') as rectangle,
            patch('src.local_yolo_detector.cv2.putText') as put_text,
            patch('src.local_yolo_detector.cv2.destroyAllWindows') as destroy,
        ):
            await self.detector.run_detection('stream')

        capture.release.assert_called_once()
        rectangle.assert_called_once()
        put_text.assert_called_once()
        destroy.assert_called_once()

    async def test_close_releases_local_model(self) -> None:
        """Interactive callers can explicitly release their local detector."""
        with patch.object(self.detector, '_release_local_model') as release:
            await self.detector.close()
        release.assert_called_once()

    def test_local_preview_script_guard_invokes_async_main(self) -> None:
        """Running the module directly reaches its protected CLI entry."""
        script = local_module.Path(local_module.__file__).resolve()
        with patch.object(sys, 'argv', [str(script), '--help']):
            with self.assertRaises(SystemExit):
                runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    unittest.main()
