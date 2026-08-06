from __future__ import annotations

import sys
import unittest
from types import ModuleType
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

import src.local_yolo_detector as local_module
from src.local_yolo_detector import LocalYoloDetector


class TestLocalYoloDetector(unittest.IsolatedAsyncioTestCase):
    """Verify optional local inference independently of the stream worker."""

    def setUp(self) -> None:
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
            @staticmethod
            def from_pretrained(*args: object, **kwargs: object) -> str:
                factory_calls.append((args, kwargs))
                return 'sahi-model'

        sahi_module = ModuleType('sahi')
        setattr(sahi_module, 'AutoDetectionModel', DetectionFactory)
        predict_module = ModuleType('sahi.predict')
        setattr(predict_module, 'get_sliced_prediction', lambda *args: args)
        ultralytics_module = ModuleType('ultralytics')

        class FakeYolo:
            pass

        setattr(ultralytics_module, 'YOLO', FakeYolo)
        with patch.dict(
            sys.modules,
            {
                'sahi': sahi_module,
                'sahi.predict': predict_module,
                'ultralytics': ultralytics_module,
            },
        ), patch.object(local_module, 'YOLO', None):
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


if __name__ == '__main__':
    unittest.main()
