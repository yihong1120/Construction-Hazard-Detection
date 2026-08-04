from __future__ import annotations

import sys
import unittest
from types import ModuleType
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

import src.yolo_detector as detector_module


class TestYoloDetectorResilience(unittest.IsolatedAsyncioTestCase):
    """Exercise lazy imports and local-inference recovery behaviour."""

    def setUp(self) -> None:
        self.detector = detector_module.YoloDetector(detect_with_server=True)

    def test_lazy_external_import_wrappers(self) -> None:
        """Optional SAHI, SciPy, and Ultralytics imports stay deferred."""
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
        scipy_module = ModuleType('scipy')
        optimize_module = ModuleType('scipy.optimize')
        setattr(
            optimize_module,
            'linear_sum_assignment',
            lambda cost: ('rows', cost),
        )
        ultralytics_module = ModuleType('ultralytics')

        class FakeYolo:
            pass

        setattr(ultralytics_module, 'YOLO', FakeYolo)

        with patch.dict(
            sys.modules,
            {
                'sahi': sahi_module,
                'sahi.predict': predict_module,
                'scipy': scipy_module,
                'scipy.optimize': optimize_module,
                'ultralytics': ultralytics_module,
            },
        ), patch.object(detector_module, 'YOLO', None):
            self.assertEqual(
                detector_module._LazyAutoDetectionModel.from_pretrained(
                    'model-path',
                    device='cpu',
                ),
                'sahi-model',
            )
            self.assertEqual(
                factory_calls, [(('model-path',), {'device': 'cpu'})],
            )
            self.assertEqual(
                detector_module.get_sliced_prediction('frame', 'model'),
                ('frame', 'model'),
            )
            self.assertEqual(
                detector_module.linear_sum_assignment('cost'),
                ('rows', 'cost'),
            )
            self.assertIs(detector_module._yolo_class(), FakeYolo)

    async def test_local_cuda_oom_releases_model_and_returns_empty_result(
        self,
    ) -> None:
        """A local CUDA OOM frees resources without ending the camera loop."""
        self.detector.detect_with_server = False
        self.detector.ultralytics_model = MagicMock()
        self.detector.ultralytics_model.track.side_effect = RuntimeError(
            'CUDA out of memory',
        )

        with (
            patch.object(self.detector, '_release_local_model') as release,
            patch.object(self.detector, '_cleanup_prev_centers') as cleanup,
        ):
            datas, tracked = await self.detector.generate_detections(
                np.zeros((2, 2, 3), dtype=np.uint8),
            )

        self.assertEqual((datas, tracked), ([], []))
        release.assert_called_once()
        cleanup.assert_called_once()
        self.assertFalse(
            self.detector._is_cuda_oom(
                RuntimeError('network down'),
            ),
        )

    async def test_local_non_cuda_errors_are_not_silenced(self) -> None:
        """Only CUDA memory pressure is recoverable in the local detector."""
        self.detector.detect_with_server = False
        self.detector.ultralytics_model = MagicMock()
        self.detector.ultralytics_model.track.side_effect = RuntimeError(
            'model file is corrupt',
        )

        with self.assertRaisesRegex(RuntimeError, 'model file is corrupt'):
            await self.detector.generate_detections(
                np.zeros((2, 2, 3), dtype=np.uint8),
            )

    def test_release_local_model_clears_gpu_resources(self) -> None:
        """Explicit release removes model references and clears CUDA caches."""
        self.detector.ultralytics_model = object()
        self.detector.model = object()

        with (
            patch(
                'src.yolo_detector.torch.cuda.is_available',
                return_value=True,
            ),
            patch('src.yolo_detector.torch.cuda.empty_cache') as empty_cache,
            patch('src.yolo_detector.torch.cuda.ipc_collect') as ipc_collect,
        ):
            self.detector._release_local_model()

        self.assertFalse(hasattr(self.detector, 'ultralytics_model'))
        self.assertFalse(hasattr(self.detector, 'model'))
        empty_cache.assert_called_once()
        ipc_collect.assert_called_once()

    def test_tracking_helpers_handle_empty_and_non_overlapping_inputs(
        self,
    ) -> None:
        """Tracking math handles empty matrices and disjoint boxes safely."""
        self.assertEqual(
            self.detector._bbox_iou((0, 0, 1, 1), (2, 2, 3, 3)),
            0.0,
        )
        matrix = self.detector._build_group_cost_matrix([], [], [])
        self.assertEqual(matrix.shape, (0, 0))
        rows = [[1, 2, 3, 4, 0.9, 1]]
        self.assertIs(self.detector._without_indices(rows, set()), rows)


if __name__ == '__main__':
    unittest.main()
