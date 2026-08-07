from __future__ import annotations

import sys
import unittest
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

import numpy as np

import src.yolo_detector as detector_module


class TestYoloDetectorResilience(unittest.TestCase):
    """Exercise the lightweight shared-worker detector helpers."""

    def setUp(self) -> None:
        self.detector = detector_module.YoloDetector()

    def test_linear_assignment_import_stays_deferred(self) -> None:
        """SciPy is imported only when Hungarian tracking needs it."""
        scipy_module = ModuleType('scipy')
        optimize_module = ModuleType('scipy.optimize')
        setattr(
            optimize_module,
            'linear_sum_assignment',
            lambda cost: ('rows', cost),
        )

        with patch.dict(
            sys.modules,
            {'scipy': scipy_module, 'scipy.optimize': optimize_module},
        ):
            self.assertEqual(
                detector_module.linear_sum_assignment('cost'),
                ('rows', 'cost'),
            )

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

    def test_local_detection_requires_compatibility_detector(self) -> None:
        """Server-mode callers receive an actionable local-inference error."""
        with self.assertRaisesRegex(RuntimeError, 'Local inference'):
            import asyncio

            asyncio.run(
                self.detector._detect_local(
                    np.zeros((2, 2, 3), dtype=np.uint8),
                ),
            )

    def test_local_compatibility_close_and_main_delegate(self) -> None:
        """Test local detector compatibility and preview CLI delegation."""
        local_detector = ModuleType('src.local_yolo_detector')
        setattr(local_detector, 'LocalYoloDetector', lambda **_kwargs: None)
        setattr(local_detector, 'main', AsyncMock())
        local = SimpleNamespace(close=AsyncMock())
        self.detector._local_detector = local

        import asyncio

        asyncio.run(self.detector.close())
        local.close.assert_awaited_once()
        with patch.dict(
            sys.modules,
            {'src.local_yolo_detector': local_detector},
        ):
            asyncio.run(detector_module.main())
        getattr(local_detector, 'main').assert_awaited_once()

    def test_local_delegation_tracking_and_worker_close_paths(self) -> None:
        """Compatibility calls delegate cleanly and worker clients close."""
        import asyncio

        local = SimpleNamespace(
            _detect_local=AsyncMock(return_value=[[1, 2, 3, 4, 0.9, 1]]),
        )
        self.detector._local_detector = local
        assert asyncio.run(
            self.detector._detect_local(np.zeros((2, 2, 3), dtype=np.uint8)),
        ) == [[1, 2, 3, 4, 0.9, 1]]

        self.detector._local_detector = None
        self.assertEqual(self.detector.frame_count, 0)
        self.detector.track_detections([])
        self.assertEqual(self.detector.frame_count, 1)

        self.detector.worker_client = SimpleNamespace(close=AsyncMock())
        asyncio.run(self.detector.close())
        self.detector.worker_client.close.assert_awaited_once()

    def test_hungarian_assignment_skips_invalid_solver_edge(self) -> None:
        """A solver result on a blocked edge never becomes a tracker match."""
        cost = np.array([[0.1, 2.0], [0.2, 0.3]])
        with patch(
            'src.yolo_detector._linear_sum_assignment',
            return_value=lambda _cost: (
                np.array([0, 1]),
                np.array([1, 0]),
            ),
        ):
            matches, _rows, _cols = self.detector._hungarian_assign(
                cost,
                cost_threshold=1.0,
            )

        self.assertEqual(matches, [(1, 0)])


if __name__ == '__main__':
    unittest.main()
