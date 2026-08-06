from __future__ import annotations

import sys
import unittest
from types import ModuleType
from unittest.mock import patch

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


if __name__ == '__main__':
    unittest.main()
