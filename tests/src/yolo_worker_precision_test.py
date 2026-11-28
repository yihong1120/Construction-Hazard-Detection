from __future__ import annotations

import unittest
from unittest.mock import patch

from src import yolo_worker


class TestYoloWorkerPrecisionCoverage(unittest.TestCase):
    """Provide TestYoloWorkerPrecisionCoverage."""

    def test_precision_parser_handles_legacy_and_invalid_values(self) -> None:
        """Test precision parser handles legacy and invalid values."""
        self.assertIsNone(yolo_worker._parse_worker_precision(None))
        for value in ['', ' none ', 'null', 'default', 'auto', 'legacy']:
            self.assertIsNone(yolo_worker._parse_worker_precision(value))
        with self.assertRaisesRegex(
            ValueError,
            'Unsupported YOLO_WORKER_PRECISION',
        ):
            yolo_worker._parse_worker_precision('bf16')

    def test_precision_config_sets_predict_options_and_rejects_unknown_modes(
        self,
    ) -> None:
        self.assertEqual(
            yolo_worker._worker_precision_config('f32'), {'quantize': 32},
        )
        self.assertEqual(
            yolo_worker._worker_precision_config('f16'), {'quantize': 16},
        )
        self.assertEqual(
            yolo_worker._worker_precision_config('int8'),
            {'rect': False},
        )
        with self.assertRaisesRegex(AssertionError, 'unhandled'):
            yolo_worker._worker_precision_config('bf16')

    def test_worker_uses_selected_precision_configuration(self) -> None:
        """Test worker uses selected precision configuration."""
        with patch.dict(
            'os.environ',
            {'YOLO_WORKER_PRECISION': 'int8'},
            clear=False,
        ):
            worker = yolo_worker.YoloWorker(None)

        self.assertEqual(worker.precision_mode, 'int8')
        self.assertEqual(worker.precision_label, 'int8')
        self.assertEqual(worker.precision_args, {'rect': False})

    def test_worker_uses_legacy_precision_configuration(self) -> None:
        """Test worker uses legacy precision configuration."""
        environment = {
            'YOLO_WORKER_PRECISION': '',
            'YOLO_WORKER_HALF': 'false',
            'YOLO_WORKER_QUANTIZE': '32',
        }
        with patch.dict('os.environ', environment, clear=False):
            with patch.object(
                yolo_worker,
                'precision_kwargs',
                return_value={'quantize': 32},
            ) as precision_kwargs:
                worker = yolo_worker.YoloWorker(None)

        self.assertIsNone(worker.precision_mode)
        self.assertEqual(worker.precision_label, 'legacy')
        self.assertEqual(worker.precision_args, {'quantize': 32})
        precision_kwargs.assert_called_once_with(False, 32)


if __name__ == '__main__':
    unittest.main()
