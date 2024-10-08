from __future__ import annotations

import argparse
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

from examples.YOLO_evaluation.evaluate_sahi_yolo import COCOEvaluator
from examples.YOLO_evaluation.evaluate_sahi_yolo import main


class TestCOCOEvaluator(unittest.TestCase):
    def setUp(self) -> None:
        self.model_path: str = 'models/pt/best_yolov8n.pt'
        self.coco_json: str = 'tests/dataset/coco_annotations.json'
        self.image_dir: str = 'tests/dataset/val/images'
        self.evaluator = COCOEvaluator(
            model_path=self.model_path,
            coco_json=self.coco_json,
            image_dir=self.image_dir,
        )

    def tearDown(self) -> None:
        """
        Clean up after each test.
        """
        del self.evaluator

    @patch(
        'examples.YOLO_evaluation.evaluate_sahi_yolo.'
        'AutoDetectionModel.from_pretrained',
    )
    @patch(
        'examples.YOLO_evaluation.evaluate_sahi_yolo.'
        'get_sliced_prediction',
    )
    @patch(
        'examples.YOLO_evaluation.evaluate_sahi_yolo.'
        'Coco.from_coco_dict_or_path',
    )
    @patch('examples.YOLO_evaluation.evaluate_sahi_yolo.COCO')
    @patch('examples.YOLO_evaluation.evaluate_sahi_yolo.COCOeval')
    def test_evaluate(
        self,
        mock_cocoeval: MagicMock,
        mock_coco: MagicMock,
        mock_coco_from_path: MagicMock,
        mock_get_sliced_prediction: MagicMock,
        mock_auto_model: MagicMock,
    ) -> None:
        """
        Test the evaluate method for computing COCO metrics.
        """
        # Mock the model
        mock_model: MagicMock = MagicMock()
        mock_auto_model.return_value = mock_model

        # Mock COCO annotations and evaluation
        mock_coco_instance: MagicMock = MagicMock()
        mock_coco.return_value = mock_coco_instance

        mock_coco_from_instance: MagicMock = MagicMock()
        mock_coco_from_path.return_value = mock_coco_from_instance

        mock_eval_instance: MagicMock = MagicMock()
        # Simulate the precision and recall values
        mock_eval_instance.eval = {
            'precision': np.random.rand(10, 10, 10, 10, 10),
            'recall': np.random.rand(10, 10, 10, 10),
        }
        mock_cocoeval.return_value = mock_eval_instance

        # Mock the predictions
        mock_get_sliced_prediction.return_value.object_prediction_list = [
            MagicMock(
                category=MagicMock(name='Hardhat'),
                bbox=MagicMock(minx=10, miny=20, maxx=110, maxy=220),
                score=MagicMock(value=0.9),
            ),
        ]

        # Run the evaluation
        metrics: dict[str, float] = self.evaluator.evaluate()

        # Verify that the metrics returned
        # by the evaluate method are as expected
        expected_metrics: dict[str, float] = {
            'Average Precision': np.mean(
                mock_eval_instance.eval['precision'][:, :, :, 0, -1],
            ),
            'Average Recall': np.mean(
                mock_eval_instance.eval['recall'][:, :, 0, -1],
            ),
            'mAP at IoU=50': np.mean(
                mock_eval_instance.eval['precision'][0, :, :, 0, 2],
            ),
            'mAP at IoU=50-95': np.mean(
                mock_eval_instance.eval['precision'][0, :, :, 0, :],
            ),
        }

        self.assertEqual(metrics, expected_metrics)

    @patch(
        'examples.YOLO_evaluation.evaluate_sahi_yolo.'
        'COCOEvaluator.evaluate',
    )
    @patch(
        'argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(
            model_path='models/pt/best_yolov8n.pt',
            coco_json='tests/dataset/coco_annotations.json',
            image_dir='tests/dataset/val/images',
        ),
    )
    def test_main(
        self,
        mock_parse_args: MagicMock,
        mock_evaluate: MagicMock,
    ) -> None:
        """
        Test the main function.
        """
        mock_evaluate.return_value = {
            'Average Precision': 0.5,
            'Average Recall': 0.6,
            'mAP at IoU=50': 0.7,
            'mAP at IoU=50-95': 0.8,
        }

        with patch('builtins.print') as mock_print:
            main()
            mock_print.assert_any_call(
                'Evaluation metrics:', {
                    'Average Precision': 0.5,
                    'Average Recall': 0.6,
                    'mAP at IoU=50': 0.7,
                    'mAP at IoU=50-95': 0.8,
                },
            )


if __name__ == '__main__':
    unittest.main()
