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
        self.model_path: str = 'models/pt/best_yolo11n.pt'
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
        'examples.YOLO_evaluation.evaluate_sahi_yolo.COCOeval',
    )
    @patch(
        'examples.YOLO_evaluation.evaluate_sahi_yolo.COCO',
    )
    @patch(
        'examples.YOLO_evaluation.evaluate_sahi_yolo.'
        'Coco.from_coco_dict_or_path',
    )
    @patch(
        'examples.YOLO_evaluation.evaluate_sahi_yolo.get_sliced_prediction',
    )
    @patch(
        'examples.YOLO_evaluation.evaluate_sahi_yolo.'
        'AutoDetectionModel.from_pretrained',
    )
    def test_evaluate(
        self,
        mock_auto_model: MagicMock,
        mock_get_sliced_prediction: MagicMock,
        mock_coco_from_path: MagicMock,
        mock_coco: MagicMock,
        mock_cocoeval: MagicMock,
    ) -> None:
        """
        Test the evaluate method for computing COCO metrics.
        """
        # Mock the model
        mock_model: MagicMock = MagicMock()
        mock_auto_model.return_value = mock_model

        # Mock COCO annotations and evaluation
        mock_coco_instance: MagicMock = MagicMock()

        # Define mock categories with specific names and ids
        mock_category1 = MagicMock()
        mock_category1.name = 'Hardhat'
        mock_category1.id = 1

        mock_category2 = MagicMock()
        mock_category2.name = 'Helmet'
        mock_category2.id = 2

        mock_coco_instance.categories = [mock_category1, mock_category2]

        # Define mock images with specific ids and file names
        mock_image1 = MagicMock()
        mock_image1.id = 101
        mock_image1.file_name = 'image1.jpg'

        mock_image2 = MagicMock()
        mock_image2.id = 102
        mock_image2.file_name = 'image2.jpg'

        mock_coco_instance.images = [mock_image1, mock_image2]

        mock_coco.return_value = mock_coco_instance

        # Mock the Coco.from_coco_dict_or_path to return the mock_coco_instance
        mock_coco_from_path.return_value = mock_coco_instance

        mock_eval_instance: MagicMock = MagicMock()
        # Simulate the precision and recall values
        mock_eval_instance.eval = {
            'precision': np.random.rand(10, 10, 10, 10, 10),
            'recall': np.random.rand(10, 10, 10, 10),
        }
        mock_cocoeval.return_value = mock_eval_instance

        # Mock the predictions with specific category names
        mock_pred1 = MagicMock()
        mock_pred1.category.name = 'Hardhat'
        mock_pred1.bbox.minx = 10
        mock_pred1.bbox.miny = 20
        mock_pred1.bbox.maxx = 110
        mock_pred1.bbox.maxy = 220
        mock_pred1.score.value = 0.9

        mock_pred2 = MagicMock()
        mock_pred2.category.name = 'Helmet'
        mock_pred2.bbox.minx = 15
        mock_pred2.bbox.miny = 25
        mock_pred2.bbox.maxx = 115
        mock_pred2.bbox.maxy = 225
        mock_pred2.score.value = 0.85

        mock_get_sliced_prediction.return_value.object_prediction_list = [
            mock_pred1,
            mock_pred2,
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
        'examples.YOLO_evaluation.evaluate_sahi_yolo.COCOEvaluator.evaluate',
    )
    @patch(
        'argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(
            model_path='models/pt/best_yolo11n.pt',
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
