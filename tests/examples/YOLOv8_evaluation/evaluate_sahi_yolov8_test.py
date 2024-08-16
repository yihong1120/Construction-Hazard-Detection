from __future__ import annotations

import argparse
import unittest
from unittest.mock import patch, MagicMock
from examples.YOLOv8_evaluation.evaluate_sahi_yolov8 import COCOEvaluator, main


class TestCOCOEvaluator(unittest.TestCase):
    def setUp(self):
        self.model_path = 'models/pt/best_yolov8n.pt'
        self.coco_json = 'tests/dataset/coco_annotations.json'
        self.image_dir = 'tests/dataset/val/images'
        self.evaluator = COCOEvaluator(
            model_path=self.model_path,
            coco_json=self.coco_json,
            image_dir=self.image_dir,
        )

    @patch('examples.YOLOv8_evaluation.evaluate_sahi_yolov8.AutoDetectionModel.from_pretrained')
    @patch('examples.YOLOv8_evaluation.evaluate_sahi_yolov8.get_sliced_prediction')
    @patch('examples.YOLOv8_evaluation.evaluate_sahi_yolov8.Coco.from_coco_dict_or_path')
    @patch('examples.YOLOv8_evaluation.evaluate_sahi_yolov8.COCO')
    @patch('examples.YOLOv8_evaluation.evaluate_sahi_yolov8.COCOeval')
    def test_evaluate(self, mock_cocoeval, mock_coco, mock_coco_from_path, mock_get_sliced_prediction, mock_auto_model):
        """
        Test the evaluate method for computing COCO metrics.
        """
        # Mock the model
        mock_model = MagicMock()
        mock_auto_model.return_value = mock_model

        # Mock COCO annotations and evaluation
        mock_coco_instance = MagicMock()
        mock_coco.return_value = mock_coco_instance

        mock_coco_from_instance = MagicMock()
        mock_coco_from_path.return_value = mock_coco_from_instance

        mock_eval_instance = MagicMock()
        mock_cocoeval.return_value = mock_eval_instance

        # Mock the predictions
        mock_get_sliced_prediction.return_value.object_prediction_list = [
            MagicMock(
                category=MagicMock(name='Hardhat'),
                bbox=MagicMock(minx=10, miny=20, maxx=110, maxy=220),
                score=MagicMock(value=0.9),
            )
        ]

        # Run the evaluation
        metrics = self.evaluator.evaluate()

        # Assert that the predictions and COCO evaluation were called
        mock_auto_model.assert_called_once_with(
            model_type='yolov8',
            model_path=self.model_path,
            confidence_threshold=0.3,
        )
        mock_coco_from_path.assert_called_once_with(self.coco_json)
        mock_get_sliced_prediction.assert_called()
        mock_cocoeval.assert_called_once_with(mock_coco_instance, mock_coco_instance.loadRes.return_value, 'bbox')
        mock_eval_instance.evaluate.assert_called_once()
        mock_eval_instance.accumulate.assert_called_once()
        mock_eval_instance.summarize.assert_called_once()

        # Verify the returned metrics
        self.assertIn('Average Precision', metrics)
        self.assertIn('Average Recall', metrics)
        self.assertIn('mAP at IoU=50', metrics)
        self.assertIn('mAP at IoU=50-95', metrics)

    @patch('examples.YOLOv8_evaluation.evaluate_sahi_yolov8.COCOEvaluator.evaluate')
    @patch('argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(
        model_path='models/pt/best_yolov8n.pt',
        coco_json='tests/dataset/coco_annotations.json',
        image_dir='tests/dataset/val/images'
    ))
    def test_main(self, mock_parse_args, mock_evaluate):
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
            mock_print.assert_any_call('Evaluation metrics:', {
                'Average Precision': 0.5,
                'Average Recall': 0.6,
                'mAP at IoU=50': 0.7,
                'mAP at IoU=50-95': 0.8,
            })

if __name__ == '__main__':
    unittest.main()
