from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock
from io import StringIO
import sys
from examples.YOLOv8_evaluation.evaluate_yolov8 import ModelEvaluator, main


class TestModelEvaluator(unittest.TestCase):
    def setUp(self):
        self.model_path = 'models/pt/best_yolov8n.pt'
        self.data_path = 'tests/dataset/data.yaml'
        self.evaluator = ModelEvaluator(
            model_path=self.model_path,
            data_path=self.data_path,
        )

    @patch('examples.YOLOv8_evaluation.evaluate_yolov8.YOLO')
    def test_evaluate(self, mock_yolo):
        """
        Test the evaluate method to ensure it returns the expected results.
        """
        # Create a mock model and mock return value for the val method
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        expected_results = {'metrics': 'some_evaluation_results'}
        mock_model.val.return_value = expected_results
    
        # Call the evaluate method
        results = self.evaluator.evaluate()
    
        # Verify that the YOLO class was instantiated with the correct model path
        mock_yolo.assert_called_once_with(self.model_path)
        self.assertEqual(results, expected_results)

    @patch('examples.YOLOv8_evaluation.evaluate_yolov8.ModelEvaluator')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main(self, mock_parse_args, mock_model_evaluator):
        # Mock the command line arguments
        mock_parse_args.return_value = MagicMock(
            model_path='models/pt/best_yolov8n.pt',
            data_path='tests/dataset/data.yaml'
        )
    
        # Mock the ModelEvaluator and its evaluate method
        mock_evaluator_instance = MagicMock()
        mock_model_evaluator.return_value = mock_evaluator_instance
        mock_evaluator_instance.evaluate.return_value = {'metrics': 'some_evaluation_results'}
    
        # Capture the output
        captured_output = StringIO()
        sys.stdout = captured_output
    
        # Call the main function
        main()
    
        # Verify the output
        self.assertIn('some_evaluation_results', captured_output.getvalue())

if __name__ == '__main__':
    unittest.main()
