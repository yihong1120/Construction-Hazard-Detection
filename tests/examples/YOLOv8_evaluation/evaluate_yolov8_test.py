from __future__ import annotations

import sys
import unittest
from io import StringIO
from unittest.mock import MagicMock
from unittest.mock import patch

from examples.YOLOv8_evaluation.evaluate_yolov8 import main
from examples.YOLOv8_evaluation.evaluate_yolov8 import ModelEvaluator


class TestModelEvaluator(unittest.TestCase):
    def setUp(self) -> None:
        self.model_path: str = 'models/pt/best_yolov8n.pt'
        self.data_path: str = 'tests/dataset/data.yaml'
        self.evaluator: ModelEvaluator | None = None

    def tearDown(self) -> None:
        """
        Clean up after each test.
        """
        self.evaluator = None

    @patch('examples.YOLOv8_evaluation.evaluate_yolov8.YOLO')
    def test_evaluate(self, mock_yolo: MagicMock) -> None:
        """
        Test the evaluate method to ensure it returns the expected results.
        """
        # Create a mock model and mock return value for the val method
        mock_model: MagicMock = MagicMock()
        mock_yolo.return_value = mock_model
        expected_results: dict[str, str] = {
            'metrics': 'some_evaluation_results',
        }
        mock_model.val.return_value = expected_results

        # Initialise the ModelEvaluator after mocking YOLO
        self.evaluator = ModelEvaluator(
            model_path=self.model_path,
            data_path=self.data_path,
        )

        # Call the evaluate method
        results: dict[str, str] = self.evaluator.evaluate()

        # Verify that the YOLO class was instantiated
        # with the correct model path
        mock_yolo.assert_called_once_with(self.model_path)
        self.assertEqual(results, expected_results)

    @patch('examples.YOLOv8_evaluation.evaluate_yolov8.ModelEvaluator')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main(
        self,
        mock_parse_args: MagicMock,
        mock_model_evaluator: MagicMock,
    ) -> None:
        """
        Test the main function.
        """
        # Mock the command line arguments
        mock_parse_args.return_value = MagicMock(
            model_path=self.model_path,
            data_path=self.data_path,
        )

        # Mock the ModelEvaluator and its evaluate method
        mock_evaluator_instance: MagicMock = MagicMock()
        mock_model_evaluator.return_value = mock_evaluator_instance
        mock_evaluator_instance.evaluate.return_value = {
            'metrics': 'some_evaluation_results',
        }

        # Capture the output
        captured_output: StringIO = StringIO()
        sys.stdout = captured_output

        # Call the main function
        main()

        # Verify the output
        self.assertIn('some_evaluation_results', captured_output.getvalue())

        # Reset stdout
        sys.stdout = sys.__stdout__


if __name__ == '__main__':
    unittest.main()
