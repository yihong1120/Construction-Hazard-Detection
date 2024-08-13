import unittest
from unittest.mock import patch, MagicMock
from examples.YOLOv8_evaluation.evaluate_yolov8 import ModelEvaluator


class TestModelEvaluator(unittest.TestCase):
    def setUp(self):
        self.model_path = 'path/to/model.pt'
        self.data_path = 'path/to/data.yaml'
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

        # Verify that the val method was called with the correct data path
        mock_model.val.assert_called_once_with(data=self.data_path)

        # Check that the results are as expected
        self.assertEqual(results, expected_results)

if __name__ == '__main__':
    unittest.main()
