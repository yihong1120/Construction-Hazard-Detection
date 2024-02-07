import unittest
from unittest.mock import patch, MagicMock
from train import YOLOModelHandler

class TestYOLOModelHandler(unittest.TestCase):
    """
    Test cases for train.py.
    """

    @patch('train.YOLO')
    def test_load_model(self, mock_yolo):
        """
        Test the load_model method to ensure it loads the model correctly.
        """
        # Mock the YOLO model constructor
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance

        # Initialise the YOLOModelHandler with a mock model name
        model_name = 'yolov8n.yaml'
        handler = YOLOModelHandler(model_name)

        # Assert that the YOLO constructor was called with the correct model name
        mock_yolo.assert_called_with(model_name)
        # Assert that the model attribute is set correctly
        self.assertEqual(handler.model, mock_model_instance)

    @patch('train.YOLO')
    def test_train_model(self, mock_yolo):
        """
        Test the train_model method to ensure it trains the model.
        """
        # Mock the YOLO model and its train method
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance

        # Initialise the YOLOModelHandler and call train_model
        handler = YOLOModelHandler('yolov8n.yaml')
        data_config = 'dataset/data.yaml'
        epochs = 10
        handler.train_model(data_config, epochs)

        # Assert that the train method was called with the correct arguments
        mock_model_instance.train.assert_called_with(data=data_config, epochs=epochs)

    @patch('train.YOLO')
    def test_validate_model(self, mock_yolo):
        """
        Test the validate_model method to ensure it validates the model.
        """
        # Mock the YOLO model and its val method
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance

        # Initialise the YOLOModelHandler and call validate_model
        handler = YOLOModelHandler('yolov8n.yaml')
        validation_results = handler.validate_model()

        # Assert that the val method was called
        mock_model_instance.val.assert_called_once()

    @patch('train.YOLO')
    def test_predict_image(self, mock_yolo):
        """
        Test the predict_image method to ensure it makes predictions.
        """
        # Mock the YOLO model and its prediction method
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance

        # Initialise the YOLOModelHandler and call predict_image
        handler = YOLOModelHandler('yolov8n.yaml')
        image_path = 'path/to/image.jpg'
        prediction_results = handler.predict_image(image_path)

        # Assert that the prediction method was called with the correct image path
        mock_model_instance.assert_called_with(image_path)

    @patch('train.YOLO')
    def test_export_model(self, mock_yolo):
        """
        Test the export_model method to ensure it exports the model.
        """
        # Mock the YOLO model and its export method
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance

        # Initialise the YOLOModelHandler and call export_model
        handler = YOLOModelHandler('yolov8n.yaml')
        export_format = 'onnx'
        export_path = handler.export_model(export_format)

        # Assert that the export method was called with the correct format
        mock_model_instance.export.assert_called_with(format=export_format)

    @patch('train.YOLO')
    @patch('train.torch.save')
    def test_save_model(self, mock_torch_save, mock_yolo):
        """
        Test the save_model method to ensure it saves the model.
        """
        # Mock the YOLO model and its state_dict method
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance

        # Initialise the YOLOModelHandler and call save_model
        handler = YOLOModelHandler('yolov8n.yaml')
        save_path = 'path/to/model.pt'
        handler.save_model(save_path)

        # Assert that the torch.save method was called with the correct arguments
        mock_torch_save.assert_called_with(mock_model_instance.state_dict(), save_path)

if __name__ == '__main__':
    unittest.main()