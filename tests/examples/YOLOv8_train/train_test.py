from __future__ import annotations

import argparse
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from examples.YOLOv8_train.train import main
from examples.YOLOv8_train.train import YOLOModelHandler


class TestYOLOModelHandler(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the test environment before each test.
        """
        self.model_name: str = 'models/pt/best_yolov8x.pt'
        self.handler: YOLOModelHandler = YOLOModelHandler(self.model_name)

    def tearDown(self) -> None:
        """
        Clean up after each test.
        """
        # Delete the handler object
        if hasattr(self, 'handler'):
            del self.handler

    @patch('examples.YOLOv8_train.train.YOLO')
    def test_load_model_with_yaml(
        self,
        mock_yolo: unittest.mock.MagicMock,
    ) -> None:
        """
        Test loading a model from a YAML file.
        """
        mock_yolo.return_value = MagicMock()
        handler = YOLOModelHandler('models/config.yaml')
        mock_yolo.assert_called_with('models/config.yaml')
        self.assertIsNotNone(handler.model)

    @patch('examples.YOLOv8_train.train.YOLO')
    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    def test_load_model_with_pt_and_device_selection(
        self,
        mock_mps_available: unittest.mock.MagicMock,
        mock_cuda_available: unittest.mock.MagicMock,
        mock_yolo: unittest.mock.MagicMock,
    ) -> None:
        """
        Test loading a model from a .pt file and selecting the device.
        """
        mock_yolo.return_value = MagicMock()

        # Case 1: MPS available
        mock_mps_available.return_value = True
        mock_cuda_available.return_value = False
        handler = YOLOModelHandler('models/pt/best_yolov8x.pt')
        mock_yolo.assert_called_with('models/pt/best_yolov8x.pt')
        self.assertEqual(handler.device.type, 'mps')

        # Case 2: CUDA available but MPS is not
        mock_mps_available.return_value = False
        mock_cuda_available.return_value = True
        handler = YOLOModelHandler('models/pt/best_yolov8x.pt')
        mock_yolo.assert_called_with('models/pt/best_yolov8x.pt')
        self.assertEqual(handler.device.type, 'cuda')

        # Case 3: Neither MPS nor CUDA is available, should fall back to CPU
        mock_mps_available.return_value = False
        mock_cuda_available.return_value = False
        handler = YOLOModelHandler('models/pt/best_yolov8x.pt')
        mock_yolo.assert_called_with('models/pt/best_yolov8x.pt')
        self.assertEqual(handler.device.type, 'cpu')

    @patch('examples.YOLOv8_train.train.YOLO')
    def test_load_model(self, mock_yolo: unittest.mock.MagicMock) -> None:
        """
        Test loading a model.
        """
        mock_yolo.return_value = MagicMock()
        handler = YOLOModelHandler(self.model_name)
        mock_yolo.assert_called_with(self.model_name)
        self.assertIsNotNone(handler.model)

    def test_load_model_with_unsupported_format(self) -> None:
        """
        Test that loading a model with an unsupported format raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            _ = YOLOModelHandler('unsupported_format.txt')
        self.assertEqual(
            str(context.exception),
            "Unsupported model format. Use '.yaml' or '.pt'",
        )

    @patch('examples.YOLOv8_train.train.YOLO')
    def test_train_model(self, mock_yolo: unittest.mock.MagicMock) -> None:
        """
        Test training the model.
        """
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        handler = YOLOModelHandler(self.model_name)
        handler.train_model(
            data_config='tests/cv_dataset/data.yaml',
            epochs=10,
            optimizer='auto',
        )
        mock_model.train.assert_called_with(
            data='tests/cv_dataset/data.yaml',
            epochs=10,
            batch=handler.batch_size,
            optimizer='auto',
        )

    @patch('examples.YOLOv8_train.train.YOLO')
    def test_train_model_without_loading(
        self,
        mock_yolo: unittest.mock.MagicMock,
    ) -> None:
        """
        Test that training without loading a model raises RuntimeError.
        """
        handler = YOLOModelHandler('models/pt/best_yolov8x.pt')
        handler.model = None  # Simulate that model is not loaded properly
        with self.assertRaises(RuntimeError) as context:
            handler.train_model('dataset/data.yaml', 10, 'auto')
        self.assertEqual(
            str(context.exception),
            'The model is not loaded properly.',
        )

    @patch('examples.YOLOv8_train.train.YOLO')
    def test_validate_model_without_loading(
        self,
        mock_yolo: unittest.mock.MagicMock,
    ) -> None:
        """
        Test that validating without loading a model raises RuntimeError.
        """
        handler = YOLOModelHandler('models/pt/best_yolov8x.pt')
        handler.model = None  # Simulate that model is not loaded properly
        with self.assertRaises(RuntimeError) as context:
            handler.validate_model()
        self.assertEqual(
            str(context.exception),
            'The model is not loaded properly.',
        )

    @patch('examples.YOLOv8_train.train.YOLO')
    def test_predict_image_without_loading(
        self,
        mock_yolo: unittest.mock.MagicMock,
    ) -> None:
        """
        Test that predicting without loading a model raises RuntimeError.
        """
        handler = YOLOModelHandler('models/pt/best_yolov8x.pt')
        handler.model = None  # Simulate that model is not loaded properly
        with self.assertRaises(RuntimeError) as context:
            handler.predict_image('path/to/image.jpg')
        self.assertEqual(
            str(context.exception),
            'The model is not loaded properly.',
        )

    @patch('examples.YOLOv8_train.train.YOLO')
    def test_export_model_without_loading(
        self,
        mock_yolo: unittest.mock.MagicMock,
    ) -> None:
        """
        Test that exporting without loading a model raises RuntimeError.
        """
        handler = YOLOModelHandler('models/pt/best_yolov8x.pt')
        handler.model = None  # Simulate that model is not loaded properly
        with self.assertRaises(RuntimeError) as context:
            handler.export_model('onnx')
        self.assertEqual(
            str(context.exception),
            'The model is not loaded properly.',
        )

    @patch('examples.YOLOv8_train.train.YOLO')
    def test_save_model_without_loading(
        self,
        mock_yolo: unittest.mock.MagicMock,
    ) -> None:
        """
        Test that saving without loading a model raises RuntimeError.
        """
        handler = YOLOModelHandler('models/pt/best_yolov8x.pt')
        handler.model = None  # Simulate that model is not loaded properly
        with self.assertRaises(RuntimeError) as context:
            handler.save_model('path/to/save/model.pt')
        self.assertEqual(
            str(context.exception),
            'The model is not loaded properly.',
        )

    @patch('examples.YOLOv8_train.train.YOLO')
    def test_cross_validate_model_without_loading(
        self, mock_yolo: unittest.mock.MagicMock,
    ) -> None:
        """
        Test that cross-validation without loading a model raises RuntimeError.
        """
        handler = YOLOModelHandler('models/pt/best_yolov8x.pt')
        handler.model = None  # Simulate that model is not loaded properly
        with self.assertRaises(RuntimeError) as context:
            handler.cross_validate_model('dataset/data.yaml', 10, 'auto')
        self.assertEqual(
            str(context.exception),
            'The model is not loaded properly.',
        )

    @patch('examples.YOLOv8_train.train.YOLO')
    def test_validate_model(self, mock_yolo: unittest.mock.MagicMock) -> None:
        """
        Test validating the model.
        """
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        handler = YOLOModelHandler(self.model_name)
        handler.validate_model()
        mock_model.val.assert_called_with(batch=handler.batch_size)

    @patch('examples.YOLOv8_train.train.YOLO')
    def test_predict_image(self, mock_yolo: unittest.mock.MagicMock) -> None:
        """
        Test predicting an image.
        """
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        handler = YOLOModelHandler(self.model_name)
        handler.predict_image('path/to/image.jpg')
        mock_model.assert_called_with('path/to/image.jpg')

    @patch('examples.YOLOv8_train.train.YOLO')
    def test_export_model(self, mock_yolo: unittest.mock.MagicMock) -> None:
        """
        Test exporting the model.
        """
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        handler = YOLOModelHandler(self.model_name)
        handler.export_model('onnx')
        mock_model.export.assert_called_with(format='onnx')

    @patch('examples.YOLOv8_train.train.torch.save')
    @patch('examples.YOLOv8_train.train.YOLO')
    def test_save_model(
        self,
        mock_yolo: unittest.mock.MagicMock,
        mock_save: unittest.mock.MagicMock,
    ) -> None:
        """
        Test saving the model.
        """
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        handler = YOLOModelHandler(self.model_name)
        handler.save_model('path/to/save/model.pt')
        mock_save.assert_called_with(
            mock_model.state_dict(), 'path/to/save/model.pt',
        )

    @patch('examples.YOLOv8_train.train.KFold.split')
    @patch('examples.YOLOv8_train.train.YOLO')
    def test_cross_validate_model(
        self,
        mock_yolo: unittest.mock.MagicMock,
        mock_kfold_split: unittest.mock.MagicMock,
    ) -> None:
        """
        Test cross-validation of the model.
        """
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        mock_kfold_split.return_value = [([0, 1], [2]), ([2], [0, 1])]
        handler = YOLOModelHandler(self.model_name)
        handler.cross_validate_model(
            data_config='tests/cv_dataset/data.yaml',
            epochs=10, optimizer='auto',
            n_splits=2,
        )

        self.assertEqual(mock_model.train.call_count, 2)
        self.assertEqual(mock_model.val.call_count, 2)

    @patch('examples.YOLOv8_train.train.AutoDetectionModel.from_pretrained')
    @patch('examples.YOLOv8_train.train.get_sliced_prediction')
    def test_predict_image_sahi(
        self,
        mock_sahi_predict: unittest.mock.MagicMock,
        mock_auto_detection_model: unittest.mock.MagicMock,
    ) -> None:
        """
        Test predicting an image using SAHI.
        """
        mock_model = MagicMock()
        mock_auto_detection_model.return_value = mock_model
        mock_sahi_predict.return_value = MagicMock(
            object_prediction_list=[{'label': 'test', 'score': 0.9}],
        )

        result = YOLOModelHandler.predict_image_sahi(
            'models/pt/best_yolov8x.pt', 'path/to/image.jpg',
        )
        mock_auto_detection_model.assert_called_with(
            model_type='yolov8',
            model_path='models/pt/best_yolov8x.pt',
            confidence_threshold=0.3,
        )
        mock_sahi_predict.assert_called()
        self.assertIsNotNone(result)

    def test_predict_image_sahi_without_model_path(self) -> None:
        """
        Test that calling predict_image_sahi
        without a valid model path raises RuntimeError.
        """
        with self.assertRaises(RuntimeError) as context:
            YOLOModelHandler.predict_image_sahi('', 'path/to/image.jpg')

        self.assertEqual(
            str(context.exception),
            'The model path is not provided.',
        )

    @patch('argparse.ArgumentParser.parse_args')
    @patch('examples.YOLOv8_train.train.YOLOModelHandler')
    def test_main(
        self,
        mock_handler_class: unittest.mock.MagicMock,
        mock_parse_args: unittest.mock.MagicMock,
    ) -> None:
        """
        Test the main function.
        """
        mock_handler = MagicMock()
        mock_handler_class.return_value = mock_handler
        mock_parse_args.return_value = argparse.Namespace(
            data_config='dataset/data.yaml',
            epochs=100,
            model_name='models/pt/best_yolov8x.pt',
            export_format='onnx',
            onnx_path=None,
            pt_path='model.pt',
            sahi_image_path='../../assets/IMG_1091.PNG',
            batch_size=16,
            optimizer='auto',
            cross_validate=False,
            n_splits=5,
        )

        main()

        mock_handler.train_model.assert_called()
        mock_handler.validate_model.assert_called()
        mock_handler.export_model.assert_called_with(export_format='onnx')
        mock_handler.save_model.assert_called_with('model.pt')

    @patch('argparse.ArgumentParser.parse_args')
    @patch('examples.YOLOv8_train.train.YOLOModelHandler')
    def test_main_with_cross_validate(
        self,
        mock_handler_class: unittest.mock.MagicMock,
        mock_parse_args: unittest.mock.MagicMock,
    ) -> None:
        """
        Test the main function when cross-validation is enabled.
        """
        mock_handler = MagicMock()
        mock_handler_class.return_value = mock_handler
        mock_parse_args.return_value = argparse.Namespace(
            data_config='dataset/data.yaml',
            epochs=100,
            model_name='models/pt/best_yolov8x.pt',
            export_format='onnx',
            onnx_path=None,
            pt_path='model.pt',
            sahi_image_path='../../assets/IMG_1091.PNG',
            batch_size=16,
            optimizer='auto',
            cross_validate=True,  # Enable cross-validation
            n_splits=5,
        )

        main()

        # Check that cross-validation was called
        mock_handler.cross_validate_model.assert_called_with(
            data_config='dataset/data.yaml',
            epochs=100,
            optimizer='auto',
            n_splits=5,
        )

        # Since cross-validation was used, we do not expect train_model
        # or validate_model to be called separately
        mock_handler.train_model.assert_not_called()
        mock_handler.validate_model.assert_not_called()

        # Export and save should still be called
        mock_handler.export_model.assert_called_with(export_format='onnx')
        mock_handler.save_model.assert_called_with('model.pt')

    @patch('examples.YOLOv8_train.train.YOLOModelHandler.train_model')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_exception_handling(
        self,
        mock_parse_args: unittest.mock.MagicMock,
        mock_train_model: unittest.mock.MagicMock,
    ) -> None:
        """
        Test the main function's exception handling.
        """
        mock_parse_args.return_value = argparse.Namespace(
            data_config='dataset/data.yaml',
            epochs=100,
            model_name='models/pt/best_yolov8x.pt',
            export_format='onnx',
            onnx_path=None,
            pt_path='model.pt',
            sahi_image_path='../../assets/IMG_1091.PNG',
            batch_size=16,
            optimizer='auto',
            cross_validate=False,
            n_splits=5,
        )

        # Simulate an exception during training
        mock_train_model.side_effect = Exception('Mocked training error')

        with (
            patch('builtins.print') as mock_print,
            self.assertRaises(SystemExit),
        ):
            main()
            mock_print.assert_any_call('Error occurred: Mocked training error')


if __name__ == '__main__':
    unittest.main()
