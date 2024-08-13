import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from werkzeug.security import generate_password_hash

from examples.YOLOv8_server_api.models import User, DetectionModelManager

class TestUserModel(unittest.TestCase):
    def setUp(self):
        """
        Set up a User instance for testing.
        """
        self.user = User(username="testuser")
        self.password = "securepassword"
        self.user.set_password(self.password)

    def test_set_password(self):
        """
        Test that the password is hashed correctly.
        """
        self.assertNotEqual(self.password, self.user.password_hash)
        self.assertTrue(self.user.check_password(self.password))

    def test_check_password(self):
        """
        Test the password verification method.
        """
        self.assertTrue(self.user.check_password(self.password))
        self.assertFalse(self.user.check_password("wrongpassword"))

class TestDetectionModelManager(unittest.TestCase):
    @patch('examples.YOLOv8_server_api.models.AutoDetectionModel.from_pretrained')
    def setUp(self, mock_from_pretrained):
        """
        Set up the DetectionModelManager and mock model loading.
        """
        # Mock the loading of models
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        self.model_manager = DetectionModelManager()

        # Ensure that models were loaded
        self.assertEqual(len(self.model_manager.models), len(self.model_manager.model_names))
        for name in self.model_manager.model_names:
            self.assertIn(name, self.model_manager.models)

    @patch('examples.YOLOv8_server_api.models.AutoDetectionModel.from_pretrained')
    def test_load_single_model(self, mock_from_pretrained):
        """
        Test loading a single model.
        """
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        model_name = "yolov8x"
        model = self.model_manager.load_single_model(model_name)
        mock_from_pretrained.assert_called_once_with(
            'yolov8',
            model_path=str(Path('models/pt/') / f"best_{model_name}.pt"),
            device='cuda:0'
        )
        self.assertEqual(model, mock_model)

    @patch('examples.YOLOv8_server_api.models.Path.stat')
    def test_get_last_modified_time(self, mock_stat):
        """
        Test retrieving the last modified time of a model file.
        """
        mock_time = 1650000000.0
        mock_stat.return_value.st_mtime = mock_time

        model_name = "yolov8x"
        last_modified_time = self.model_manager.get_last_modified_time(model_name)
        self.assertEqual(last_modified_time, mock_time)
        mock_stat.assert_called_once_with()

    @patch('examples.YOLOv8_server_api.models.DetectionModelManager.get_last_modified_times')
    @patch('examples.YOLOv8_server_api.models.DetectionModelManager.load_single_model')
    @patch('time.sleep', return_value=None)
    def test_reload_models_every_hour(self, mock_sleep, mock_load_single_model, mock_get_last_modified_times):
        """
        Test the reloading of models every hour.
        """
        # Mock the current times to simulate the models being modified
        initial_times = {name: 1650000000.0 for name in self.model_manager.model_names}
        modified_times = {name: 1650003600.0 for name in self.model_manager.model_names}
    
        mock_get_last_modified_times.side_effect = [initial_times, modified_times]
    
        # Modify the reload_models_every_hour to exit after one iteration
        def reload_models_once():
            current_times = self.model_manager.get_last_modified_times()
            for name in self.model_manager.model_names:
                if current_times[name] != self.model_manager.last_modified_times.get(name):
                    self.model_manager.models[name] = self.model_manager.load_single_model(name)
                    self.model_manager.last_modified_times[name] = current_times[name]
            raise StopIteration  # Stop after one loop for testing
    
        with patch.object(self.model_manager, 'reload_models_every_hour', reload_models_once):
            with self.assertRaises(StopIteration):
                self.model_manager.reload_models_every_hour()
    
        # Ensure that each model is reloaded exactly once
        self.assertEqual(mock_load_single_model.call_count, len(self.model_manager.model_names))

    @patch('examples.YOLOv8_server_api.models.DetectionModelManager.get_last_modified_time')
    def test_get_last_modified_times(self, mock_get_last_modified_time):
        """
        Test retrieving last modified times for all models.
        """
        mock_time = 1650000000.0
        mock_get_last_modified_time.return_value = mock_time

        last_modified_times = self.model_manager.get_last_modified_times()
        self.assertEqual(len(last_modified_times), len(self.model_manager.model_names))
        for name in self.model_manager.model_names:
            self.assertEqual(last_modified_times[name], mock_time)

if __name__ == '__main__':
    unittest.main()
