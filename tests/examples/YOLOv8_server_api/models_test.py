from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

from examples.YOLOv8_server_api.models import DetectionModelManager
from examples.YOLOv8_server_api.models import User


class TestUserModel(unittest.TestCase):
    user: User
    password: str

    def setUp(self) -> None:
        """
        Set up a User instance for testing.
        """
        self.user: User = User(username='testuser')
        self.password: str = 'securepassword'
        self.user.set_password(self.password)

    def test_set_password(self) -> None:
        """
        Test that the password is hashed correctly.
        """
        self.assertNotEqual(self.password, self.user.password_hash)
        self.assertTrue(self.user.check_password(self.password))

    def test_check_password(self) -> None:
        """
        Test the password verification method.
        """
        self.assertTrue(self.user.check_password(self.password))
        self.assertFalse(self.user.check_password('wrongpassword'))


class TestDetectionModelManager(unittest.TestCase):
    model_manager: DetectionModelManager

    @patch(
        'examples.YOLOv8_server_api.models.'
        'AutoDetectionModel.from_pretrained',
    )
    def setUp(self, mock_from_pretrained: MagicMock) -> None:
        """
        Set up the DetectionModelManager and mock model loading.
        """
        mock_model: MagicMock = MagicMock()
        mock_from_pretrained.return_value = mock_model

        self.model_manager: DetectionModelManager = DetectionModelManager()

        # Ensure that models were loaded
        self.assertEqual(
            len(self.model_manager.models),
            len(self.model_manager.model_names),
        )
        for name in self.model_manager.model_names:
            self.assertIn(name, self.model_manager.models)

    @patch(
        'examples.YOLOv8_server_api.models.'
        'AutoDetectionModel.from_pretrained',
    )
    def test_load_single_model(
        self, mock_from_pretrained: MagicMock,
    ) -> None:
        """
        Test loading a single model.
        """
        mock_model: MagicMock = MagicMock()
        mock_from_pretrained.return_value = mock_model

        model_name: str = 'yolov8x'
        model: MagicMock = self.model_manager.load_single_model(model_name)
        mock_from_pretrained.assert_called_once_with(
            'yolov8',
            model_path=str(Path('models/pt/') / f"best_{model_name}.pt"),
            device='cuda:0',
        )
        self.assertEqual(model, mock_model)

    @patch('examples.YOLOv8_server_api.models.Path.stat')
    def test_get_last_modified_time(self, mock_stat: MagicMock) -> None:
        """
        Test retrieving the last modified time of a model file.
        """
        mock_time: float = 1650000000.0
        mock_stat.return_value.st_mtime = mock_time

        model_name: str = 'yolov8x'
        last_modified_time: float = self.model_manager.get_last_modified_time(
            model_name,
        )
        self.assertEqual(last_modified_time, mock_time)
        mock_stat.assert_called_once_with()

    @patch(
        'examples.YOLOv8_server_api.models.'
        'DetectionModelManager.get_last_modified_times',
    )
    @patch(
        'examples.YOLOv8_server_api.models.'
        'DetectionModelManager.load_single_model',
    )
    @patch('time.sleep', return_value=None)
    def test_reload_models_every_hour(
        self,
        mock_sleep: MagicMock,
        mock_load_single_model: MagicMock,
        mock_get_last_modified_times: MagicMock,
    ) -> None:
        """
        Test the reloading of models every hour.
        """
        initial_times: dict[str, float] = {
            name: 1650000000.0 for name in self.model_manager.model_names
        }
        modified_times: dict[str, float] = {
            name: 1650003600.0 for name in self.model_manager.model_names
        }

        mock_get_last_modified_times.side_effect = [
            initial_times, modified_times,
        ]

        def reload_models_once() -> None:
            current_times: dict[str, float] = (
                self.model_manager.get_last_modified_times()
            )
            for name in self.model_manager.model_names:
                if (
                    current_times[name] !=
                    self.model_manager.last_modified_times.get(name)
                ):
                    self.model_manager.models[name] = (
                        self.model_manager.load_single_model(name)
                    )
                    self.model_manager.last_modified_times[name] = (
                        current_times[name]
                    )
            raise StopIteration  # Stop after one loop for testing

        with patch.object(
            self.model_manager,
            'reload_models_every_hour',
            reload_models_once,
        ):
            with self.assertRaises(StopIteration):
                self.model_manager.reload_models_every_hour()

        self.assertEqual(
            mock_load_single_model.call_count,
            len(self.model_manager.model_names),
        )

    @patch(
        'examples.YOLOv8_server_api.models.'
        'DetectionModelManager.get_last_modified_time',
    )
    def test_get_last_modified_times(
        self, mock_get_last_modified_time: MagicMock,
    ) -> None:
        """
        Test retrieving last modified times for all models.
        """
        mock_time: float = 1650000000.0
        mock_get_last_modified_time.return_value = mock_time

        last_modified_times: dict[str, float] = (
            self.model_manager.get_last_modified_times()
        )
        self.assertEqual(
            len(last_modified_times),
            len(self.model_manager.model_names),
        )
        for name in self.model_manager.model_names:
            self.assertEqual(last_modified_times[name], mock_time)


if __name__ == '__main__':
    unittest.main()
