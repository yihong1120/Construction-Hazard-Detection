from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import mock_open
from unittest.mock import patch

from examples.YOLO_server_api.models import DetectionModelManager
from examples.YOLO_server_api.models import User


class TestUserModel(unittest.TestCase):
    user: User
    password: str

    def setUp(self) -> None:
        """
        Set up a User instance for testing.
        """
        self.user = User(username='testuser')
        self.password = 'securepassword'
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
        'examples.YOLO_server_api.models.'
        'AutoDetectionModel.from_pretrained',
    )
    @patch('builtins.open', new_callable=mock_open, read_data='dummy data')
    def setUp(
        self,
        mock_open_file: MagicMock,
        mock_from_pretrained: MagicMock,
    ) -> None:
        """
        Set up the DetectionModelManager
        and mock model loading and file access.

        Args:
            mock_open_file (MagicMock): Mocked file open function.
            mock_from_pretrained (MagicMock): Mocked function to
                load pretrained models.
        """
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        # Initialize the model manager with mocks
        self.model_manager = DetectionModelManager()

        # Ensure that models were loaded correctly
        self.assertEqual(
            len(self.model_manager.models),
            len(self.model_manager.model_names),
        )
        for name in self.model_manager.model_names:
            self.assertIn(name, self.model_manager.models)

    @patch(
        'examples.YOLO_server_api.models.'
        'AutoDetectionModel.from_pretrained',
    )
    def test_load_single_model(
        self,
        mock_from_pretrained: MagicMock,
    ) -> None:
        """
        Test loading a single model with mocked file access.

        Args:
            mock_from_pretrained (MagicMock): Mocked function to
                load pretrained models.
        """
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        model_name: str = 'yolov8x'
        model = self.model_manager.load_single_model(model_name)
        mock_from_pretrained.assert_called_once_with(
            'yolov8',
            model_path=str(Path('models/pt/') / f"best_{model_name}.pt"),
            device='cuda:0',
        )
        self.assertEqual(model, mock_model)

    @patch('examples.YOLO_server_api.models.Path.stat')
    def test_get_last_modified_time(
        self,
        mock_stat: MagicMock,
    ) -> None:
        """
        Test retrieving the last modified time of
        a model file with mocked stat.

        Args:
            mock_stat (MagicMock): Mocked function to
                retrieve file statistics.
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
        'examples.YOLO_server_api.models.DetectionModelManager.'
        'get_last_modified_times',
    )
    @patch(
        'examples.YOLO_server_api.models.DetectionModelManager.'
        'load_single_model',
    )
    @patch('time.sleep', return_value=None)
    def test_reload_models_every_hour(
        self,
        mock_sleep: MagicMock,
        mock_load_single_model: MagicMock,
        mock_get_last_modified_times: MagicMock,
    ) -> None:
        """
        Test the reloading of models every hour
        with mocked file access and timing.

        Args:
            mock_sleep (MagicMock): Mocked sleep function to
                simulate time delays.
            mock_load_single_model (MagicMock): Mocked function to
                load a single model.
            mock_get_last_modified_times (MagicMock): Mocked function to
                retrieve last modified times.
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
                if current_times[name] != (
                    self.model_manager.last_modified_times.get(name)
                ):
                    self.model_manager.models[name] = (
                        self.model_manager.load_single_model(name)
                    )
                    self.model_manager.last_modified_times[name] = (
                        current_times[name]
                    )

            # Stop after one loop for testing
            raise StopIteration

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
        'examples.YOLO_server_api.models.DetectionModelManager.'
        'get_last_modified_time',
    )
    def test_get_last_modified_times(
        self,
        mock_get_last_modified_time: MagicMock,
    ) -> None:
        """
        Test retrieving last modified times
        for all models with mocked file access.

        Args:
            mock_get_last_modified_time (MagicMock): Mocked function to
                retrieve last modified time.
        """
        mock_time: float = 1650000000.0
        mock_get_last_modified_time.return_value = mock_time

        last_modified_times: dict[
            str,
            float,
        ] = self.model_manager.get_last_modified_times()
        self.assertEqual(
            len(last_modified_times),
            len(self.model_manager.model_names),
        )
        for name in self.model_manager.model_names:
            self.assertEqual(last_modified_times[name], mock_time)


if __name__ == '__main__':
    unittest.main()
