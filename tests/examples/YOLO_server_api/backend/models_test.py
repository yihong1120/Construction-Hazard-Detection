"""Comprehensive test suite for YOLO model management functionality.

This module provides extensive test coverage for the DetectionModelManager and
ModelFileChangeHandler classes, ensuring robust model loading, hot-reloading,
and lifecycle management functionality.
"""
from __future__ import annotations

import asyncio
import importlib
import unittest
from pathlib import Path
from unittest.mock import create_autospec
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

import examples.YOLO_server_api.backend.models
from examples.auth.database import get_db
from examples.auth.models import Base

DetectionModelManager = (
    examples.YOLO_server_api.backend.models.DetectionModelManager
)
ModelFileChangeHandler = (
    examples.YOLO_server_api.backend.models.ModelFileChangeHandler
)

# Define the in-memory database URI for testing
DATABASE_URL = 'sqlite:///:memory:'

# Configure the testing database and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


class TestDetectionModelManager(unittest.TestCase):
    """Comprehensive test cases for the DetectionModelManager class.

    Tests model loading, retrieval, hot-reloading functionality, and lifecycle
    management across both TensorRT and PyTorch configurations.
    """

    def setUp(self) -> None:
        """Set up the model manager for testing.

        Configures mock dependencies and initialises a DetectionModelManager
        instance with SAHI configuration for consistent testing.
        """
        # Add mock AutoDetectionModel to the models module if it doesn't exist
        if not hasattr(
            examples.YOLO_server_api.backend.models,
            'AutoDetectionModel',
        ):
            mock_auto_detection = Mock()
            mock_auto_detection.from_pretrained = Mock(return_value=Mock())
            examples.YOLO_server_api.backend.models.AutoDetectionModel = (
                mock_auto_detection
            )

        # Add mock YOLO to the models module if it doesn't exist
        if not hasattr(examples.YOLO_server_api.backend.models, 'YOLO'):
            mock_yolo = Mock()
            mock_yolo.return_value = Mock()
            examples.YOLO_server_api.backend.models.YOLO = mock_yolo

        # Mock all the necessary components to prevent actual model loading
        self.mock_observer = patch(
            'examples.YOLO_server_api.backend.models.Observer',
        ).start()
        self.mock_thread = patch(
            'examples.YOLO_server_api.backend.models.threading.Thread',
        ).start()

        # Ensure USE_TENSORRT is False for default testing
        with patch(
            'examples.YOLO_server_api.backend.models.USE_TENSORRT',
            False,
        ):
            self.model_manager = DetectionModelManager()

    def tearDown(self) -> None:
        """
        Stop all patches.
        """
        patch.stopall()

    def test_load_single_model(self) -> None:
        """
        Test loading a single model into the manager.
        """
        # Get the mock from the module
        mock_auto_detection = (
            examples.YOLO_server_api.backend.models.AutoDetectionModel
        )
        # The call count might be higher due to multiple test runs,
        # so we just check that it was called for yolo11n
        mock_auto_detection.from_pretrained.assert_any_call(
            'yolo11', model_path=str(Path('models/pt/best_yolo11n.pt')),
            device='cuda:0',
        )

    def test_get_model(self) -> None:
        """
        Test retrieving a model by its key from the manager.
        """
        # Mock a loaded model for testing retrieval
        mock_model = Mock()
        self.model_manager.models['yolo11n'] = mock_model
        model = self.model_manager.get_model('yolo11n')

        # Validate that the retrieved model matches the mock model
        self.assertEqual(model, mock_model)

        # Test retrieval of non-existent model
        model_none = self.model_manager.get_model('nonexistent')
        self.assertIsNone(model_none)

    def test_cleanup_on_delete(self) -> None:
        """
        Test that the observer is stopped and joined upon deletion.
        """
        # Test cleanup method
        self.model_manager.__del__()
        self.mock_observer.return_value.stop.assert_called_once()
        self.mock_observer.return_value.join.assert_called_once()

    def test_tensorrt_path(self) -> None:
        """
        Test the TensorRT model loading path.
        """
        with patch(
            'examples.YOLO_server_api.backend.models.USE_TENSORRT',
            True,
        ):
            # Create a fresh mock for this test
            with patch(
                'examples.YOLO_server_api.backend.models.YOLO',
            ) as mock_yolo_fresh:
                mock_yolo_fresh.return_value = Mock()

                # Create a manager instance and test load_single_model directly
                manager = DetectionModelManager.__new__(DetectionModelManager)
                manager.model_names = ['test_model']
                manager.base_model_path = Path('models/int8_engine/')
                manager.extension = '.engine'

                result = manager.load_single_model('test_model')
                expected_path = str(
                    Path('models/int8_engine/best_test_model.engine'),
                )
                mock_yolo_fresh.assert_called_with(expected_path)
                self.assertEqual(result, mock_yolo_fresh.return_value)

    def test_tensorrt_initialization(self) -> None:
        """
        Test DetectionModelManager initialization with TensorRT enabled.
        """
        # Mock all the necessary components
        with (
            patch(
                'examples.YOLO_server_api.backend.models.Observer',
            ),
            patch(
                'examples.YOLO_server_api.backend.models.threading.Thread',
            ),
            patch(
                'examples.YOLO_server_api.backend.models.USE_TENSORRT', True,
            ),
            patch('examples.YOLO_server_api.backend.models.YOLO') as mock_yolo,
        ):

            mock_yolo.return_value = Mock()

            # This should trigger the TensorRT path in __init__
            manager = DetectionModelManager()

            # Verify TensorRT configuration
            self.assertEqual(
                manager.base_model_path,
                Path('models/int8_engine/'),
            )
            self.assertEqual(manager.extension, '.engine')

            # Verify YOLO was called for all models
            self.assertEqual(mock_yolo.call_count, 5)

    def test_sahi_initialization(self) -> None:
        """
        Test DetectionModelManager initialisation with SAHI
        (USE_TENSORRT=False).
        """
        # This is already covered by the setUp method, but let's be explicit
        self.assertEqual(
            self.model_manager.base_model_path,
            Path('models/pt/'),
        )
        self.assertEqual(self.model_manager.extension, '.pt')

    def test_sahi_import_path(self) -> None:
        """
        Test that the correct import path is used for SAHI.
        """
        with patch(
            'examples.YOLO_server_api.backend.config.USE_TENSORRT',
            False,
        ):
            # Force reimport to trigger the sahi import line
            importlib.reload(examples.YOLO_server_api.backend.models)

            # Verify that AutoDetectionModel is available in the module
            self.assertTrue(
                hasattr(
                    examples.YOLO_server_api.backend.models,
                    'AutoDetectionModel',
                ),
            )

    def test_get_available_models(self) -> None:
        """
        Test getting available models with mixed loaded/failed models.
        """
        # Set up a mix of loaded and None models
        mock_model = Mock()
        self.model_manager.models = {
            'yolo11x': mock_model,
            'yolo11l': None,  # Simulate failed model
            'yolo11m': mock_model,
            'yolo11s': None,  # Simulate failed model
            'yolo11n': mock_model,
        }

        available = self.model_manager.get_available_models()
        expected = ['yolo11x', 'yolo11m', 'yolo11n']
        self.assertEqual(sorted(available), sorted(expected))

    def test_is_model_loaded(self) -> None:
        """
        Test checking if specific models are loaded.
        """
        # Set up test models
        mock_model = Mock()
        self.model_manager.models = {
            'yolo11x': mock_model,
            'yolo11l': None,
        }

        # Test loaded model
        self.assertTrue(self.model_manager.is_model_loaded('yolo11x'))

        # Test failed/None model
        self.assertFalse(self.model_manager.is_model_loaded('yolo11l'))

        # Test non-existent model
        self.assertFalse(self.model_manager.is_model_loaded('nonexistent'))

    def test_reload_model_success(self) -> None:
        """
        Test successful manual model reloading.
        """
        # Mock the load_single_model method
        mock_model = Mock()
        self.model_manager.load_single_model = Mock(return_value=mock_model)

        # Test successful reload
        result = self.model_manager.reload_model('yolo11x')
        self.assertTrue(result)
        self.model_manager.load_single_model.assert_called_once_with('yolo11x')
        self.assertEqual(self.model_manager.models['yolo11x'], mock_model)

    def test_reload_model_invalid_name(self) -> None:
        """
        Test reloading with invalid model name.
        """
        # Test with invalid model name
        result = self.model_manager.reload_model('invalid_model')
        self.assertFalse(result)

    def test_reload_model_failure(self) -> None:
        """
        Test model reloading failure handling.
        """
        # Mock load_single_model to raise an exception
        self.model_manager.load_single_model = Mock(
            side_effect=Exception('Model loading failed'),
        )

        # Test reload failure
        result = self.model_manager.reload_model('yolo11x')
        self.assertFalse(result)

    def test_destructor_without_observer(self) -> None:
        """
        Test destructor when observer doesn't exist.
        """
        # Create a manager without observer
        manager = DetectionModelManager.__new__(DetectionModelManager)

        # Call destructor - should not raise an exception
        manager.__del__()

        # No assertion needed, just ensuring no exception is raised

    def test_destructor_with_none_observer(self) -> None:
        """
        Test destructor when observer is None.
        """
        # Create a manager and set observer to None
        manager = DetectionModelManager.__new__(DetectionModelManager)
        manager.observer = None

        # Call destructor - should not raise an exception
        manager.__del__()

        # No assertion needed, just ensuring no exception is raised


class TestModelFileChangeHandler(unittest.TestCase):
    """Comprehensive test cases for the ModelFileChangeHandler class.

    Tests file system event handling for automatic model hot-reloading,
    including edge cases and error conditions.
    """

    def setUp(self) -> None:
        """
        Set up the model manager and file change handler.
        """
        # Mock DetectionModelManager with necessary attributes and methods
        self.model_manager = create_autospec(DetectionModelManager)
        self.model_manager.model_names = ['yolo11n']
        self.model_manager.models = {}
        self.model_manager.extension = '.pt'  # Add missing extension attribute
        self.model_manager.load_single_model = Mock(return_value='dummy_model')

        # Initialise the handler with the mocked model manager
        self.handler = ModelFileChangeHandler(self.model_manager)

    def test_on_modified_with_directory(self) -> None:
        """
        Test handling of a directory modification event (should ignore).
        """
        event = MagicMock()
        event.is_directory = True
        # Ensure directory modifications do not trigger model loading
        self.handler.on_modified(event)
        self.model_manager.load_single_model.assert_not_called()

    def test_on_modified_with_model_file(self) -> None:
        """
        Test handling of a model file modification event.
        """
        event = MagicMock()
        event.is_directory = False
        event.src_path = 'models/pt/best_yolo11n.pt'

        # Trigger the file modification event
        self.handler.on_modified(event)

        # Verify the model was reloaded
        self.model_manager.load_single_model.assert_called_once_with('yolo11n')
        self.assertEqual(self.model_manager.models['yolo11n'], 'dummy_model')

    def test_on_modified_with_wrong_extension(self) -> None:
        """
        Test handling of file with wrong extension (should ignore).
        """
        event = MagicMock()
        event.is_directory = False
        event.src_path = 'models/pt/best_yolo11n.txt'  # Wrong extension

        # Trigger the file modification event
        self.handler.on_modified(event)

        # Verify no model loading was triggered
        self.model_manager.load_single_model.assert_not_called()

    def test_on_modified_with_unknown_model(self) -> None:
        """
        Test handling of unknown model file (should ignore).
        """
        event = MagicMock()
        event.is_directory = False
        event.src_path = 'models/pt/best_unknown_model.pt'

        # Trigger the file modification event
        self.handler.on_modified(event)

        # Verify no model loading was triggered since 'unknown_model'
        # is not in model_names
        self.model_manager.load_single_model.assert_not_called()


class TestDatabase(unittest.TestCase):
    """Test cases for database connectivity and session management.

    Ensures proper database session creation and resource management
    for the authentication system.
    """

    @staticmethod
    async def async_get_db() -> AsyncSession:
        """
        Yield an asynchronous database session for testing.
        """
        async for session in get_db():
            return session
        raise RuntimeError('Failed to create AsyncSession.')

    def test_get_db(self) -> None:
        """
        Test the database session generator.
        """
        session = asyncio.run(self.async_get_db())

        # Confirm the session is an AsyncSession instance
        self.assertIsInstance(session, AsyncSession)

        # Close session to avoid resource leak
        asyncio.run(session.close())


if __name__ == '__main__':
    unittest.main()

'''
pytest \
    --cov=examples.YOLO_server_api.backend.models \
    --cov-report=term-missing \
    tests/examples/YOLO_server_api/backend/models_test.py
'''
