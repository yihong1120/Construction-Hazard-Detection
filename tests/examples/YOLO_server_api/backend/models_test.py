from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from unittest.mock import create_autospec
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from examples.auth.database import get_db
from examples.auth.models import Base
from examples.YOLO_server_api.backend.models import DetectionModelManager
from examples.YOLO_server_api.backend.models import ModelFileChangeHandler

# Define the in-memory database URI for testing
DATABASE_URL = 'sqlite:///:memory:'

# Configure the testing database and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


class TestDetectionModelManager(unittest.TestCase):
    """
    Test cases for the DetectionModelManager.
    """

    def setUp(self) -> None:
        """
        Set up the model manager for testing.
        """
        # Patch AutoDetectionModel to avoid actual model loading
        self.patcher = patch(
            'examples.YOLO_server_api.backend.models.AutoDetectionModel',
        )
        self.mock_model = self.patcher.start()
        self.model_manager = DetectionModelManager()

    def tearDown(self) -> None:
        """
        Stop patching the model manager.
        """
        self.patcher.stop()

    def test_load_single_model(self) -> None:
        """
        Test loading a single model into the manager.
        """
        # Confirm model is loaded with correct parameters
        self.assertEqual(self.mock_model.from_pretrained.call_count, 5)
        self.mock_model.from_pretrained.assert_any_call(
            'yolo11', model_path=str(Path('models/pt/best_yolo11n.pt')),
            device='cuda:0',
        )

    def test_get_model(self) -> None:
        """
        Test retrieving a model by its key from the manager.
        """
        # Mock a loaded model for testing retrieval
        self.model_manager.models['yolo11n'] = self.mock_model
        model = self.model_manager.get_model('yolo11n')

        # Validate that the retrieved model matches the mock model
        self.assertEqual(model, self.mock_model)

        # Test retrieval of non-existent model
        model_none = self.model_manager.get_model('nonexistent')
        self.assertIsNone(model_none)

    @patch.object(DetectionModelManager, 'observer', create=True)
    def test_cleanup_on_delete(self, mock_observer: Mock) -> None:
        """
        Test that the observer is stopped and joined upon deletion.
        """
        manager = DetectionModelManager()
        manager.observer = mock_observer

        # Explicitly call __del__ to test cleanup
        manager.__del__()
        mock_observer.stop.assert_called_once()
        mock_observer.join.assert_called_once()


class TestModelFileChangeHandler(unittest.TestCase):
    """
    Test cases for the ModelFileChangeHandler.
    """

    def setUp(self) -> None:
        """
        Set up the model manager and file change handler.
        """
        # Mock DetectionModelManager with necessary attributes and methods
        self.model_manager = create_autospec(DetectionModelManager)
        self.model_manager.model_names = ['yolo11n']
        self.model_manager.models = {}
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


class TestDatabase(unittest.TestCase):
    """
    Test cases for database connection and session generator.
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
    --cov=examples.YOLO_server_api.backend \
    --cov-report=term-missing \
    tests/examples/YOLO_server_api/backend/models_test.py
'''
