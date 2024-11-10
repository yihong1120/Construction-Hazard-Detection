from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import create_autospec
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker

from examples.YOLO_server.models import Base
from examples.YOLO_server.models import DetectionModelManager
from examples.YOLO_server.models import get_db
from examples.YOLO_server.models import ModelFileChangeHandler
from examples.YOLO_server.models import User

# Define the in-memory database URI for testing
DATABASE_URL = 'sqlite:///:memory:'

# Configure the testing database and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


class TestUserModel(unittest.TestCase):
    """Test cases for the User model"""

    def setUp(self):
        """Set up a test database session"""
        self.session = SessionLocal()

    def tearDown(self):
        """Clean up the database after each test"""
        self.session.close()

    def test_set_password(self):
        """Test password hashing in User model"""
        user = User(username='testuser')
        user.set_password('secure_password')
        self.assertTrue(user.password_hash != 'secure_password')
        self.assertTrue(user.check_password('secure_password'))

    def test_check_password(self):
        """Test password verification in User model"""
        user = User(username='testuser')
        user.set_password('secure_password')
        self.assertTrue(user.check_password('secure_password'))
        self.assertFalse(user.check_password('wrong_password'))

    def test_to_dict(self):
        """Test the to_dict method of the User model"""
        user = User(username='testuser', role='admin', is_active=True)
        user.set_password('secure_password')
        self.session.add(user)
        self.session.commit()

        user_dict = user.to_dict()
        self.assertEqual(user_dict['username'], 'testuser')
        self.assertEqual(user_dict['role'], 'admin')
        self.assertTrue(user_dict['is_active'])
        self.assertIn('created_at', user_dict)
        self.assertIn('updated_at', user_dict)


class TestDetectionModelManager(unittest.TestCase):
    """Test cases for DetectionModelManager"""

    def setUp(self):
        """Set up the model manager for testing"""
        self.model_manager = DetectionModelManager()
        # Mock the AutoDetectionModel to avoid loading actual models
        self.patcher = patch('examples.YOLO_server.models.AutoDetectionModel')
        self.mock_model = self.patcher.start()

    def tearDown(self):
        """Stop the patching of the model manager"""
        self.patcher.stop()

    def test_load_single_model(self):
        """Test loading a single model"""
        self.mock_model.from_pretrained.assert_called_once_with(
            'yolov8', model_path=str(Path('models/pt/best_yolo11n.pt')),
            device='cuda:0',
        )

    def test_get_model(self):
        """Test retrieving a model by its key"""
        self.model_manager.models['yolo11n'] = self.mock_model
        model = self.model_manager.get_model('yolo11n')
        self.assertEqual(model, self.mock_model)
        model_none = self.model_manager.get_model('nonexistent')
        self.assertIsNone(model_none)

    @patch.object(DetectionModelManager, 'observer')
    def test_cleanup_on_delete(self, mock_observer):
        """Test that the observer is stopped and joined on deletion"""
        manager = DetectionModelManager()
        del manager
        mock_observer.stop.assert_called_once()
        mock_observer.join.assert_called_once()


class TestModelFileChangeHandler(unittest.TestCase):
    """Test cases for ModelFileChangeHandler"""

    def setUp(self):
        """Set up the model manager and file change handler"""
        # Mock DetectionModelManager with required attributes and methods
        self.model_manager = create_autospec(DetectionModelManager)
        self.model_manager.model_names = ['yolo11n']
        self.model_manager.models = {}
        self.model_manager.load_single_model = Mock(return_value='dummy_model')

        # Initialise the handler with the mocked model manager
        self.handler = ModelFileChangeHandler(self.model_manager)

    def test_on_modified_with_directory(self):
        """Test handling of a directory modification event (should ignore)"""
        event = MagicMock()
        event.is_directory = True
        self.handler.on_modified(event)
        self.model_manager.load_single_model.assert_not_called()

    def test_on_modified_with_model_file(self):
        """Test handling of a model file modification event"""
        event = MagicMock()
        event.is_directory = False
        event.src_path = 'models/pt/best_yolo11n.pt'

        # Trigger the file modification event
        self.handler.on_modified(event)

        # Check if the model was reloaded
        self.model_manager.load_single_model.assert_called_once_with('yolo11n')
        self.assertEqual(self.model_manager.models['yolo11n'], 'dummy_model')


class TestDatabase(unittest.TestCase):
    """Test cases for database connection and session generator"""

    def test_get_db(self):
        """Test the database session generator"""
        db_generator = get_db()
        session = next(db_generator)
        self.assertIsInstance(session, Session)
        session.close()


# Run the tests with pytest if this script is called directly
if __name__ == '__main__':
    unittest.main()
