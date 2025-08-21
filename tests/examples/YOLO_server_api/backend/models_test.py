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
            examples.YOLO_server_api.backend.models.YOLO = (
                mock_yolo
            )

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
        # Test both SAHI and standard modes

        # Test SAHI mode first
        with (
            patch.object(Path, 'exists', return_value=True),
            patch('builtins.__import__') as mock_import,
        ):
            # Setup the mock for SAHI import
            mock_sahi_module = Mock()
            mock_auto_detection_class = Mock()
            mock_model_instance = Mock()
            mock_auto_detection_class.from_pretrained.return_value = (
                mock_model_instance
            )
            mock_sahi_module.AutoDetectionModel = mock_auto_detection_class

            def side_effect(name, *args, **kwargs):
                if 'sahi.predict' in name:
                    return mock_sahi_module
                else:
                    # Call the original import for other modules
                    return __import__(name, *args, **kwargs)

            mock_import.side_effect = side_effect

            # Create a manager instance and test SAHI mode explicitly
            manager = DetectionModelManager.__new__(DetectionModelManager)
            manager.base_model_path = Path('models/pt/')
            manager.extension = '.pt'

            # Test SAHI mode by simulating it
            with patch(
                'examples.YOLO_server_api.backend.models.USE_SAHI', True,
            ):
                result = manager.load_single_model('yolo11n')

                # Verify from_pretrained was called correctly
                mock_auto_detection_class.from_pretrained.assert_called_with(
                    'yolo11',
                    model_path=str(Path('models/pt/best_yolo11n.pt')),
                    device='cuda:0',
                )
                self.assertEqual(result, mock_model_instance)

        # Test standard PyTorch mode
        with (
            patch.object(Path, 'exists', return_value=True),
            patch('examples.YOLO_server_api.backend.models.YOLO') as mock_yolo,
        ):
            mock_yolo_instance = Mock()
            mock_yolo.return_value = mock_yolo_instance

            manager = DetectionModelManager.__new__(DetectionModelManager)
            manager.base_model_path = Path('models/pt/')
            manager.extension = '.pt'

            # Test standard mode
            with (
                patch(
                    'examples.YOLO_server_api.backend.models.USE_SAHI', False,
                ),
                patch(
                    'examples.YOLO_server_api.backend.models.USE_TENSORRT',
                    False,
                ),
            ):
                result = manager.load_single_model('yolo11n')

                # Verify YOLO was called correctly
                mock_yolo.assert_called_with(
                    str(Path('models/pt/best_yolo11n.pt')),
                )
                self.assertEqual(result, mock_yolo_instance)

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
            with (
                patch(
                    'examples.YOLO_server_api.backend.models.YOLO',
                ) as mock_yolo_fresh,
                patch.object(Path, 'exists', return_value=True),
            ):
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
            patch(
                'examples.YOLO_server_api.backend.models.LAZY_LOAD_MODELS',
                False,
            ),
            patch('examples.YOLO_server_api.backend.models.YOLO') as mock_yolo,
            patch.object(Path, 'exists', return_value=True),
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

    def test_sahi_import_failure(self) -> None:
        """
        Test SAHI import failure handling.
        """
        with (
            patch.object(Path, 'exists', return_value=True),
            patch('builtins.__import__') as mock_import,
        ):
            # Mock import to raise an exception for SAHI
            def side_effect(name, *args, **kwargs):
                if 'sahi.predict' in name:
                    raise ImportError('SAHI not installed')
                else:
                    return __import__(name, *args, **kwargs)

            mock_import.side_effect = side_effect

            manager = DetectionModelManager.__new__(DetectionModelManager)
            manager.base_model_path = Path('models/pt/')
            manager.extension = '.pt'

            # Test SAHI mode with import failure
            with patch(
                'examples.YOLO_server_api.backend.models.USE_SAHI', True,
            ):
                with self.assertRaises(RuntimeError) as cm:
                    manager.load_single_model('yolo11n')

                self.assertIn(
                    'SAHI mode is enabled but the sahi package is not '
                    'installed',
                    str(cm.exception),
                )

    def test_file_not_found_errors(self) -> None:
        """
        Test FileNotFoundError handling for all model types.
        """
        manager = DetectionModelManager.__new__(DetectionModelManager)
        manager.base_model_path = Path('models/pt/')
        manager.extension = '.pt'

        # Test SAHI mode with missing file
        with (
            patch(
                'examples.YOLO_server_api.backend.models.USE_SAHI', True,
            ),
            patch.object(Path, 'exists', return_value=False),
        ):
            with self.assertRaises(FileNotFoundError) as cm:
                manager.load_single_model('nonexistent')
            self.assertIn('Model file not found', str(cm.exception))

        # Test TensorRT mode with missing file
        manager.base_model_path = Path('models/int8_engine/')
        manager.extension = '.engine'
        with (
            patch(
                'examples.YOLO_server_api.backend.models.USE_SAHI', False,
            ),
            patch(
                'examples.YOLO_server_api.backend.models.USE_TENSORRT', True,
            ),
            patch.object(Path, 'exists', return_value=False),
        ):
            with self.assertRaises(FileNotFoundError) as cm:
                manager.load_single_model('nonexistent')
            self.assertIn('Model file not found', str(cm.exception))

        # Test standard mode with missing file
        manager.base_model_path = Path('models/pt/')
        manager.extension = '.pt'
        with (
            patch(
                'examples.YOLO_server_api.backend.models.USE_SAHI', False,
            ),
            patch(
                'examples.YOLO_server_api.backend.models.USE_TENSORRT',
                False,
            ),
            patch.object(Path, 'exists', return_value=False),
        ):
            with self.assertRaises(FileNotFoundError) as cm:
                manager.load_single_model('nonexistent')
            self.assertIn('Model file not found', str(cm.exception))

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
        with patch.object(
            self.model_manager, 'load_single_model', return_value=mock_model,
        ) as mock_load:
            # Test successful reload
            result = self.model_manager.reload_model('yolo11x')
            self.assertTrue(result)
            mock_load.assert_called_once_with('yolo11x')
            self.assertEqual(self.model_manager.models['yolo11x'], mock_model)

    def test_reload_model_invalid_name(self) -> None:
        """
        Test reloading with invalid model name.
        """
        # Test with invalid model name
        result = self.model_manager.reload_model('invalid_model')
        self.assertFalse(result)

    def test_safe_load_invalid_name(self) -> None:
        """
        Test _safe_load with invalid model name.
        """
        manager = DetectionModelManager.__new__(DetectionModelManager)
        manager.model_names = ['valid_model']

        # Test with invalid name - should return early
        manager._safe_load('invalid_model')
        # No assertion needed, just ensuring it doesn't crash

    def test_safe_load_already_loaded(self) -> None:
        """
        Test _safe_load when model is already loaded.
        """
        manager = DetectionModelManager.__new__(DetectionModelManager)
        manager.model_names = ['test_model']
        manager.models = {'test_model': Mock()}
        manager._lru_order = []

        with patch.object(manager, '_touch_lru') as mock_touch:
            # Test when model is already loaded
            manager._safe_load('test_model')

            # Should call _touch_lru but not load the model again
            mock_touch.assert_called_once_with('test_model')

    def test_safe_load_exception_handling(self) -> None:
        """
        Test _safe_load exception handling.
        """
        manager = DetectionModelManager.__new__(DetectionModelManager)
        manager.model_names = ['test_model']
        manager.base_model_path = Path('models/pt/')
        manager.extension = '.pt'
        manager.models = {'test_model': None}
        manager._lru_order = []

        with (
            patch.object(
                manager, 'load_single_model',
                side_effect=Exception('Load failed'),
            ),
            patch.object(manager, '_touch_lru') as mock_touch,
            patch.object(manager, '_enforce_lru_limit') as mock_enforce,
        ):
            # Call _safe_load
            manager._safe_load('test_model')

            # Verify model is set to None on failure
            self.assertIsNone(manager.models['test_model'])
            mock_touch.assert_not_called()
            mock_enforce.assert_not_called()

    def test_reload_model_failure(self) -> None:
        """
        Test model reloading failure handling.
        """
        with patch.object(
            self.model_manager, 'load_single_model',
            side_effect=Exception('Model loading failed'),
        ):
            # Test reload failure
            result = self.model_manager.reload_model('yolo11x')
            self.assertFalse(result)

    def test_no_lazy_load_mode(self) -> None:
        """
        Test _enforce_lru_limit when lazy loading is disabled.
        """
        manager = DetectionModelManager.__new__(DetectionModelManager)
        manager.models = {'model1': Mock(), 'model2': Mock()}
        manager._lru_order = ['model1', 'model2']

        with patch(
            'examples.YOLO_server_api.backend.models.LAZY_LOAD_MODELS', False,
        ):
            # Should return early without doing anything
            manager._enforce_lru_limit()

            # Models should remain unchanged
            self.assertIsNotNone(manager.models['model1'])
            self.assertIsNotNone(manager.models['model2'])

    def test_type_checking_import(self) -> None:
        """
        Test the TYPE_CHECKING import block for coverage.
        """
        # This is a bit tricky since TYPE_CHECKING is False at runtime
        # We can at least verify the import structure is correct
        import typing

        with patch.object(typing, 'TYPE_CHECKING', True):
            # Reload the module to trigger the TYPE_CHECKING block
            import importlib
            importlib.reload(examples.YOLO_server_api.backend.models)

            # The module should still work
            self.assertTrue(
                hasattr(examples.YOLO_server_api.backend.models, 'ModelType'),
            )

    def test_get_model_lazy_loading(self) -> None:
        """
        Test get_model with lazy loading behavior.
        """
        manager = DetectionModelManager.__new__(DetectionModelManager)
        manager.models = {'test_model': None}

        with (
            patch.object(manager, '_safe_load') as mock_safe_load,
            patch.object(manager, '_touch_lru') as mock_touch,
        ):
            # Test when model is None - should trigger lazy loading
            manager.get_model('test_model')
            mock_safe_load.assert_called_once_with('test_model')

            # Test when model exists - should update LRU
            manager.models['test_model'] = Mock()
            manager.get_model('test_model')
            mock_touch.assert_called_with('test_model')

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

        # Mock the _safe_load method instead of load_single_model
        self.model_manager._safe_load = Mock()

        # Trigger the file modification event
        self.handler.on_modified(event)

        # Verify the _safe_load method was called
        self.model_manager._safe_load.assert_called_once_with('yolo11n')

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


class TestLRUEviction(unittest.TestCase):
    """Comprehensive test cases for LRU eviction logic.

    Tests the complete LRU eviction workflow including edge cases,
    error handling, and different configuration scenarios.
    """

    def test_lru_eviction_comprehensive(self) -> None:
        """
        Test comprehensive LRU eviction scenarios.
        """
        manager = DetectionModelManager.__new__(DetectionModelManager)

        # Test single model eviction
        manager.models = {'model1': Mock()}
        manager._lru_order = ['model1']

        with (
            patch(
                'examples.YOLO_server_api.backend.models.LAZY_LOAD_MODELS',
                True,
            ),
            patch(
                'examples.YOLO_server_api.backend.models.MAX_LOADED_MODELS',
                0,
            ),
            patch(
                'examples.YOLO_server_api.backend.models.USE_TENSORRT',
                False,
            ),
            patch(
                'examples.YOLO_server_api.backend.'
                'models.EXPLICIT_CUDA_CLEANUP',
                True,
            ),
            patch('torch.cuda.empty_cache') as m_c,
        ):
            manager._enforce_lru_limit()

            # Verify eviction
            self.assertIsNone(manager.models['model1'])
            self.assertEqual(manager._lru_order, [])
            m_c.assert_called()

    def test_lru_multiple_evictions(self) -> None:
        """
        Test multiple model evictions.
        """
        manager = DetectionModelManager.__new__(DetectionModelManager)

        # Setup multiple loaded models
        manager.models = {
            'model1': Mock(),
            'model2': Mock(),
            'model3': None,
        }
        manager._lru_order = ['model1', 'model2']

        with (
            patch(
                'examples.YOLO_server_api.backend.models.LAZY_LOAD_MODELS',
                True,
            ),
            patch(
                'examples.YOLO_server_api.backend.models.MAX_LOADED_MODELS',
                1,
            ),
            patch(
                'examples.YOLO_server_api.backend.models.USE_TENSORRT',
                False,
            ),
            patch(
                'examples.YOLO_server_api.backend.'
                'models.EXPLICIT_CUDA_CLEANUP',
                True,
            ),
            patch('torch.cuda.empty_cache') as m_c,
        ):
            manager._enforce_lru_limit()

            # Should evict oldest model (model1)
            self.assertIsNone(manager.models['model1'])
            self.assertIsNotNone(manager.models['model2'])
            self.assertEqual(manager._lru_order, ['model2'])
            m_c.assert_called()

    def test_cuda_cleanup_exception_handling(self) -> None:
        """
        Test CUDA cleanup exception handling.
        """
        manager = DetectionModelManager.__new__(DetectionModelManager)
        manager.models = {'model1': Mock()}
        manager._lru_order = ['model1']

        with (
            patch(
                'examples.YOLO_server_api.backend.models.LAZY_LOAD_MODELS',
                True,
            ),
            patch(
                'examples.YOLO_server_api.backend.models.MAX_LOADED_MODELS',
                0,
            ),
            patch(
                'examples.YOLO_server_api.backend.models.USE_TENSORRT',
                False,
            ),
            patch(
                'examples.YOLO_server_api.backend.'
                'models.EXPLICIT_CUDA_CLEANUP',
                True,
            ),
            patch(
                'torch.cuda.empty_cache',
                side_effect=Exception('CUDA error'),
            ),
        ):
            # Should not raise exception
            manager._enforce_lru_limit()

            # Model should still be evicted despite CUDA cleanup failure
            self.assertIsNone(manager.models['model1'])

    def test_torch_import_failure(self) -> None:
        """
        Test torch import failure in CUDA cleanup.
        """
        manager = DetectionModelManager.__new__(DetectionModelManager)
        manager.models = {'model1': Mock()}
        manager._lru_order = ['model1']

        # Mock torch module to raise ImportError
        import sys
        original_modules = sys.modules.copy()
        if 'torch' in sys.modules:
            del sys.modules['torch']

        with (
            patch(
                'examples.YOLO_server_api.backend.models.LAZY_LOAD_MODELS',
                True,
            ),
            patch(
                'examples.YOLO_server_api.backend.models.MAX_LOADED_MODELS',
                0,
            ),
            patch(
                'examples.YOLO_server_api.backend.models.USE_TENSORRT',
                False,
            ),
            patch(
                'examples.YOLO_server_api.backend.'
                'models.EXPLICIT_CUDA_CLEANUP',
                True,
            ),
            patch.dict(
                sys.modules, {'torch': None},
            ),  # Make torch import fail
        ):
            # Should not raise exception even if torch import fails
            manager._enforce_lru_limit()

            # Model should still be evicted
            self.assertIsNone(manager.models['model1'])

        # Restore original modules
        sys.modules.clear()
        sys.modules.update(original_modules)

    def test_eviction_with_tensorrt_mode(self) -> None:
        """
        Test eviction without CUDA cleanup in TensorRT mode.
        """
        manager = DetectionModelManager.__new__(DetectionModelManager)
        manager.models = {'model1': Mock()}
        manager._lru_order = ['model1']

        with (
            patch(
                'examples.YOLO_server_api.backend.models.LAZY_LOAD_MODELS',
                True,
            ),
            patch(
                'examples.YOLO_server_api.backend.models.MAX_LOADED_MODELS',
                0,
            ),
            patch(
                'examples.YOLO_server_api.backend.models.USE_TENSORRT',
                True,
            ),
            patch(
                'examples.YOLO_server_api.backend.'
                'models.EXPLICIT_CUDA_CLEANUP',
                True,
            ),
            patch('torch.cuda.empty_cache') as m_c,
        ):
            manager._enforce_lru_limit()

            # Verify eviction without CUDA cleanup
            self.assertIsNone(manager.models['model1'])
            m_c.assert_not_called()

    def test_eviction_exception_handling(self) -> None:
        """
        Test exception handling during eviction process.
        """
        manager = DetectionModelManager.__new__(DetectionModelManager)

        # Create a custom dict that fails on None assignment
        class FailingDict(dict):
            def __setitem__(self, key, value):
                if value is None:
                    raise Exception('Simulated failure')
                super().__setitem__(key, value)

        failing_dict = FailingDict({'model1': Mock()})
        manager.models = failing_dict
        manager._lru_order = ['model1']

        with (
            patch(
                'examples.YOLO_server_api.backend.models.LAZY_LOAD_MODELS',
                True,
            ),
            patch(
                'examples.YOLO_server_api.backend.models.MAX_LOADED_MODELS',
                0,
            ),
            patch(
                'examples.YOLO_server_api.backend.models.USE_TENSORRT',
                False,
            ),
            patch(
                'examples.YOLO_server_api.backend.'
                'models.EXPLICIT_CUDA_CLEANUP',
                False,
            ),
        ):
            # Should not raise exception due to outer try-except
            manager._enforce_lru_limit()

            # Verify model was not evicted due to exception
            self.assertIsNotNone(manager.models['model1'])

    def test_eviction_empty_lru_order(self) -> None:
        """
        Test eviction when LRU order is empty (safety check).
        """
        manager = DetectionModelManager.__new__(DetectionModelManager)
        manager.models = {'model1': Mock()}
        manager._lru_order = []  # Empty LRU order

        with (
            patch(
                'examples.YOLO_server_api.backend.models.LAZY_LOAD_MODELS',
                True,
            ),
            patch(
                'examples.YOLO_server_api.backend.models.MAX_LOADED_MODELS',
                0,
            ),
        ):
            # Should not raise exception due to safety check
            manager._enforce_lru_limit()

            # Model should remain since LRU order is empty
            self.assertIsNotNone(manager.models['model1'])

    def test_eviction_cleanup_disabled(self) -> None:
        """
        Test eviction when CUDA cleanup is disabled.
        """
        manager = DetectionModelManager.__new__(DetectionModelManager)
        manager.models = {'model1': Mock()}
        manager._lru_order = ['model1']

        with (
            patch(
                'examples.YOLO_server_api.backend.models.LAZY_LOAD_MODELS',
                True,
            ),
            patch(
                'examples.YOLO_server_api.backend.models.MAX_LOADED_MODELS',
                0,
            ),
            patch(
                'examples.YOLO_server_api.backend.models.USE_TENSORRT',
                False,
            ),
            patch(
                'examples.YOLO_server_api.backend.'
                'models.EXPLICIT_CUDA_CLEANUP',
                False,
            ),
            patch('torch.cuda.empty_cache') as m_c,
        ):
            manager._enforce_lru_limit()

            # Verify eviction without CUDA cleanup
            self.assertIsNone(manager.models['model1'])
            m_c.assert_not_called()


if __name__ == '__main__':
    unittest.main()

'''
pytest \
    --cov=examples.YOLO_server_api.backend.models \
    --cov-report=term-missing \
    tests/examples/YOLO_server_api/backend/models_test.py
'''
