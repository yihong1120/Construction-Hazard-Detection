import unittest
from unittest.mock import patch, MagicMock
import atexit
from flask import Flask
from flask_jwt_extended import JWTManager
from examples.YOLO_server_api.app import app, scheduler, db

class TestYOLOServerAPI(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the Flask test client and other necessary mocks.
        """
        self.app = app
        self.app.testing = True
        self.client = self.app.test_client()

    @patch('examples.YOLO_server_api.app.JWTManager')
    def test_jwt_initialization(self, mock_jwt_manager: MagicMock) -> None:
        """
        Test if the JWTManager is properly initialized.
        """
        mock_jwt_manager.return_value = JWTManager(self.app)
        jwt = mock_jwt_manager(self.app)
        self.assertIsInstance(jwt, JWTManager)

    @patch('examples.YOLO_server_api.app.secrets.token_urlsafe', return_value='mocked_secret_key')
    def test_jwt_secret_key(self, mock_secrets: MagicMock) -> None:
        """
        Test that the JWT secret key is securely generated.
        """
        with patch.dict(self.app.config, {'JWT_SECRET_KEY': mock_secrets.return_value}):
            self.assertEqual(self.app.config['JWT_SECRET_KEY'], 'mocked_secret_key')

    @patch('examples.YOLO_server_api.app.db.init_app')
    def test_database_initialization(self, mock_init_app: MagicMock) -> None:
        """
        Test if the database is properly initialized.
        """
        with self.app.app_context():
            db.init_app(self.app)
            mock_init_app.assert_called_once_with(self.app)

    def test_routes_registration(self) -> None:
        """
        Test that the blueprints are properly registered.
        """
        self.assertIn('auth', self.app.blueprints)
        self.assertIn('detection', self.app.blueprints)
        self.assertIn('models', self.app.blueprints)

    @patch('examples.YOLO_server_api.app.update_secret_key')
    def test_scheduler_job(self, mock_update_secret_key: MagicMock) -> None:
        """
        Test that the scheduler has a job to update the JWT secret key every 30 days.
        """
        job = scheduler.get_jobs()[0]
        self.assertEqual(job.func.__name__, '<lambda>')
        self.assertEqual(job.trigger.interval.days, 30)

    @patch('examples.YOLO_server_api.app.atexit.register')
    def test_scheduler_shutdown(self, mock_atexit_register: MagicMock) -> None:
        """
        Test that the scheduler is shut down gracefully upon application exit.
        """
        atexit.register(lambda: scheduler.shutdown())
        mock_atexit_register.assert_called_once()

    def test_app_running_configuration(self) -> None:
        """
        Test that the application runs with the expected configurations.
        """
        with patch.object(self.app, 'run') as mock_run:
            self.app.run(threaded=True, host='0.0.0.0', port=5000)
            mock_run.assert_called_once_with(threaded=True, host='0.0.0.0', port=5000)

    def tearDown(self) -> None:
        """
        Clean up after each test.
        """
        self.client = None

if __name__ == '__main__':
    unittest.main()