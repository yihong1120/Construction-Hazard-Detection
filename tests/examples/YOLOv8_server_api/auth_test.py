import unittest
from unittest.mock import patch, MagicMock
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager

from examples.YOLOv8_server_api.auth import auth_blueprint
from examples.YOLOv8_server_api.models import User, db  # 確保從 models 中導入 db

# 不再需要創建全局的 SQLAlchemy 實例
jwt = JWTManager()

class AuthBlueprintTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up the Flask application and register extensions once for all tests.
        """
        cls.app = Flask(__name__)
        cls.app.config['TESTING'] = True
        cls.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        cls.app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        cls.app.config['JWT_SECRET_KEY'] = 'super-secret'

        # Initialize the database and JWT with the app
        db.init_app(cls.app)
        jwt.init_app(cls.app)

        # Register the blueprint
        cls.app.register_blueprint(auth_blueprint)

    def setUp(self):
        """
        Ensure that each test runs with a fresh application context.
        """
        self.app_context = self.app.app_context()
        self.app_context.push()

        # Create the test client
        self.client = self.app.test_client()

        # Create all tables
        with self.app_context:
            db.create_all()

    def tearDown(self):
        """
        Pop the application context to clean up.
        """
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    @patch('examples.YOLOv8_server_api.auth.user_cache')
    @patch('examples.YOLOv8_server_api.auth.User.query')
    def test_create_token_success_with_cache(self, mock_user_query, mock_user_cache):
        """
        Test successful token creation with user data from cache.
        """
        # Setup mock cache to return a user
        mock_user = MagicMock()
        mock_user.check_password.return_value = True
        mock_user_cache.get.return_value = mock_user

        response = self.client.post('/token', json={
            'username': 'testuser',
            'password': 'password123'
        })

        # Check that the cache was accessed
        mock_user_cache.get.assert_called_once_with('testuser')
        # Check that the response contains a token and has a 200 status code
        self.assertEqual(response.status_code, 200)
        self.assertIn('access_token', response.json)

    @patch('examples.YOLOv8_server_api.auth.user_cache')
    @patch('examples.YOLOv8_server_api.auth.User.query')
    def test_create_token_success_with_db(self, mock_user_query, mock_user_cache):
        """
        Test successful token creation with user data from the database.
        """
        # Setup mock cache to return None (forcing DB query)
        mock_user_cache.get.return_value = None
        # Setup mock database query to return a user
        mock_user = MagicMock()
        mock_user.check_password.return_value = True
        mock_user_query.filter_by.return_value.first.return_value = mock_user

        response = self.client.post('/token', json={
            'username': 'testuser',
            'password': 'password123'
        })

        # Check that the DB query was made and the user was cached
        mock_user_query.filter_by.assert_called_once_with(username='testuser')
        mock_user_cache.__setitem__.assert_called_once_with('testuser', mock_user)
        # Check that the response contains a token and has a 200 status code
        self.assertEqual(response.status_code, 200)
        self.assertIn('access_token', response.json)

    @patch('examples.YOLOv8_server_api.auth.user_cache')
    @patch('examples.YOLOv8_server_api.auth.User.query')
    def test_create_token_invalid_credentials(self, mock_user_query, mock_user_cache):
        """
        Test token creation failure due to invalid credentials.
        """
        # Setup mock cache and DB to return no valid user
        mock_user_cache.get.return_value = None
        mock_user_query.filter_by.return_value.first.return_value = None

        response = self.client.post('/token', json={
            'username': 'testuser',
            'password': 'wrongpassword'
        })

        # Check that the cache and DB were accessed but no valid user was found
        mock_user_cache.get.assert_called_once_with('testuser')
        mock_user_query.filter_by.assert_called_once_with(username='testuser')
        # Check that the response indicates failure and has a 401 status code
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json['msg'], 'Wrong user name or passcode.')

    @patch('examples.YOLOv8_server_api.auth.User.query')
    @patch('examples.YOLOv8_server_api.auth.user_cache')
    def test_create_token_user_not_in_db_or_cache(self, mock_user_cache, mock_user_query):
        """
        Test token creation failure when user is not found in cache or DB.
        """
        # Setup mock cache to return None (forcing DB query)
        mock_user_cache.get.return_value = None
        # Setup mock database query to return None
        mock_user_query.filter_by.return_value.first.return_value = None

        response = self.client.post('/token', json={
            'username': 'nonexistentuser',
            'password': 'password123'
        })

        # Check that the cache was accessed but no user was found
        mock_user_cache.get.assert_called_once_with('nonexistentuser')
        mock_user_query.filter_by.assert_called_once_with(username='nonexistentuser')
        # Check that the response indicates failure and has a 401 status code
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json['msg'], 'Wrong user name or passcode.')

if __name__ == '__main__':
    unittest.main()