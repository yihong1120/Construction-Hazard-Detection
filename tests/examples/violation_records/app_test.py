from __future__ import annotations

import unittest
from typing import ClassVar
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi.testclient import TestClient
from fastapi_jwt import JwtAuthorizationCredentials

from examples.auth.jwt_config import jwt_access
from examples.violation_records.app import app
from examples.violation_records.app import main


class TestViolationRecordsApp(unittest.IsolatedAsyncioTestCase):
    """
    Collection of test cases for the violation_records FastAPI application.
    """

    # Class variable to hold the test client.
    client: ClassVar[TestClient]

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the test client and override the JWT dependency for testing.
        """
        super().setUpClass()

        def override_jwt() -> JwtAuthorizationCredentials:
            """
            Mock implementation of the jwt_access dependency for testing.
            """
            return JwtAuthorizationCredentials(
                subject={'username': 'test_user'},
            )

        # Override the jwt_access dependency with a mock implementation.
        app.dependency_overrides[jwt_access] = override_jwt

        # Create a test client for the application.
        cls.client = TestClient(app)

    @patch('uvicorn.run')
    def test_run_uvicorn(self, mock_uvicorn_run: MagicMock) -> None:
        """
        Test the main function to ensure it runs the FastAPI application with
        Uvicorn.

        Args:
            mock_uvicorn_run (MagicMock): Mock object for the Uvicorn run
                function.
        """
        main()
        mock_uvicorn_run.assert_called_once_with(
            'examples.violation_records.app:app',
            host='0.0.0.0',
            port=8081,
            reload=True,
        )


if __name__ == '__main__':
    unittest.main()

'''
pytest \
    --cov=examples.violation_records.app \
    --cov-report=term-missing tests/examples/violation_records/app_test.py
'''
