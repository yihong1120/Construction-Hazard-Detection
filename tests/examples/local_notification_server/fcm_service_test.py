from __future__ import annotations

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from examples.local_notification_server.fcm_service import init_firebase_app
from examples.local_notification_server.fcm_service import (
    send_fcm_notification_service,
)


class TestInitFirebaseApp(unittest.TestCase):
    """
    Test suite for initialising the Firebase application.
    """

    @patch('firebase_admin.credentials.Certificate', return_value=MagicMock())
    @patch('firebase_admin.initialize_app')
    @patch('firebase_admin._apps', new=[])
    def test_init_firebase_app_when_not_inited(
        self,
        mock_init_app: MagicMock,
        mock_cred: MagicMock,
    ) -> None:
        """
        Test that init_firebase_app() calls initialise_app
        when no apps are registered.

        Args:
            mock_init_app (MagicMock): Mocked initialise_app function.
        """
        init_firebase_app()
        mock_init_app.assert_called_once()
        mock_cred.assert_called_once()

    @patch('firebase_admin.credentials.Certificate', return_value=MagicMock())
    @patch('firebase_admin.initialize_app')
    @patch('firebase_admin._apps', new=['already_inited'])
    def test_init_firebase_app_when_already_inited(
        self,
        mock_init_app: MagicMock,
        mock_cred: MagicMock,
    ) -> None:
        """
        Test that init_firebase_app() does not re-initialise
        if an app is already registered.

        Args:
            mock_init_app (MagicMock): Mocked initialise_app function.
        """
        init_firebase_app()
        mock_init_app.assert_not_called()
        mock_cred.assert_not_called()


@patch(
    'examples.local_notification_server.fcm_service.init_firebase_app',
    lambda: None,
)
class TestSendFCMNotificationService(unittest.IsolatedAsyncioTestCase):
    """
    Test suite for sending FCM notifications
    using the send_fcm_notification_service function.
    """

    async def test_no_tokens(self) -> None:
        """
        Test that sending notifications with an empty token list returns False.
        """
        result = await send_fcm_notification_service([], 'Title', 'Body')
        self.assertFalse(result)

    @patch('firebase_admin.messaging.send_each')
    async def test_all_success(self, mock_send_each: MagicMock) -> None:
        """
        Test that the service returns True when all tokens send successfully.

        Args:
            mock_send_each (MagicMock): Mocked send_each function.
        """
        mock_response = MagicMock()
        mock_response.failure_count = 0
        mock_response.responses = [MagicMock(success=True)]
        mock_send_each.return_value = mock_response

        tokens = ['valid_token']
        result = await send_fcm_notification_service(tokens, 'Title', 'Body')
        self.assertTrue(result)

    @patch('firebase_admin.messaging.send_each')
    async def test_partial_fail(self, mock_send_each: MagicMock) -> None:
        """
        Test that the service returns False when at least one token fails.

        Args:
            mock_send_each (MagicMock): Mocked send_each function.
        """
        mock_response = MagicMock()
        mock_response.failure_count = 1
        mock_response.responses = [
            MagicMock(success=True), MagicMock(success=False),
        ]
        mock_send_each.return_value = mock_response

        tokens = ['valid_token', 'invalid_token']
        result = await send_fcm_notification_service(tokens, 'Title', 'Body')
        self.assertFalse(result)

    @patch('firebase_admin.messaging.send_each')
    async def test_exception(self, mock_send_each: MagicMock) -> None:
        """
        Test that the service returns False when an exception occurs.

        Args:
            mock_send_each (MagicMock): Mocked send_each function.
        """
        mock_send_each.side_effect = Exception('FCM send error')

        tokens = ['token']
        result = await send_fcm_notification_service(tokens, 'Title', 'Body')
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()

"""
pytest \
    --cov=examples.local_notification_server.fcm_service \
    --cov-report=term-missing \
    tests/examples/local_notification_server/fcm_service_test.py
"""
