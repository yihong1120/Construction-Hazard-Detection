from __future__ import annotations

import unittest
from collections.abc import MutableMapping
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx

from src.notifiers.fcm_notifier import FCMSender
from src.utils import TokenManager


class TestFCMSender(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for the FCMSender class.

    This class contains asynchronous unit tests for the FCMSender class,
    verifying its behaviour when sending FCM messages under various conditions.
    """

    def setUp(self) -> None:
        """
        Set up the test case with a mock shared token and example data.
        """
        # Mock shared token dictionary for authentication simulation.
        self.mock_shared_token: MutableMapping[str, str | bool] = {
            'access_token': 'initial_token',
            'refresh_token': '',
            'is_refreshing': False,
        }
        # Instantiate FCMSender with test parameters.
        self.sender: FCMSender = FCMSender(
            api_url='http://testserver.local/api/fcm',
            max_retries=2,
            timeout=5,
        )
        # Override FCMSender's shared_token with the mock token.
        self.sender.shared_token = self.mock_shared_token
        # Example data used in tests.
        self.site: str = 'TestSite'
        self.stream_name: str = 'TestStream'
        self.mock_message: dict[str, dict[str, int]] = {
            'warn_code': {
                'count': 3,
            },
        }
        self.image_path: str = 'http://example.com/image.jpg'
        self.violation_id: int = 123

    @patch.object(TokenManager, 'authenticate', new_callable=AsyncMock)
    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_send_fcm_no_token_triggers_authenticate(
        self: TestFCMSender,
        mock_post: AsyncMock,
        mock_authenticate: AsyncMock,
    ) -> None:
        """
        Test that if no token is present,
        FCMSender calls authenticate before sending.

        Args:
            mock_post (AsyncMock): Mocked post method of httpx.AsyncClient.
            mock_authenticate (AsyncMock):
                Mocked authenticate method of TokenManager.

        Returns:
            None
        """
        # Clear the shared token to simulate no existing token.
        self.sender.shared_token.clear()

        # Mock a successful 200 response with JSON data indicating success.
        mock_response: MagicMock = MagicMock(status_code=200)
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.side_effect = None
        mock_post.return_value = mock_response

        # Attempt to send FCM message,
        # expecting authentication to be triggered.
        result: bool = await self.sender.send_fcm_message_to_site(
            site=self.site,
            stream_name=self.stream_name,
            message=self.mock_message,
            image_path=self.image_path,
            violation_id=self.violation_id,
        )
        self.assertTrue(
            result,
            'Expected True when the request is successful.',
        )

        # Verify authenticate was awaited once,
        # since no token was present initially.
        mock_authenticate.assert_awaited_once()
        mock_post.assert_awaited_once()

    @patch.object(TokenManager, 'authenticate', new_callable=AsyncMock)
    @patch.object(TokenManager, 'handle_401', new_callable=AsyncMock)
    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_send_fcm_success_true(
        self: TestFCMSender,
        mock_post: AsyncMock,
        mock_handle_401: AsyncMock,
        mock_authenticate: AsyncMock,
    ) -> None:
        """
        Test that if the API returns 200 with {"success": True},
        the method returns True.

        Args:
            mock_post (AsyncMock): Mocked post method of httpx.AsyncClient.
            mock_handle_401 (AsyncMock):
                Mocked handle_401 method of TokenManager.
            mock_authenticate (AsyncMock):
                Mocked authenticate method of TokenManager.

        Returns:
            None
        """
        # Simulate a valid token.
        self.sender.shared_token['access_token'] = 'valid_token'

        # Mock a 200 response with success.
        mock_response: MagicMock = MagicMock(status_code=200)
        mock_response.json.return_value = {'success': True}
        mock_response.raise_for_status.side_effect = None
        mock_post.return_value = mock_response

        # Attempt to send FCM message, expecting success.
        result: bool = await self.sender.send_fcm_message_to_site(
            site=self.site,
            stream_name=self.stream_name,
            message=self.mock_message,
        )
        self.assertTrue(
            result,
            'Expected True when the API indicates success.',
        )
        mock_authenticate.assert_not_awaited()
        mock_handle_401.assert_not_awaited()
        mock_post.assert_awaited_once()

    @patch.object(TokenManager, 'authenticate', new_callable=AsyncMock)
    @patch.object(TokenManager, 'handle_401', new_callable=AsyncMock)
    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_send_fcm_success_false(
        self: TestFCMSender,
        mock_post: AsyncMock,
        mock_handle_401: AsyncMock,
        mock_authenticate: AsyncMock,
    ) -> None:
        """
        Test that if the API returns 200 but {"success": False},
        the method returns False.

        Args:
            mock_post (AsyncMock): Mocked post method of httpx.AsyncClient.
            mock_handle_401 (AsyncMock):
                Mocked handle_401 method of TokenManager.
            mock_authenticate (AsyncMock):
                Mocked authenticate method of TokenManager.

        Returns:
            None
        """
        self.sender.shared_token['access_token'] = 'valid_token'
        # The API returns 200 but no 'success' key in JSON.
        mock_response: MagicMock = MagicMock(status_code=200)
        mock_response.json.return_value = {'not_success_field': 123}
        mock_post.return_value = mock_response

        # Attempt to send FCM message,
        # expecting failure due to missing 'success' field.
        result: bool = await self.sender.send_fcm_message_to_site(
            site=self.site,
            stream_name=self.stream_name,
            message=self.mock_message,
        )
        self.assertFalse(
            result,
            "Expected False when the response lacks {'success': True}.",
        )

    @patch.object(TokenManager, 'handle_401', new_callable=AsyncMock)
    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_send_fcm_401_refresh_token_and_retry_once(
        self: TestFCMSender,
        mock_post: AsyncMock,
        mock_handle_401: AsyncMock,
    ) -> None:
        """
        Test that if the API returns 401,
        the method attempts to refresh the token and retries the request once.

        Args:
            mock_post (AsyncMock): Mocked post method of httpx.AsyncClient.
            mock_handle_401 (AsyncMock):
                Mocked handle_401 method of TokenManager.

        Returns:
            None
        """
        self.sender.shared_token['access_token'] = 'expired_token'

        # First call => 401
        mock_response_1: MagicMock = MagicMock(status_code=401)
        mock_response_1.json.return_value = {}
        # Second call => 200 + success
        mock_response_2: MagicMock = MagicMock(status_code=200)
        mock_response_2.json.return_value = {'success': True}

        # Mock the side effect of post to simulate retry after token refresh.
        mock_post.side_effect = [mock_response_1, mock_response_2]

        # Attempt to send FCM message, expecting retry and eventual success.
        result: bool = await self.sender.send_fcm_message_to_site(
            site=self.site,
            stream_name=self.stream_name,
            message=self.mock_message,
        )
        self.assertTrue(
            result,
            'Expected True after refresh and successful retry.',
        )
        self.assertEqual(
            mock_post.await_count,
            2,
            'Expected two POST calls.',
        )
        mock_handle_401.assert_awaited_once()

    @patch.object(TokenManager, 'handle_401', new_callable=AsyncMock)
    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_send_fcm_401_exceed_max_retries(
        self: TestFCMSender,
        mock_post: AsyncMock,
        mock_handle_401: AsyncMock,
    ) -> None:
        """
        Test that if the API keeps returning 401 even after max_retries,
        the method eventually returns False.

        Args:
            mock_post (AsyncMock): Mocked post method of httpx.AsyncClient.
            mock_handle_401 (AsyncMock):
                Mocked handle_401 method of TokenManager.

        Returns:
            None
        """
        self.sender.shared_token['access_token'] = 'always_expired'

        # Suppose all attempts return 401, exceeding max_retries.
        mock_response: MagicMock = MagicMock(status_code=401)
        mock_response.json.return_value = {}
        mock_post.return_value = mock_response

        # Attempt to send FCM message, expecting repeated failures
        # and eventual False.
        result: bool = await self.sender.send_fcm_message_to_site(
            site=self.site,
            stream_name=self.stream_name,
            message=self.mock_message,
        )
        self.assertFalse(
            result,
            'Expected False if all attempts result in 401.',
        )
        # Check that the post method was called more than max_retries.
        self.assertGreaterEqual(
            mock_post.await_count,
            3,
            'Expected repeated attempts.',
        )
        self.assertGreaterEqual(
            mock_handle_401.await_count,
            2,
            'Expected multiple handle_401 calls.',
        )

    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_send_fcm_request_error(
        self: TestFCMSender,
        mock_post: AsyncMock,
    ) -> None:
        """
        Test that if an httpx.RequestError occurs (e.g. network issues),
        the method returns False.

        Args:
            mock_post (AsyncMock): Mocked post method of httpx.AsyncClient.

        Returns:
            None
        """
        self.sender.shared_token['access_token'] = 'valid_token'
        # Simulate network error by raising RequestError.
        mock_post.side_effect = httpx.RequestError('Network error')

        # Attempt to send FCM message, expecting failure due to network error.
        result: bool = await self.sender.send_fcm_message_to_site(
            site=self.site,
            stream_name=self.stream_name,
            message=self.mock_message,
        )
        self.assertFalse(
            result,
            'Expected False when a RequestError is raised by httpx.',
        )

    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_send_fcm_http_status_error(
        self: TestFCMSender,
        mock_post: AsyncMock,
    ) -> None:
        """
        Test that if an httpx.HTTPStatusError occurs (e.g. 403 Forbidden),
        the method returns False.

        Args:
            mock_post (AsyncMock): Mocked post method of httpx.AsyncClient.

        Returns:
            None
        """
        self.sender.shared_token['access_token'] = 'valid_token'

        # Simulate a 403 Forbidden or similar scenario.
        mock_response: MagicMock = MagicMock(
            status_code=403,
            reason_phrase='Forbidden',
        )
        # Must raise HTTPStatusError upon raise_for_status().
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            'Forbidden',
            request=MagicMock(),
            response=mock_response,
        )
        mock_post.return_value = mock_response

        # Attempt to send FCM message, expecting failure
        # due to HTTPStatusError.
        result: bool = await self.sender.send_fcm_message_to_site(
            site=self.site,
            stream_name=self.stream_name,
            message=self.mock_message,
        )
        self.assertFalse(
            result,
            'Expected False for non-401 HTTPStatusError.',
        )


if __name__ == '__main__':
    unittest.main()

'''
pytest \
    --cov=src.notifiers.fcm_notifier \
    --cov-report=term-missing \
    tests/src/notifiers/fcm_notifier_test.py
'''
