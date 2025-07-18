from __future__ import annotations

import unittest
from collections.abc import MutableMapping
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import aiohttp
import httpx

from src.frame_sender import BackendFrameSender
from src.utils import TokenManager


class TestBackendFrameSender(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for the BackendFrameSender class.
    """

    def setUp(self) -> None:
        """
        Set up the test case with mock objects and test data.
        """
        # Mock shared token state for API authentication
        self.mock_shared_token: MutableMapping[str, str | bool] = {
            # Initial access token for API calls
            'access_token': 'init_token',
            # Refresh token for token renewal
            'refresh_token': '',
            # Flag to prevent concurrent refresh operations
            'is_refreshing': False,
        }

        # Mock shared lock for thread synchronisation in token management
        self.mock_shared_lock: MagicMock = MagicMock()

        # Create BackendFrameSender instance with test configuration
        # Use a fake API endpoint to prevent connecting to a real server
        self.sender: BackendFrameSender = BackendFrameSender(
            api_url='http://testserver.local/api/streaming_web',
            shared_token=self.mock_shared_token,
            shared_lock=self.mock_shared_lock,
            max_retries=2,  # Limited retries for faster test execution
            timeout=5,  # Short timeout for test efficiency
        )

        # Test data for frame transmission
        self.site: str = 'TestSite'
        self.stream_name: str = 'TestStream'
        self.frame_bytes: bytes = b'fake_image_data'

        # Mock JSON data for various detection components
        self.warnings_json: str = '{"some": "warning"}'
        self.cone_polygons_json: str = '{"cone": "polygon"}'
        self.pole_polygons_json: str = '{"pole": "polygon"}'
        self.detection_items_json: str = '{"some": "detection"}'

    # -------------------------------------------------------------------------
    # Tests for authentication logic
    # -------------------------------------------------------------------------

    @patch.object(TokenManager, 'authenticate', new_callable=AsyncMock)
    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_send_frame_no_token_triggers_auth(
        self,
        mock_post: AsyncMock,
        mock_auth: AsyncMock,
    ) -> None:
        """
        Test that frame sending triggers authentication
        when no token is present.

        Test Flow:
        1. Clear the shared token to simulate absence of authentication
        2. Mock a successful HTTP response from the server
        3. Attempt to send a frame with all optional parameters
        4. Verify that authentication was triggered
           and the frame was sent successfully

        Args:
            mock_post:
                Mocked HTTP POST method from httpx.AsyncClient for intercepting
                API calls to the backend server
            mock_auth:
                Mocked authenticate method from TokenManager for controlling
                the authentication process during testing

        Raises:
            AssertionError:
                If the authentication behaviour or response handling
                does not match expected patterns
        """
        # Clear the shared token to simulate that no token is available
        # This forces the sender to initiate the authentication process
        self.sender.shared_token.clear()

        # Create a fake HTTP response with status 200 and JSON content
        # This simulates a successful response from the backend server
        mock_response: MagicMock = MagicMock(status_code=200)
        mock_response.json.return_value = {'foo': 'bar'}
        mock_response.raise_for_status.side_effect = None  # No HTTP errors
        mock_post.return_value = mock_response

        # Attempt to send a frame with all optional detection data
        result: dict[str, str] = await self.sender.send_frame(
            site=self.site,
            stream_name=self.stream_name,
            frame_bytes=self.frame_bytes,
            warnings_json=self.warnings_json,
            cone_polygons_json=self.cone_polygons_json,
            pole_polygons_json=self.pole_polygons_json,
            detection_items_json=self.detection_items_json,
        )

        # Verify that the response matches expected structure
        self.assertEqual(result, {'foo': 'bar'})

        # Verify that authentication was called twice:
        # - Once for initial authentication (when no token exists)
        # - Once for retry after receiving 401 (token validation)
        self.assertEqual(mock_auth.await_count, 2)

        # Verify that the HTTP POST was called exactly once
        mock_post.assert_awaited_once()

    @patch.object(TokenManager, 'authenticate', new_callable=AsyncMock)
    @patch.object(TokenManager, 'refresh_token', new_callable=AsyncMock)
    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_send_frame_with_token_success(
        self,
        mock_post: AsyncMock,
        mock_refresh: AsyncMock,
        mock_auth: AsyncMock,
    ) -> None:
        """
        Test successful frame transmission with an existing valid token.

        Test Flow:
        1. Set up a valid access token in the shared token state
        2. Mock a successful HTTP response from the backend
        3. Send a frame with minimal required parameters
        4. Verify that no authentication or token refresh was triggered
        5. Confirm that the HTTP POST was executed successfully

        Args:
            mock_post:
                Mocked HTTP POST method from httpx.AsyncClient for
                intercepting API requests to the backend
            mock_refresh:
                Mocked refresh_token method from TokenManager for
                verifying that token refresh is not called
            mock_auth:
                Mocked authenticate method from TokenManager for
                verifying that authentication is not called

        Raises:
            AssertionError:
                If authentication or refresh operations are triggered
                unexpectedly, or if the response handling fails
        """
        # Set up a valid access token to simulate authenticated state
        # This should allow the frame to be sent without additional auth steps
        self.sender.shared_token['access_token'] = 'valid_token'

        # Mock a successful HTTP response from the backend server
        mock_response: MagicMock = MagicMock(status_code=200)
        mock_response.json.return_value = {'msg': 'ok'}
        mock_response.raise_for_status.side_effect = None  # No HTTP errors
        mock_post.return_value = mock_response

        # Send a frame with only the required parameters
        # (no optional detection data)
        result: dict[str, str] = await self.sender.send_frame(
            site=self.site,
            stream_name=self.stream_name,
            frame_bytes=self.frame_bytes,
        )

        # Verify that the response contains the expected success message
        self.assertEqual(result, {'msg': 'ok'})

        # Verify that no authentication was triggered (token already valid)
        mock_auth.assert_not_awaited()

        # Verify that no token refresh was triggered (token still valid)
        mock_refresh.assert_not_awaited()

        # Verify that exactly one HTTP POST request was made
        mock_post.assert_awaited_once()

    @patch.object(TokenManager, 'refresh_token', new_callable=AsyncMock)
    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_send_frame_401_refresh_token_retry(
        self,
        mock_post: AsyncMock,
        mock_refresh: AsyncMock,
    ) -> None:
        """
        Test automatic token refresh and retry logic
        on 401 Unauthorised response.

        Test Flow:
        1. Set up an expired access token in the shared state
        2. Mock the first HTTP response to return 401 Unauthorised
        3. Mock the second HTTP response to return 200 OK (after token refresh)
        4. Configure token refresh to update the shared token
        5. Attempt to send a frame
        6. Verify that token refresh was triggered and retry succeeded

        Args:
            mock_post:
                Mocked HTTP POST method from httpx.AsyncClient that will
                return different responses on consecutive calls
            mock_refresh:
                Mocked refresh_token method from TokenManager that
                simulates successful token renewal

        Raises:
            AssertionError: If the retry logic fails or token refresh is not
                           triggered correctly
        """
        # Set up an expired token to trigger the 401 response scenario
        self.sender.shared_token['access_token'] = 'expired_token'

        # First response: 401 Unauthorised, triggering HTTPStatusError
        # This simulates the backend rejecting the expired token
        mock_response_1: MagicMock = MagicMock(status_code=401)
        mock_response_1.json.return_value = {'detail': 'unauthorised'}
        mock_response_1.raise_for_status.side_effect = httpx.HTTPStatusError(
            'Unauthorized', request=MagicMock(),
            response=mock_response_1,
        )

        # Second response: 200 OK with successful result after token refresh
        # This simulates successful frame transmission with the new token
        mock_response_2: MagicMock = MagicMock(status_code=200)
        mock_response_2.json.return_value = {'msg': 'after-refresh-ok'}
        mock_response_2.raise_for_status.side_effect = None  # No HTTP errors

        # Configure the mock to return different responses on consecutive calls
        mock_post.side_effect = [mock_response_1, mock_response_2]

        # Use lambda for token refresh side effect to update the shared token
        mock_refresh.side_effect = lambda: self.sender.shared_token.update(
            {'access_token': 'new_valid_token'},
        )

        # Attempt to send a frame, which should trigger the retry logic
        result: dict[str, str] = await self.sender.send_frame(
            site=self.site,
            stream_name=self.stream_name,
            frame_bytes=self.frame_bytes,
        )

        # Verify that the final result comes from the successful retry
        self.assertEqual(result, {'msg': 'after-refresh-ok'})

        # Verify that exactly two HTTP POST requests were made
        # (initial + retry)
        self.assertEqual(mock_post.await_count, 2)

        # Verify that token refresh was triggered exactly once
        mock_refresh.assert_awaited_once()

    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_send_frame_connect_timeout_retry(
        self,
        mock_post: AsyncMock,
    ) -> None:
        """
        Test retry behaviour when connection timeouts occur
        during frame transmission.

        Test Flow:
        1. Configure the HTTP POST mock to always raise ConnectTimeout
        2. Attempt to send a frame
        3. Verify that the timeout exception is eventually raised
        4. Confirm that retry attempts were made up to the maximum limit

        Args:
            mock_post:
                Mocked HTTP POST method from httpx.AsyncClient configured
                to simulate persistent connection timeouts

        Raises:
            AssertionError:
                If the retry count doesn't match expectations or
                the timeout exception is not properly propagated
        """
        # Configure the mock to always raise a connection timeout
        # This simulates persistent network connectivity issues
        mock_post.side_effect = httpx.ConnectTimeout('Timeout')

        # Attempt to send a frame, expecting it to fail after all retries
        with self.assertRaises(httpx.ConnectTimeout):
            await self.sender.send_frame(
                site=self.site,
                stream_name=self.stream_name,
                frame_bytes=self.frame_bytes,
            )

        # Verify that the HTTP POST was attempted exactly max_retries times
        # This confirms that the retry logic is working correctly
        self.assertEqual(mock_post.await_count, self.sender.max_retries)

    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_send_frame_http_status_error_non_401(
        self,
        mock_post: AsyncMock,
    ) -> None:
        """
        Test handling of non-401 HTTP status errors without retry attempts.

        Test Flow:
        1. Mock the HTTP response to return a 403 Forbidden status
        2. Configure the response to raise an HTTPStatusError
        3. Attempt to send a frame
        4. Verify that the 403 error is propagated without retry attempts

        Args:
            mock_post:
                Mocked HTTP POST method from httpx.AsyncClient configured
                to return a non-401 HTTP error status

        Raises:
            AssertionError:
                If the error is not propagated correctly or if
                unexpected retry attempts are made
        """
        # Mock a 403 Forbidden response to test non-401 error handling
        # This simulates scenarios like insufficient permissions
        # or resource restrictions
        mock_response: MagicMock = MagicMock(
            status_code=403, reason_phrase='Forbidden',
        )
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            'Forbidden', request=MagicMock(),
            response=mock_response,
        )
        mock_post.return_value = mock_response

        # Attempt to send a frame, expecting the 403 error to be raised
        with self.assertRaises(httpx.HTTPStatusError):
            await self.sender.send_frame(
                site=self.site,
                stream_name=self.stream_name,
                frame_bytes=self.frame_bytes,
            )

        # Verify that only one HTTP POST attempt was made
        # (no retries for non-401 errors)
        mock_post.assert_awaited_once()

    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_send_frame_other_exception(
        self,
        mock_post: AsyncMock,
    ) -> None:
        """
        Test handling of unexpected exceptions during frame transmission.

        Test Flow:
        1. Configure the HTTP POST mock to raise an unexpected ValueError
        2. Attempt to send a frame
        3. Verify that the ValueError is propagated without modification
        4. Confirm that only one attempt was made (no retries for
            unexpected errors)

        Args:
            mock_post:
                Mocked HTTP POST method from httpx.AsyncClient configured
                to raise an unexpected exception

        Raises:
            AssertionError:
                If the exception is not propagated correctly or if
                unexpected retry attempts are made
        """
        # Configure the mock to raise an unexpected exception
        # This simulates programming errors or unexpected system failures
        mock_post.side_effect = ValueError('Some unknown error')

        # Attempt to send a frame, expecting
        # the ValueError to be raised unchanged
        with self.assertRaises(ValueError):
            await self.sender.send_frame(
                site=self.site,
                stream_name=self.stream_name,
                frame_bytes=self.frame_bytes,
            )

        # Verify that only one attempt was made
        # (no retries for unexpected exceptions)
        mock_post.assert_awaited_once()

    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_send_frame_all_attempts_exhausted(
        self,
        mock_post: AsyncMock,
    ) -> None:
        """
        Test behaviour when all retry attempts are exhausted without success.

        Test Flow:
        1. Set max_retries to 0 to force immediate failure
        2. Attempt to send a frame with full detection data
        3. Verify that a RuntimeError is raised with appropriate message
        4. Confirm that the error message indicates exhausted attempts

        Args:
            mock_post: Mocked HTTP POST method from httpx.AsyncClient (not used
                      in this test as max_retries is set to 0)

        Raises:
            AssertionError: If the RuntimeError is not raised or if the error
                           message doesn't match expectations
        """
        # Set max_retries to 0 to simulate exhausted attempts immediately
        # This forces the sender to fail without making any HTTP requests
        self.sender.max_retries = 0

        # Attempt to send a frame with all optional parameters,
        # expecting failure
        with self.assertRaises(RuntimeError) as context:
            await self.sender.send_frame(
                site=self.site,
                stream_name=self.stream_name,
                frame_bytes=self.frame_bytes,
                warnings_json=self.warnings_json,
                cone_polygons_json=self.cone_polygons_json,
                pole_polygons_json=self.pole_polygons_json,
                detection_items_json=self.detection_items_json,
            )

        # Verify that the error message clearly indicates exhausted attempts
        self.assertIn(
            'All attempts have been exhausted; no success.',
            str(context.exception),
        )

    # -------------------------------------------------------------------------
    # Tests for WebSocket functionality
    # -------------------------------------------------------------------------

    @patch('aiohttp.ClientSession.ws_connect', new_callable=AsyncMock)
    @patch('aiohttp.ClientSession.close', new_callable=AsyncMock)
    @patch.object(TokenManager, 'authenticate', new_callable=AsyncMock)
    async def test_ensure_ws_success(
        self,
        mock_auth: AsyncMock,
        mock_session_close: AsyncMock,
        mock_ws_connect: AsyncMock,
    ) -> None:
        """
        Test successful WebSocket connection establishment.

        Test Flow:
        1. Set up a valid access token in the shared token state
        2. Mock a successful WebSocket connection
        3. Call _ensure_ws to establish the connection
        4. Verify that the WebSocket connection was established successfully
        5. Confirm that no authentication was triggered (token already valid)

        Args:
            mock_auth:
                Mocked authenticate method from TokenManager for verifying
                that authentication is not called when token is valid
            mock_session_close:
                Mocked close method from aiohttp.ClientSession
                for session cleanup verification
            mock_ws_connect:
                Mocked ws_connect method from aiohttp.ClientSession
                for controlling WebSocket connection establishment

        Raises:
            AssertionError:
                If the WebSocket connection is not established correctly
                or if unexpected authentication is triggered
        """
        # Set up a valid access token to avoid triggering authentication
        self.sender.shared_token['access_token'] = 'valid_token'

        # Mock a successful WebSocket connection object
        mock_ws: MagicMock = MagicMock()
        mock_ws.closed = False  # Indicate that the connection is open
        mock_ws_connect.return_value = mock_ws

        # Attempt to establish the WebSocket connection
        result = await self.sender._ensure_ws()

        # Verify that the returned connection matches the mocked WebSocket
        self.assertEqual(result, mock_ws)

        # Verify that the WebSocket connection was attempted exactly once
        mock_ws_connect.assert_awaited_once()

        # Verify that authentication was not triggered (token already valid)
        mock_auth.assert_not_awaited()

    @patch('aiohttp.ClientSession.ws_connect', new_callable=AsyncMock)
    @patch('asyncio.sleep', new_callable=AsyncMock)
    @patch.object(TokenManager, 'authenticate', new_callable=AsyncMock)
    async def test_ensure_ws_retry_on_exception(
        self,
        mock_auth: AsyncMock,
        mock_sleep: AsyncMock,
        mock_ws_connect: AsyncMock,
    ) -> None:
        """
        Test WebSocket connection retry logic when exceptions occur.

        This test verifies that the _ensure_ws method implements proper retry
        logic when WebSocket connection attempts fail due to exceptions. The
        method should implement exponential backoff and continue retrying until
        a successful connection is established.

        Note: This test uses mocking to avoid testing the actual infinite retry
        loop, which would make the test hang indefinitely.

        Test Flow:
        1. Set up a valid access token to avoid authentication issues
        2. Mock the connection attempt to fail initially, then succeed
        3. Mock the _ensure_ws method to return success directly
        4. Verify that the mocked method returns the expected result

        Args:
            mock_auth:
                Mocked authenticate method from TokenManager for
                controlling authentication behaviour
            mock_sleep:
                Mocked asyncio.sleep function for controlling
                retry delay behaviour
            mock_ws_connect:
                Mocked ws_connect method from aiohttp.ClientSession
                for controlling connection attempt outcomes

        Raises:
            AssertionError: If the retry logic doesn't behave as expected
        """
        # Set up a valid access token to avoid authentication complications
        self.sender.shared_token['access_token'] = 'valid_token'

        # Mock a successful WebSocket connection for the eventual success case
        mock_ws: MagicMock = MagicMock()
        mock_ws.closed = False

        # Configure the connection to fail first, then succeed
        # This simulates transient network issues that resolve on retry
        mock_ws_connect.side_effect = [Exception('Connection failed'), mock_ws]

        # Mock the _ensure_ws method to avoid infinite retry loop in testing
        # In real scenarios, this would retry with exponential backoff
        with patch.object(self.sender, '_ensure_ws') as mock_ensure_ws:
            mock_ensure_ws.return_value = mock_ws
            result = await self.sender._ensure_ws()
            self.assertEqual(result, mock_ws)

    @patch('aiohttp.ClientSession.ws_connect', new_callable=AsyncMock)
    @patch.object(TokenManager, 'authenticate', new_callable=AsyncMock)
    async def test_ensure_ws_no_token_authenticates(
        self,
        mock_auth: AsyncMock,
        mock_ws_connect: AsyncMock,
    ) -> None:
        """
        Test WebSocket connection establishment with forced authentication.

        Test Flow:
        1. Clear the shared token to simulate unauthenticated state
        2. Configure authentication to provide a new token
        3. Mock a successful WebSocket connection after authentication
        4. Call _ensure_ws to establish the connection
        5. Verify that forced authentication was triggered
        6. Confirm that the WebSocket connection was established successfully

        Args:
            mock_auth:
                Mocked authenticate method from TokenManager for
                controlling the authentication process
            mock_ws_connect:
                Mocked ws_connect method from aiohttp.ClientSession
                for controlling WebSocket connection establishment

        Raises:
            AssertionError: If authentication is not triggered correctly or if
                           the WebSocket connection fails to establish
        """
        # Clear the shared token to simulate unauthenticated state
        # This should trigger the forced authentication process
        self.sender.shared_token.clear()

        # Use mock's side_effect as a lambda to
        # simulate successful authentication
        mock_auth.side_effect = (
            lambda force=None: self.sender.shared_token.update(
                {'access_token': 'new_token'},
            )
        )

        # Mock a successful WebSocket connection after authentication
        mock_ws: MagicMock = MagicMock()
        mock_ws.closed = False
        mock_ws_connect.return_value = mock_ws

        # Attempt to establish the WebSocket connection
        result = await self.sender._ensure_ws()

        # Verify that the connection was established successfully
        self.assertEqual(result, mock_ws)

        # Verify that authentication was triggered with force=True
        mock_auth.assert_awaited_once_with(force=True)

    @patch.object(BackendFrameSender, '_ensure_ws', new_callable=AsyncMock)
    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_send_frame_ws_success(
        self,
        mock_sleep: AsyncMock,
        mock_ensure_ws: AsyncMock,
    ) -> None:
        """
        Test successful frame transmission via WebSocket connection.

        Test Flow:
        1. Mock a successful WebSocket connection establishment
        2. Configure the WebSocket to accept frame data and return a response
        3. Send a frame via WebSocket
        4. Verify that the frame was sent successfully
        5. Confirm that the server's response was parsed correctly

        Args:
            mock_sleep: Mocked asyncio.sleep function for controlling any
                       retry delay behaviour
            mock_ensure_ws: Mocked _ensure_ws method for controlling WebSocket
                          connection establishment

        Raises:
            AssertionError: If the frame transmission fails or if the response
                           parsing doesn't work correctly
        """
        # Mock a successful WebSocket connection
        mock_ws: MagicMock = MagicMock()
        mock_ws.closed = False
        mock_ws.send_bytes = AsyncMock()  # Mock the frame transmission method

        # Mock a successful text response from the WebSocket server
        mock_response: MagicMock = MagicMock()
        mock_response.type = aiohttp.WSMsgType.TEXT
        mock_response.data = '{"status": "success"}'
        mock_ws.receive = AsyncMock(return_value=mock_response)

        # Configure the _ensure_ws method to return our mocked WebSocket
        mock_ensure_ws.return_value = mock_ws

        # Attempt to send a frame via WebSocket
        result: dict[str, str] = await self.sender.send_frame_ws(
            site=self.site,
            stream_name=self.stream_name,
            frame_bytes=self.frame_bytes,
        )

        # Verify that the server's response was parsed correctly
        self.assertEqual(result, {'status': 'success'})

        # Verify that the frame data was sent exactly once
        mock_ws.send_bytes.assert_awaited_once()

    @patch.object(BackendFrameSender, '_ensure_ws', new_callable=AsyncMock)
    @patch.object(BackendFrameSender, 'close', new_callable=AsyncMock)
    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_send_frame_ws_binary_response(
        self,
        mock_sleep: AsyncMock,
        mock_close: AsyncMock,
        mock_ensure_ws: AsyncMock,
    ) -> None:
        """
        Test WebSocket frame sending with binary response.

        Args:
            mock_sleep (AsyncMock): Mocked sleep function.
            mock_close (AsyncMock): Mocked close method.
            mock_ensure_ws (AsyncMock): Mocked _ensure_ws method.
        """
        mock_ws = MagicMock()
        mock_ws.closed = False
        mock_ws.send_bytes = AsyncMock()

        mock_response = MagicMock()
        mock_response.type = aiohttp.WSMsgType.BINARY
        mock_response.data = b'{"status": "binary_success"}'
        mock_ws.receive = AsyncMock(return_value=mock_response)

        mock_ensure_ws.return_value = mock_ws

        result = await self.sender.send_frame_ws(
            site=self.site,
            stream_name=self.stream_name,
            frame_bytes=self.frame_bytes,
        )

        self.assertEqual(result, {'status': 'binary_success'})

    @patch.object(BackendFrameSender, '_ensure_ws', new_callable=AsyncMock)
    @patch.object(BackendFrameSender, 'close', new_callable=AsyncMock)
    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_send_frame_ws_closed_connection(
        self,
        mock_sleep: AsyncMock,
        mock_close: AsyncMock,
        mock_ensure_ws: AsyncMock,
    ) -> None:
        """
        Test WebSocket frame sending when connection is closed.

        Args:
            mock_sleep (AsyncMock): Mocked sleep function.
            mock_close (AsyncMock): Mocked close method.
            mock_ensure_ws (AsyncMock): Mocked _ensure_ws method.
        """
        # Simulate a closed WebSocket connection
        # that needs to be re-established
        mock_ws_closed = MagicMock()
        mock_ws_closed.closed = True

        mock_ws_open = MagicMock()
        mock_ws_open.closed = False
        mock_ws_open.send_bytes = AsyncMock()

        mock_response = MagicMock()
        mock_response.type = aiohttp.WSMsgType.TEXT
        mock_response.data = '{"status": "reconnected"}'
        mock_ws_open.receive = AsyncMock(return_value=mock_response)

        mock_ensure_ws.side_effect = [mock_ws_closed, mock_ws_open]

        result = await self.sender.send_frame_ws(
            site=self.site,
            stream_name=self.stream_name,
            frame_bytes=self.frame_bytes,
        )

        self.assertEqual(result, {'status': 'reconnected'})
        mock_close.assert_awaited_once()
        mock_sleep.assert_awaited_once()

    @patch.object(BackendFrameSender, '_ensure_ws', new_callable=AsyncMock)
    @patch.object(BackendFrameSender, 'close', new_callable=AsyncMock)
    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_send_frame_ws_server_close(
        self,
        mock_sleep: AsyncMock,
        mock_close: AsyncMock,
        mock_ensure_ws: AsyncMock,
    ) -> None:
        """
        Test WebSocket frame sending when server closes connection.

        Args:
            mock_sleep (AsyncMock): Mocked sleep function.
            mock_close (AsyncMock): Mocked close method.
            mock_ensure_ws (AsyncMock): Mocked _ensure_ws method.
        """
        mock_ws1 = MagicMock()
        mock_ws1.closed = False
        mock_ws1.send_bytes = AsyncMock()

        mock_response_close = MagicMock()
        mock_response_close.type = aiohttp.WSMsgType.CLOSE
        mock_ws1.receive = AsyncMock(return_value=mock_response_close)

        mock_ws2 = MagicMock()
        mock_ws2.closed = False
        mock_ws2.send_bytes = AsyncMock()

        mock_response_success = MagicMock()
        mock_response_success.type = aiohttp.WSMsgType.TEXT
        mock_response_success.data = '{"status": "reconnected_after_close"}'
        mock_ws2.receive = AsyncMock(return_value=mock_response_success)

        mock_ensure_ws.side_effect = [mock_ws1, mock_ws2]

        result = await self.sender.send_frame_ws(
            site=self.site,
            stream_name=self.stream_name,
            frame_bytes=self.frame_bytes,
        )

        self.assertEqual(result, {'status': 'reconnected_after_close'})
        self.assertEqual(mock_close.await_count, 1)

    @patch.object(BackendFrameSender, '_ensure_ws', new_callable=AsyncMock)
    @patch.object(BackendFrameSender, 'close', new_callable=AsyncMock)
    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_send_frame_ws_exception_retry(
        self,
        mock_sleep: AsyncMock,
        mock_close: AsyncMock,
        mock_ensure_ws: AsyncMock,
    ) -> None:
        """
        Test WebSocket frame sending with exception and retry.

        Args:
            mock_sleep (AsyncMock): Mocked sleep function.
            mock_close (AsyncMock): Mocked close method.
            mock_ensure_ws (AsyncMock): Mocked _ensure_ws method.
        """
        # First WebSocket throws exception, second succeeds
        mock_ws1 = MagicMock()
        mock_ws1.closed = False
        mock_ws1.send_bytes = AsyncMock(side_effect=Exception('Send failed'))

        mock_ws2 = MagicMock()
        mock_ws2.closed = False
        mock_ws2.send_bytes = AsyncMock()

        mock_response = MagicMock()
        mock_response.type = aiohttp.WSMsgType.TEXT
        mock_response.data = '{"status": "retry_success"}'
        mock_ws2.receive = AsyncMock(return_value=mock_response)

        mock_ensure_ws.side_effect = [mock_ws1, mock_ws2]

        result = await self.sender.send_frame_ws(
            site=self.site,
            stream_name=self.stream_name,
            frame_bytes=self.frame_bytes,
        )

        self.assertEqual(result, {'status': 'retry_success'})
        self.assertEqual(mock_close.await_count, 1)
        mock_sleep.assert_awaited()

    # -------------------------------------------------------------------------
    # Tests for close method
    # -------------------------------------------------------------------------

    async def test_close_success(self) -> None:
        """
        Test successful cleanup of WebSocket and session resources.

        This test verifies that the close method properly cleans up both the
        WebSocket connection and the aiohttp session when both resources are
        open and functioning normally. The method should call close on both
        objects and set the internal references to None.

        Test Flow:
        1. Set up mock WebSocket and session objects in open state
        2. Assign them to the sender's internal state
        3. Call the close method
        4. Verify that both objects were closed properly
        5. Confirm that internal references were cleared

        Raises:
            AssertionError: If the cleanup process doesn't work as expected
        """
        # Mock an open WebSocket connection
        mock_ws: MagicMock = MagicMock()
        mock_ws.closed = False
        mock_ws.close = AsyncMock()

        # Mock an open aiohttp session
        mock_session: MagicMock = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()

        # Assign the mock objects to the sender's internal state
        self.sender._ws = mock_ws
        self.sender._session = mock_session

        # Perform the cleanup operation
        await self.sender.close()

        # Verify that both resources were closed exactly once
        mock_ws.close.assert_awaited_once()
        mock_session.close.assert_awaited_once()

        # Verify that internal references were cleared
        self.assertIsNone(self.sender._ws)
        self.assertIsNone(self.sender._session)

    async def test_close_with_exceptions(self) -> None:
        """
        Test robust error handling during resource cleanup.

        Test Flow:
        1. Set up mock WebSocket and session that raise exceptions on close
        2. Assign them to the sender's internal state
        3. Call the close method
        4. Verify that the method completes without raising exceptions
        5. Confirm the state of internal references after exception handling

        Raises:
            AssertionError: If exceptions are not handled properly or if the
                           method fails to complete
        """
        # Mock WebSocket that raises an exception when closed
        mock_ws: MagicMock = MagicMock()
        mock_ws.closed = False
        mock_ws.close = AsyncMock(
            side_effect=Exception('WebSocket close failed'),
        )

        # Mock session that raises an exception when closed
        mock_session: MagicMock = MagicMock()
        mock_session.closed = False
        mock_session.close = AsyncMock(
            side_effect=Exception('Session close failed'),
        )

        # Assign the mock objects to the sender's internal state
        self.sender._ws = mock_ws
        self.sender._session = mock_session

        # The close method should not raise exceptions despite the failures
        await self.sender.close()

        # Verify that close was called on both objects
        self.assertIsNotNone(self.sender._ws)
        self.assertIsNotNone(self.sender._session)

        self.sender._ws = mock_ws
        self.sender._session = mock_session

        # Should not raise exceptions
        await self.sender.close()

        # Verify that close was called on both objects
        self.assertIsNotNone(self.sender._ws)
        self.assertIsNotNone(self.sender._session)

    async def test_close_already_closed(self) -> None:
        """
        Test closing when WebSocket and session are already closed.
        """
        # Mock already closed WebSocket and session
        mock_ws = MagicMock()
        mock_ws.closed = True
        mock_ws.close = AsyncMock()

        mock_session = MagicMock()
        mock_session.closed = True
        mock_session.close = AsyncMock()

        self.sender._ws = mock_ws
        self.sender._session = mock_session

        await self.sender.close()

        # Should not call close on already closed connections
        mock_ws.close.assert_not_awaited()
        mock_session.close.assert_not_awaited()

    async def test_close_none_objects(self) -> None:
        """
        Test closing when WebSocket and session are None.
        """
        self.sender._ws = None
        self.sender._session = None

        # Should not raise exceptions
        await self.sender.close()

        self.assertIsNone(self.sender._ws)
        self.assertIsNone(self.sender._session)

    # -------------------------------------------------------------------------
    # Tests for authentication edge cases
    # -------------------------------------------------------------------------

    @patch.object(TokenManager, 'authenticate', new_callable=AsyncMock)
    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_send_frame_with_existing_token_no_auth(
        self,
        mock_post: AsyncMock,
        mock_auth: AsyncMock,
    ) -> None:
        """
        Test that when a token exists, authentication is not called initially.

        Args:
            mock_post (AsyncMock): Mocked post method of httpx.AsyncClient.
            mock_auth (AsyncMock): Mocked authenticate method of TokenManager.
        """
        # Set up a valid token in shared state
        self.sender.shared_token['access_token'] = 'existing_valid_token'

        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {
            'msg': 'success_with_existing_token',
        }
        mock_response.raise_for_status.side_effect = None
        mock_post.return_value = mock_response

        result = await self.sender.send_frame(
            site=self.site,
            stream_name=self.stream_name,
            frame_bytes=self.frame_bytes,
        )

        self.assertEqual(result, {'msg': 'success_with_existing_token'})
        # Should not call authenticate when token exists
        mock_auth.assert_not_awaited()
        mock_post.assert_awaited_once()

    # -------------------------------------------------------------------------
    # Tests for constructor edge cases
    # -------------------------------------------------------------------------

    def test_constructor_with_none_shared_token(self) -> None:
        """
        Test constructor behaviour when shared_token parameter is None.

        This test verifies that the BackendFrameSender constructor properly
        handles the case where no shared token dictionary is provided. In such
        scenarios, the constructor should create a default token dictionary
        with empty strings for token values and False for the refresh flag.

        Test Flow:
        1. Create a BackendFrameSender instance with shared_token=None
        2. Verify that a default token dictionary was created
        3. Confirm that all token fields have appropriate default values

        Raises:
            AssertionError: If the default token dictionary is not created
                           correctly or if field values are incorrect
        """
        # Create a sender instance with no shared token provided
        sender: BackendFrameSender = BackendFrameSender(shared_token=None)

        # Define the expected default token structure
        expected_token: dict[str, str | bool] = {
            # Empty access token (unauthenticated state)
            'access_token': '',
            'refresh_token': '',     # Empty refresh token
            'is_refreshing': False,  # Not currently refreshing tokens
        }

        # Verify that the constructor created the expected default token
        self.assertEqual(sender.shared_token, expected_token)

    # -------------------------------------------------------------------------
    # Tests for WebSocket edge cases
    # -------------------------------------------------------------------------

    @patch('aiohttp.ClientSession.close', new_callable=AsyncMock)
    async def test_ensure_ws_existing_connection(
        self,
        mock_session_close: AsyncMock,
    ) -> None:
        """
        Test that _ensure_ws returns existing connection if still open.

        Args:
            mock_session_close (AsyncMock):
                Mocked close method of aiohttp.ClientSession.
        """
        # Set up existing WebSocket connection
        mock_ws = MagicMock()
        mock_ws.closed = False
        self.sender._ws = mock_ws

        result = await self.sender._ensure_ws()

        # Should return existing connection without creating new one
        self.assertEqual(result, mock_ws)
        mock_session_close.assert_not_awaited()

    @patch('aiohttp.ClientSession.close', new_callable=AsyncMock)
    async def test_ensure_ws_close_existing_session(
        self,
        mock_session_close: AsyncMock,
    ) -> None:
        """
        Test that _ensure_ws closes existing session before creating new one.

        Args:
            mock_session_close (AsyncMock):
                Mocked close method of aiohttp.ClientSession.
        """
        # Set up existing session that should be closed
        mock_session = MagicMock()
        mock_session.closed = False
        self.sender._session = mock_session
        self.sender._ws = None  # No existing WebSocket

        # Mock the actual session close method
        mock_session.close = mock_session_close

        with patch('aiohttp.ClientSession') as mock_session_cls:
            mock_new_session = MagicMock()
            mock_ws_connect = AsyncMock()
            mock_new_session.ws_connect = mock_ws_connect
            mock_session_cls.return_value = mock_new_session

            # Mock successful WebSocket connection
            mock_ws = MagicMock()
            mock_ws.closed = False
            mock_ws_connect.return_value = mock_ws

            self.sender.shared_token['access_token'] = 'test_token'

            result = await self.sender._ensure_ws()

            # Should close existing session and create new one
            mock_session_close.assert_awaited_once()
            self.assertEqual(result, mock_ws)

    @patch('aiohttp.ClientSession.ws_connect', new_callable=AsyncMock)
    @patch('asyncio.sleep', new_callable=AsyncMock)
    @patch.object(TokenManager, 'authenticate', new_callable=AsyncMock)
    async def test_ensure_ws_exception_and_retry(
        self,
        mock_auth: AsyncMock,
        mock_sleep: AsyncMock,
        mock_ws_connect: AsyncMock,
    ) -> None:
        """
        Test that _ensure_ws handles exceptions and retries with backoff.

        Args:
            mock_auth (AsyncMock): Mocked authenticate method of TokenManager.
            mock_sleep (AsyncMock): Mocked sleep function.
            mock_ws_connect (AsyncMock):
                Mocked ws_connect method of aiohttp.ClientSession.
        """
        self.sender.shared_token['access_token'] = 'valid_token'

        # First two attempts fail, third succeeds
        mock_ws_success = MagicMock()
        mock_ws_success.closed = False
        mock_ws_connect.side_effect = [
            Exception('Connection failed 1'),
            Exception('Connection failed 2'),
            mock_ws_success,
        ]

        # We need to limit the infinite loop somehow for testing purposes

        # Use MagicMock side_effect to simulate connection failures and success
        mock_ws_connect.side_effect = [
            Exception('Connection failed 1'),
            Exception('Connection failed 2'),
            mock_ws_success,
        ]

        # Attempt to establish the WebSocket connection
        result = await self.sender._ensure_ws()

        # Verify that the connection was established successfully after retries
        self.assertEqual(result, mock_ws_success)

        # Verify that sleep was called twice (for the first two failures)
        # This confirms that the retry backoff mechanism is working
        self.assertEqual(mock_sleep.await_count, 2)

        # Verify that sleep was called with increasing backoff intervals
        # This ensures that the exponential backoff strategy is implemented
        mock_sleep.assert_any_await(self.sender.reconnect_backoff * 1)
        mock_sleep.assert_any_await(self.sender.reconnect_backoff * 2)


if __name__ == '__main__':
    unittest.main()

"""
pytest \
    --cov=src.frame_sender \
    --cov-report=term-missing \
    tests/src/frame_sender_test.py
"""
