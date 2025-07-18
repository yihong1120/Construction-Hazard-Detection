from __future__ import annotations

import asyncio
import json
import logging

import aiohttp
import httpx

from examples.streaming_web.backend.redis_service import DELIMITER
from src.utils import TokenManager


class BackendFrameSender:
    """
    A class to send frames and metadata to a backend server via HTTP or
    WebSocket.

    Attributes:
        base_url (str): The base URL of the backend API.
        shared_token (dict[str, str | bool]):
            Shared token state for authentication.
        max_retries (int):
            Maximum number of retries for HTTP requests.
        timeout (int):
            Timeout in seconds for HTTP requests.
        reconnect_backoff (float):
            Backoff multiplier for reconnection attempts.
        _ws (aiohttp.ClientWebSocketResponse | None):
            WebSocket connection instance.
        _session (aiohttp.ClientSession | None):
            aiohttp session instance.
        token_manager (TokenManager):
            Token manager for handling authentication and refresh.
    """

    def __init__(
        self,
        api_url: str = 'http://127.0.0.1:8800',
        max_retries: int = 3,
        timeout: int = 10,
        reconnect_backoff: float = 1.5,
        shared_token: dict[str, str | bool] | None = None,
        shared_lock: object | None = None,
    ) -> None:
        """
        Initialise the BackendFrameSender.

        Args:
            api_url (str): The base URL of the backend API.
            max_retries (int): Maximum number of retries for HTTP requests.
            timeout (int): Timeout in seconds for HTTP requests.
            reconnect_backoff (float):
                Backoff multiplier for reconnection attempts.
            shared_token (dict[str, str | bool] | None):
                Optional shared token state for authentication and refresh.
            shared_lock (object | None):
                Optional lock for synchronising token access (for testing).
        """
        self.base_url: str = api_url.rstrip(
            '/',
        )  # Remove trailing slash for consistency
        # Shared token state for authentication and refresh
        if shared_token is not None:
            self.shared_token = shared_token
        else:
            self.shared_token = {
                'access_token': '',
                'refresh_token': '',
                'is_refreshing': False,
            }
        self.max_retries: int = max_retries  # Maximum retry attempts for HTTP
        self.timeout: int = timeout  # Timeout in seconds for HTTP/WebSocket
        # Backoff multiplier for reconnect
        self.reconnect_backoff: float = reconnect_backoff
        self._ws: aiohttp.ClientWebSocketResponse | None = None  # WebSocket
        # connection
        self._session: aiohttp.ClientSession | None = None  # aiohttp session
        # Suppress httpx debug logging for cleaner output
        logging.getLogger('httpx').setLevel(logging.WARNING)

        # Create a TokenManager instance using this object's token state
        self.token_manager: TokenManager = TokenManager(
            shared_token=self.shared_token,
        )

    async def send_frame(
        self: BackendFrameSender,
        site: str,
        stream_name: str,
        frame_bytes: bytes,
        warnings_json: str = '',
        cone_polygons_json: str = '',
        pole_polygons_json: str = '',
        detection_items_json: str = '',
        width: int = 0,
        height: int = 0,
    ) -> dict:
        """
        Send a frame and metadata to the backend via HTTP POST.

        Args:
            site (str): The site/label name.
            stream_name (str): The stream key or camera identifier.
            frame_bytes (bytes): The raw image bytes to upload.
            warnings_json (str, optional):
                JSON string of warnings. Defaults to ''.
            cone_polygons_json (str, optional):
                JSON string of cone polygons. Defaults to ''.
            pole_polygons_json (str, optional):
                JSON string of pole polygons. Defaults to ''.
            detection_items_json (str, optional):
                JSON string of detection items. Defaults to ''.
            width (int, optional): Frame width. Defaults to 0.
            height (int, optional): Frame height. Defaults to 0.

        Returns:
            dict: The JSON response from the backend.

        Raises:
            RuntimeError: If all retry attempts are exhausted.
            httpx.HTTPStatusError: For non-401 HTTP errors.
            Exception: For other unexpected errors.
        """
        # If no token is present, authenticate first
        if not self.shared_token or not self.shared_token.get('access_token'):
            # Force authentication if token is missing
            await self.token_manager.authenticate(force=True)

        # Retrieve access token from shared state
        access_token: str = str(self.shared_token.get('access_token', ''))
        headers: dict[str, str] = {}

        if not access_token:
            # If no access token exists, authenticate
            await self.token_manager.authenticate()
        else:
            # Attach the access token to the request header
            headers['Authorization'] = (
                f"Bearer {access_token}"
            )

        # Prepare multipart/form-data for the image and metadata
        files: dict[str, tuple[str, bytes, str]] = {
            'file': (
                'frame.jpg',
                frame_bytes,
                'image/jpeg',
            ),
        }
        # Prepare the POST data payload
        data: dict[str, str | int] = {
            'label': site,
            'key': stream_name,
            'warnings_json': warnings_json,
            'cone_polygons_json': cone_polygons_json,
            'pole_polygons_json': pole_polygons_json,
            'detection_items_json': detection_items_json,
            'width': width,
            'height': height,
        }

        # Attempt to send the request with retries
        for attempt in range(self.max_retries):
            try:
                # Use httpx.AsyncClient for asynchronous HTTP requests
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        self.base_url + '/frames',
                        data=data,
                        files=files,
                        headers=headers,
                    )
                    # Raise for HTTP 4xx/5xx except 401
                    resp.raise_for_status()
                    return resp.json()

            except httpx.ConnectTimeout:
                # Handle connection timeout and retry
                print(
                    f"Attempt {attempt + 1} - Connection timeout, "
                    'retrying...',
                )
                if attempt == self.max_retries - 1:
                    raise  # Raise if last attempt

            except httpx.HTTPStatusError as exc:
                # Handle 401 (unauthorised) by refreshing token and retrying
                if exc.response.status_code == 401:
                    print(
                        'Unauthorized request: Token might be invalid or '
                        'expired.',
                    )
                    await self.token_manager.refresh_token()
                    # Retry with refreshed token
                    return await self.send_frame(
                        site,
                        stream_name,
                        frame_bytes,
                        warnings_json,
                        cone_polygons_json,
                        pole_polygons_json,
                        detection_items_json,
                        width,
                        height,
                    )
                raise  # For other HTTP errors, raise the exception

            except Exception as e:
                # Handle all other exceptions
                print(f"An error occurred: {e}")
                raise

        # If all attempts fail, raise an error
        raise RuntimeError(
            'All attempts have been exhausted; no success.',
        )

    async def _ensure_ws(
            self: BackendFrameSender,
    ) -> aiohttp.ClientWebSocketResponse:
        """
        Ensure a valid WebSocket connection is available,
        reconnecting if necessary.

        Returns:
            aiohttp.ClientWebSocketResponse: The active WebSocket connection.
        """
        # If already connected and not closed, return the connection
        if self._ws and not self._ws.closed:
            return self._ws
        # Close previous session if open
        if self._session and not self._session.closed:
            await self._session.close()
        # Create a new aiohttp session for WebSocket
        self._session = aiohttp.ClientSession()
        attempt: int = 1
        while True:
            try:
                # Ensure access token is available
                if not self.shared_token.get('access_token'):
                    await self.token_manager.authenticate(force=True)
                headers: dict[str, str] = {
                    'Authorization': (
                        f"Bearer {self.shared_token['access_token']}"
                    ),
                }
                ws_url: str = self.base_url + '/ws/frames'
                # Attempt to establish WebSocket connection
                self._ws = await self._session.ws_connect(
                    ws_url,
                    headers=headers,
                    timeout=self.timeout,
                )
                return self._ws
            except Exception as e:
                # Print error and retry with backoff
                print(
                    f"[WebSocket] Connect fail (attempt {attempt}): {e}",
                )
                await asyncio.sleep(self.reconnect_backoff * attempt)
                attempt += 1

    async def send_frame_ws(
        self: BackendFrameSender,
        site: str,
        stream_name: str,
        frame_bytes: bytes,
        warnings_json: str = '',
        cone_polygons_json: str = '',
        pole_polygons_json: str = '',
        detection_items_json: str = '',
        width: int = 0,
        height: int = 0,
    ) -> dict:
        """
        Send a frame and metadata to the backend via WebSocket.
        Automatically reconnects if the connection is lost.

        Args:
            site (str): The site/label name.
            stream_name (str): The stream key or camera identifier.
            frame_bytes (bytes): The raw image bytes to upload.
            warnings_json (str, optional):
                JSON string of warnings. Defaults to ''.
            cone_polygons_json (str, optional):
            JSON string of cone polygons. Defaults to ''.
            pole_polygons_json (str, optional):
                JSON string of pole polygons. Defaults to ''.
            detection_items_json (str, optional):
                JSON string of detection items. Defaults to ''.
            width (int, optional): Frame width. Defaults to 0.
            height (int, optional): Frame height. Defaults to 0.

        Returns:
            dict: The JSON response from the backend.
        """
        # Prepare header and message for WebSocket transmission
        header: dict[str, str | int] = {
            'label': site,
            'key': stream_name,
            'warnings_json': warnings_json,
            'cone_polygons_json': cone_polygons_json,
            'pole_polygons_json': pole_polygons_json,
            'detection_items_json': detection_items_json,
            'width': width,
            'height': height,
        }
        # Encode header and concatenate with frame bytes using DELIMITER
        header_bytes: bytes = json.dumps(header).encode('utf-8')
        message: bytes = header_bytes + DELIMITER + frame_bytes
        attempt: int = 1
        while True:
            try:
                ws: aiohttp.ClientWebSocketResponse = await self._ensure_ws()
                # Check WebSocket state before sending
                if ws.closed:
                    print(
                        '[WebSocket] Already closed before send, '
                        'reconnecting...',
                    )
                    await self.close()
                    await asyncio.sleep(
                        self.reconnect_backoff * attempt,
                    )
                    attempt += 1
                    continue
                # Send the frame and metadata as a binary message
                await ws.send_bytes(message)
                resp = await ws.receive()
                # Handle different WebSocket message types
                if resp.type == aiohttp.WSMsgType.TEXT:
                    return json.loads(resp.data)
                elif resp.type == aiohttp.WSMsgType.BINARY:
                    return json.loads(resp.data.decode('utf-8'))
                elif resp.type == aiohttp.WSMsgType.CLOSE:
                    print(
                        '[WebSocket] Closed by server, reconnecting...',
                    )
                    await self.close()
                    await asyncio.sleep(
                        self.reconnect_backoff * attempt,
                    )
                    attempt += 1
                    continue  # Reconnect
            except Exception as e:
                # Print error and retry with backoff
                print(
                    f'[WebSocket] Send fail (attempt {attempt}): {e}, '
                    'reconnecting...',
                )
                await self.close()
                await asyncio.sleep(
                    self.reconnect_backoff * attempt,
                )
                attempt += 1
                continue  # Reconnect
        await self.close()

    async def close(self: BackendFrameSender) -> None:
        """
        Close the WebSocket and aiohttp session to prevent resource leaks
        and thread issues.
        """
        try:
            # Close the WebSocket connection if open
            if self._ws is not None and not self._ws.closed:
                await self._ws.close()
            self._ws = None
        except Exception as e:
            print(f"Error closing WebSocket: {e}")
        try:
            # Close the aiohttp session if open
            if self._session is not None and not self._session.closed:
                await self._session.close()
            self._session = None
        except Exception as e:
            print(f"Error closing session: {e}")
