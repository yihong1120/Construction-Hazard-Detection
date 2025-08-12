from __future__ import annotations

import asyncio
import json
import logging
import os

import aiohttp
import httpx
import numpy as np
from dotenv import load_dotenv

from examples.streaming_web.backend.redis_service import DELIMITER
from src.utils import TokenManager

# Load environment variables from .env file
load_dotenv()


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
        api_url: str | None = None,
        max_retries: int = 3,
        timeout: int = 10,
        reconnect_backoff: float = 1.5,
        shared_token: dict[str, str | bool] | None = None,
    ) -> None:
        """
        Initialise the BackendFrameSender.

        Args:
            api_url (str | None): The base URL of the backend API. If None,
                uses environment variable.
            max_retries (int): Maximum number of retries for HTTP requests.
            timeout (int): Timeout in seconds for HTTP requests.
            reconnect_backoff (float):
                Backoff multiplier for reconnection attempts.
            shared_token (dict[str, str | bool] | None):
                Optional shared token state for authentication and refresh.
        """
        # Set the base URL from the provided api_url or environment variable
        if api_url is None:
            api_url = os.getenv('STREAMING_API_URL', 'http://127.0.0.1:8800')

        # Remove trailing slash for consistency
        self.base_url: str = api_url.rstrip(
            '/',
        )

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

    async def send_optimized_frame(
        self: BackendFrameSender,
        frame: np.ndarray,
        site: str,
        stream_name: str,
        encoding_format: str = 'jpeg',
        jpeg_quality: int = 85,
        use_websocket: bool = True,
        warnings_json: str = '',
        cone_polygons_json: str = '',
        pole_polygons_json: str = '',
        detection_items_json: str = '',
    ) -> dict:
        """
        Send frame with optimized encoding to backend service.

        Args:
            frame (np.ndarray): Frame to send
            site (str): Site identifier
            stream_name (str): Stream identifier
            encoding_format (str): Image encoding format ('jpeg' or 'png')
            jpeg_quality (int): JPEG quality (1-100) when using JPEG format
            use_websocket (bool): Whether to use WebSocket instead of HTTP
            warnings_json (str): JSON string of warnings
            cone_polygons_json (str): JSON string of cone polygons
            pole_polygons_json (str): JSON string of pole polygons
            detection_items_json (str): JSON string of detection items

        Returns:
            dict: Response from backend
        """
        try:
            # Encode frame based on specified format for optimal compression
            if encoding_format.lower() == 'jpeg':
                from .utils import Utils
                encoded_frame = Utils.encode_frame(frame, 'jpeg', jpeg_quality)
            else:
                from .utils import Utils
                encoded_frame = Utils.encode_frame(frame, 'png')

            if not encoded_frame:
                logging.error('Failed to encode frame')
                return {'success': False, 'error': 'Failed to encode frame'}

            height, width = frame.shape[:2]

            # Choose transmission method based on preference
            if use_websocket:
                try:
                    return await self.send_frame_ws(
                        site=site,
                        stream_name=stream_name,
                        frame_bytes=encoded_frame,
                        warnings_json=warnings_json,
                        cone_polygons_json=cone_polygons_json,
                        pole_polygons_json=pole_polygons_json,
                        detection_items_json=detection_items_json,
                        width=width,
                        height=height,
                    )
                except Exception as ws_error:
                    # Log WebSocket errors and fall back to HTTP
                    logging.warning(
                        f"WebSocket failed, falling back to HTTP: {ws_error}",
                    )
                    return await self.send_frame(
                        site=site,
                        stream_name=stream_name,
                        frame_bytes=encoded_frame,
                        warnings_json=warnings_json,
                        cone_polygons_json=cone_polygons_json,
                        pole_polygons_json=pole_polygons_json,
                        detection_items_json=detection_items_json,
                        width=width,
                        height=height,
                    )
            else:
                return await self.send_frame(
                    site=site,
                    stream_name=stream_name,
                    frame_bytes=encoded_frame,
                    warnings_json=warnings_json,
                    cone_polygons_json=cone_polygons_json,
                    pole_polygons_json=pole_polygons_json,
                    detection_items_json=detection_items_json,
                    width=width,
                    height=height,
                )

        except Exception as e:
            logging.error(f"Error sending optimized frame: {e}")
            return {'success': False, 'error': str(e)}

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
        # Ensure we have a valid token before making the request
        try:
            access_token = await self.token_manager.get_valid_token()
            headers = {'Authorization': f"Bearer {access_token}"}
        except Exception:
            # If token acquisition fails, try to authenticate
            await self.token_manager.authenticate(force=True)
            access_token = await self.token_manager.get_valid_token()
            headers = {'Authorization': f"Bearer {access_token}"}

        # Prepare multipart/form-data for the image and metadata
        files: dict[str, tuple[str, bytes, str]] = {
            'file': (
                'frame.jpg',  # Use .jpg extension for consistency
                frame_bytes,
                'image/jpeg',  # JPEG MIME type for better compression
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
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30, connect=10),
        )

        max_connect_attempts = 3
        attempt: int = 1

        while attempt <= max_connect_attempts:
            try:
                # Ensure access token is available and valid
                try:
                    access_token = await self.token_manager.get_valid_token()
                except Exception as e:
                    print(
                        '[WebSocket] Token error: ' + str(e) +
                        ', attempting to authenticate...',
                    )
                    await self.token_manager.authenticate(force=True)
                    access_token = await self.token_manager.get_valid_token()

                headers: dict[str, str] = {
                    'Authorization': f"Bearer {access_token}",
                    'User-Agent': 'ConstructionHazardDetection/1.0',
                }
                ws_url: str = self.base_url + '/ws/frames'

                print(
                    f'[WebSocket] Connecting to {ws_url} '
                    f'(attempt {attempt})...',
                )

                # Attempt to establish WebSocket connection
                self._ws = await self._session.ws_connect(
                    ws_url,
                    headers=headers,
                    heartbeat=30,  # Heartbeat interval
                    compress=0,    # Disable compression for performance
                )

                print('[WebSocket] Connection established successfully')
                return self._ws

            except aiohttp.ClientResponseError as e:
                if e.status == 401:
                    print(
                        '[WebSocket] Authentication failed (attempt '
                        f'{attempt}), refreshing token...',
                    )
                    await self.token_manager.refresh_token()
                elif e.status in [403, 404]:
                    print(f'[WebSocket] Server error {e.status}: {e.message}')
                    raise  # Don't retry these errors
                else:
                    print(
                        f'[WebSocket] HTTP error {e.status} (attempt '
                        f'{attempt}): {e.message}',
                    )

            except aiohttp.ClientConnectorError as e:
                print(f'[WebSocket] Connection error (attempt {attempt}): {e}')

            except asyncio.TimeoutError:
                print(f'[WebSocket] Connection timeout (attempt {attempt})')

            except Exception as e:
                print(f'[WebSocket] Unexpected error (attempt {attempt}): {e}')

            # If connection failed, close the session and wait before retrying
            if attempt < max_connect_attempts:
                delay = min(self.reconnect_backoff * attempt, 10.0)
                print(f'[WebSocket] Waiting {delay:.1f}s before retry...')
                await asyncio.sleep(delay)
                attempt += 1
            else:
                break

        # All connection attempts failed
        raise ConnectionError(
            'Failed to establish WebSocket connection after '
            f'{max_connect_attempts} attempts',
        )

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

        max_attempts = 5  # Limit maximum retry attempts
        attempt: int = 1

        while attempt <= max_attempts:
            try:
                ws: aiohttp.ClientWebSocketResponse = await self._ensure_ws()
                # Check WebSocket state before sending
                if ws.closed:
                    print(
                        f'[WebSocket] Already closed before send '
                        f'(attempt {attempt}), reconnecting...',
                    )
                    await self.close()
                    # Reconnect with backoff
                    delay = min(self.reconnect_backoff * attempt, 30.0)
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue

                # Send the message over WebSocket
                try:
                    await asyncio.wait_for(
                        ws.send_bytes(message),
                        timeout=10.0,
                    )
                except asyncio.TimeoutError:
                    print(
                        f'[WebSocket] Send timeout (attempt {attempt}), '
                        'reconnecting...',
                    )
                    await self.close()
                    delay = min(self.reconnect_backoff * attempt, 30.0)
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue

                # Set receive timeout
                try:
                    resp = await asyncio.wait_for(ws.receive(), timeout=10.0)
                except asyncio.TimeoutError:
                    print(
                        f'[WebSocket] Receive timeout (attempt {attempt}), '
                        'reconnecting...',
                    )
                    await self.close()
                    delay = min(self.reconnect_backoff * attempt, 30.0)
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue

                # Handle different WebSocket message types
                if resp.type == aiohttp.WSMsgType.TEXT:
                    return json.loads(resp.data)
                elif resp.type == aiohttp.WSMsgType.BINARY:
                    return json.loads(resp.data.decode('utf-8'))
                elif resp.type == aiohttp.WSMsgType.CLOSE:
                    print(
                        f'[WebSocket] Closed by server (attempt {attempt}), '
                        f'close code: {resp.data}, reconnecting...',
                    )
                    await self.close()
                    # Reconnect with backoff
                    if resp.data == 1008:  # Policy violation (auth error)
                        print(
                            '[WebSocket] Authentication error, '
                            'refreshing token...',
                        )
                        await self.token_manager.refresh_token()
                    delay = min(self.reconnect_backoff * attempt, 30.0)
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                elif resp.type == aiohttp.WSMsgType.ERROR:
                    print(
                        f'[WebSocket] Error response (attempt {attempt}): '
                        f'{resp.data}',
                    )
                    await self.close()
                    delay = min(self.reconnect_backoff * attempt, 30.0)
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                else:
                    print(f'[WebSocket] Unexpected message type: {resp.type}')
                    return {
                        'status': 'error',
                        'message': f'Unexpected message type: {resp.type}',
                    }

            except aiohttp.ClientError as e:
                # Network-related errors
                print(
                    f'[WebSocket] Network error (attempt {attempt}): {e}, '
                    'reconnecting...',
                )
                await self.close()
                delay = min(self.reconnect_backoff * attempt, 30.0)
                await asyncio.sleep(delay)
                attempt += 1
                continue
            except Exception as e:
                # Other exceptions
                print(
                    f'[WebSocket] Unexpected error (attempt {attempt}): {e}, '
                    'reconnecting...',
                )
                await self.close()
                delay = min(self.reconnect_backoff * attempt, 30.0)
                await asyncio.sleep(delay)
                attempt += 1
                continue

        # All retries failed
        await self.close()
        return {
            'status': 'error',
            'message': f'WebSocket send failed after {max_attempts} attempts',
        }

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
