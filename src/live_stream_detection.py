from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import os
from pathlib import Path
from typing import cast
from typing import TypedDict

import aiohttp
import cv2
import numpy as np
from aiohttp import ClientSession
from aiohttp import ClientWebSocketResponse
from aiohttp import WSMsgType
from dotenv import load_dotenv
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

from src.utils import TokenManager

# Load environment variables for configuration
load_dotenv()


class InputData(TypedDict):
    """Input data structure for detection processing.

    Attributes:
        frame: The input image frame as numpy array.
        model_key: The YOLO model identifier key.
        detect_with_server: Whether to use server-based detection.
    """
    frame: np.ndarray
    model_key: str
    detect_with_server: bool


class DetectionData(TypedDict):
    """Detection result data structure.

    Attributes:
        x1: Left coordinate of bounding box.
        y1: Top coordinate of bounding box.
        x2: Right coordinate of bounding box.
        y2: Bottom coordinate of bounding box.
        confidence: Detection confidence score (0.0 to 1.0).
        label: Class label identifier.
    """
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: int


class SharedToken(TypedDict, total=False):
    access_token: str
    refresh_token: str
    is_refreshing: bool


class LiveStreamDetector:
    """
    A class to perform live stream detection and tracking
    using YOLO with SAHI.
    """

    def __init__(
        self,
        api_url: str | None = None,
        model_key: str = 'yolo11n',
        output_folder: str | None = None,
        detect_with_server: bool = False,
        shared_token: SharedToken | None = None,
        use_ultralytics: bool = True,
        movement_thr: float = 40.0,
        fps: int = 1,
        max_id_keep: int = 10,
        ws_frame_size: tuple[int, int] | None = None,
        use_jpeg_ws: bool = True,
        remote_tracker: str = 'centroid',
        remote_cost_threshold: float = 0.7,
    ) -> None:
        """Initialise the LiveStreamDetector with specified configuration.

        Args:
            api_url: Base URL for the detection API server
                (env DETECT_API_URL if None).
            model_key: YOLO model identifier.
            output_folder: Optional directory for saving outputs.
            detect_with_server: Use remote WebSocket service if True.
            shared_token: Shared token dict for auth.
            use_ultralytics: Use Ultralytics engine (else SAHI slicing).
            movement_thr: Pixel movement threshold (centroid distance).
            fps: Target FPS (reserved for future time-based logic).
            max_id_keep: Frames to retain inactive track IDs.
            ws_frame_size: Optional (w,h) resize before WS send.
            use_jpeg_ws: JPEG (True) or PNG (False) for WS transmission.
            remote_tracker: 'centroid' or 'hungarian' for remote tracking.
            remote_cost_threshold: Cost cutoff (0-1) for Hungarian match.
        """
        # Resolve API URL
        if api_url is None:
            api_url = os.getenv(
                'DETECT_API_URL', 'https://changdar-server.mooo.com/api',
            )
        self.api_url = api_url.rstrip('/')
        self.model_key = model_key
        self.output_folder = output_folder
        self.detect_with_server = detect_with_server
        self.use_ultralytics = use_ultralytics

        # Tokens
        self.shared_token: SharedToken = shared_token or SharedToken(
            access_token='', refresh_token='', is_refreshing=False,
        )
        # Cast to the TokenManager expected type (dict[str, str | bool])
        self.token_manager = TokenManager(
            shared_token=cast('dict[str, str | bool]', self.shared_token),
        )

        # Models (local inference path)
        if not detect_with_server:
            # Uncomment for local inference using .engine files
            # (quantised from .pt)
            # if self.use_ultralytics:
            #     self.ultralytics_model = YOLO(
            #         f"models/int8_engine/best_{self.model_key}.engine",
            #     )

            if self.use_ultralytics:
                self.ultralytics_model = YOLO(
                    f"models/pt/best_{self.model_key}.pt",
                )
            else:
                self.model = AutoDetectionModel.from_pretrained(
                    'yolo11',
                    model_path=str(
                        Path('models/pt') /
                        f"best_{self.model_key}.pt",
                    ),
                    device='cuda:0',
                )

        # Network handles
        self._session: ClientSession | None = None
        self._ws: ClientWebSocketResponse | None = None
        self._logger = logging.getLogger(__name__)

        # Tracking state stores
        self.remote_tracks: dict[int, dict] = {}
        self.next_remote_id = 0
        self.prev_centers: dict[int, tuple[float, float]] = {}
        self.prev_centers_last_seen: dict[int, int] = {}
        self.movement_thr = movement_thr
        self.movement_thr_sq = movement_thr * movement_thr
        self.frame_count = 0
        self.max_id_keep = max_id_keep

        # WS transmission config
        self.ws_frame_size = ws_frame_size
        self.use_jpeg_ws = use_jpeg_ws

        # Remote tracking configuration
        self.remote_tracker = remote_tracker
        self.remote_cost_threshold = remote_cost_threshold

    async def _ensure_ws_connection(self) -> ClientWebSocketResponse:
        """Ensure a valid WebSocket connection is established.

        Returns:
            Active WebSocket connection ready for communication.

        Raises:
            ConnectionError:
                If all connection attempts fail after maximum retries.
        """
        # Reduced retry count for faster failure detection
        max_retries: int = 3
        retry_count: int = 0
        # Reduced initial backoff time
        backoff: float = 2

        while retry_count < max_retries:
            try:
                # Check if existing connection is still valid
                if self._ws and not self._ws.closed:
                    # Attempt to send ping to check connection health
                    try:
                        await self._ws.ping()
                        await asyncio.sleep(0.1)  # Wait for pong response
                        return self._ws
                    except Exception:
                        # Ping failed, connection is unhealthy, need to
                        # reconnect
                        self._logger.warning(
                            'Existing WS connection unhealthy, '
                            'reconnecting...',
                        )
                        await self.close()

                if self._session and self._session.closed:
                    self._session = None

                # Ensure valid token is available, authenticate if necessary
                await self.token_manager.ensure_token_valid()

                # Check if token is empty after authentication
                if not self.shared_token.get('access_token'):
                    self._logger.error(
                        'Access token is empty after authentication',
                    )
                    raise ConnectionError(
                        'Failed to obtain valid access token',
                    )

                self._logger.info(
                    f"Attempting WebSocket connection to: {self.api_url}",
                )
                self._logger.debug(
                    f"Using access token: "
                    f"{self.shared_token['access_token'][:20]}..." if
                    self.shared_token.get('access_token') else 'No token',
                )

                # Try Method 1: Use header to transmit model_key (recommended)
                success = await self._try_header_connection()
                if success:
                    return self._ws

                # Method 1 failed, try Method 2: Use first message to transmit
                # model_key
                self._logger.info(
                    'Header method failed, trying first message method...',
                )
                success = await self._try_first_message_connection()
                if success:
                    return self._ws

                # Method 2 also failed, try Method 3: Use query parameter
                # (backward compatibility)
                self._logger.info(
                    'First message method failed, trying legacy query '
                    'parameter method...',
                )
                success = await self._try_legacy_connection()
                if success:
                    return self._ws

                # All methods failed, raise exception to enter retry logic
                raise ConnectionError('All connection methods failed')

            except Exception as e:
                # Check if this is a token expiry related error
                error_message = str(e).lower()
                if any(
                    keyword in error_message for keyword in [
                        'expired', '401', 'unauthorized', 'invalid token',
                    ]
                ):
                    self._logger.warning(
                        f"Token-related error detected: {e}. "
                        'Attempting token refresh...',
                    )
                    try:
                        await self.token_manager.refresh_token()
                        self._logger.info(
                            'Token refreshed successfully, will retry '
                            'connection',
                        )
                    except Exception as refresh_error:
                        self._logger.error(
                            f"Token refresh failed: {refresh_error}. "
                            'Will re-authenticate on next retry',
                        )

                self._logger.error(
                    f"WS connection failed: {e}, retrying "
                    f"({retry_count + 1}/{max_retries}) after {backoff}s",
                )
                await self.close()
                retry_count += 1
                if retry_count < max_retries:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 1.5, 10)  # More gradual backoff

        raise ConnectionError(
            f"Max retries ({max_retries}) reached for WebSocket "
            'connection.',
        )

    async def _try_header_connection(self) -> bool:
        """Attempt connection using header method.

        This method attempts to establish a WebSocket connection by
        transmitting the model key via HTTP headers, which is the
        recommended secure approach.

        Returns:
            True if connection successful, False otherwise.

        Raises:
            ConnectionError: For authentication-related failures that need
                token refresh.
        """
        try:
            headers = {
                # Use lowercase key
                'authorization': (
                    f"Bearer {self.shared_token['access_token']}"
                ),
                'x-model-key': self.model_key,  # Use lowercase key
            }

            ws_url = (
                self.api_url.replace(
                    'https://', 'wss://',
                ).replace('http://', 'ws://')
                + '/ws/detect'
            )

            self._logger.debug(f"Header method connecting to: {ws_url}")
            self._logger.debug(
                f"Headers: authorization=Bearer "
                f"{self.shared_token['access_token'][:20]}..., "
                f"x-model-key={self.model_key}",
            )

            if not self._session:
                timeout = aiohttp.ClientTimeout(total=30, connect=10)
                self._session = aiohttp.ClientSession(timeout=timeout)

            self._ws = await self._session.ws_connect(
                ws_url,
                headers=headers,
                heartbeat=30,
                autoping=True,
                max_msg_size=50 * 1024 * 1024,
            )

            # Wait for configuration confirmation response
            try:
                config_msg = await asyncio.wait_for(
                    self._ws.receive(), timeout=5.0,
                )
                if config_msg.type == WSMsgType.TEXT:
                    config_response = json.loads(config_msg.data)
                    if config_response.get('status') == 'ready':
                        self._logger.info(
                            f"Header method successful. "
                            f"Model: {config_response.get('model')}",
                        )
                        return True
                    else:
                        self._logger.warning(
                            f"Unexpected config response: {config_response}",
                        )
                        return False
            except asyncio.TimeoutError:
                self._logger.warning(
                    'Timeout waiting for configuration confirmation',
                )
                return False
            return False

        except Exception as e:
            self._logger.warning(f"Header method failed: {e}")
            # If this is a 403 or token-related error,
            # raise specific error for upper layer handling
            if hasattr(e, 'status') and e.status == 403:
                self._logger.error(
                    f"Authentication failed (403). Token: "
                    f"{self.shared_token['access_token'][:20]}... "
                    f"API URL: {self.api_url}",
                )
                raise ConnectionError(
                    'Authentication failed - token may be expired',
                )
            elif any(
                keyword in str(e).lower()
                for keyword in ['expired', 'unauthorized', 'invalid token']
            ):
                raise ConnectionError(f"Token-related error: {e}")
            return False

    async def _try_first_message_connection(self) -> bool:
        """
        Attempt connection using first message method.

        Returns:
            True if connection successful, False otherwise.

        Raises:
            ConnectionError:
                For authentication-related failures that need token refresh.
        """
        try:
            headers = {
                # Use lowercase key
                'authorization': f"Bearer {self.shared_token['access_token']}",
            }

            ws_url = (
                self.api_url.replace(
                    'https://', 'wss://',
                ).replace('http://', 'ws://')
                + '/ws/detect'
            )

            self._logger.debug(f"First message method connecting to: {ws_url}")

            if not self._session:
                timeout = aiohttp.ClientTimeout(total=30, connect=10)
                self._session = aiohttp.ClientSession(timeout=timeout)

            self._ws = await self._session.ws_connect(
                ws_url,
                headers=headers,
                heartbeat=30,
                autoping=True,
                max_msg_size=50 * 1024 * 1024,
            )

            # Send configuration message
            config_message = {
                'model_key': self.model_key,
                'config': {
                    'confidence_threshold': 0.5,
                    'iou_threshold': 0.4,
                },
            }
            await self._ws.send_str(json.dumps(config_message))

            # Wait for configuration confirmation response
            try:
                config_msg = await asyncio.wait_for(
                    self._ws.receive(), timeout=5.0,
                )
                if config_msg.type == WSMsgType.TEXT:
                    config_response = json.loads(config_msg.data)
                    if config_response.get('status') == 'ready':
                        self._logger.info(
                            f"First message method successful. "
                            f"Model: {config_response.get('model')}",
                        )
                        return True
                    else:
                        self._logger.warning(
                            f"Unexpected config response: {config_response}",
                        )
                        return False
            except asyncio.TimeoutError:
                self._logger.warning(
                    'Timeout waiting for configuration confirmation',
                )
                return False
            return False

        except Exception as e:
            self._logger.warning(f"First message method failed: {e}")
            # If this is a 403 or token-related error,
            # raise specific error for upper layer handling
            if hasattr(e, 'status') and e.status == 403:
                self._logger.error(
                    f"Authentication failed (403). Token: "
                    f"{self.shared_token['access_token'][:20]}... "
                    f"API URL: {self.api_url}",
                )
                raise ConnectionError(
                    'Authentication failed - token may be expired',
                )
            elif any(
                keyword in str(e).lower()
                for keyword in ['expired', 'unauthorized', 'invalid token']
            ):
                raise ConnectionError(f"Token-related error: {e}")
            return False

    async def _try_legacy_connection(self) -> bool:
        """
        Attempt connection using query parameter method
        (backward compatibility).

        Returns:
            True if connection successful, False otherwise.

        Raises:
            ConnectionError:
                For authentication-related failures that need token refresh.
        """
        try:
            headers = {
                # Use lowercase key
                'authorization': f"Bearer {self.shared_token['access_token']}",
            }

            # Use legacy query parameter method (not recommended, but
            # backward compatible)
            ws_url = (
                self.api_url.replace(
                    'https://', 'wss://',
                ).replace('http://', 'ws://')
                + f'/ws/detect?model={self.model_key}'
            )

            self._logger.debug(f"Legacy method connecting to: {ws_url}")

            if not self._session:
                timeout = aiohttp.ClientTimeout(total=30, connect=10)
                self._session = aiohttp.ClientSession(timeout=timeout)

            self._ws = await self._session.ws_connect(
                ws_url,
                headers=headers,
                heartbeat=30,
                autoping=True,
                max_msg_size=50 * 1024 * 1024,
            )

            # Wait for configuration confirmation response (if available)
            try:
                config_msg = await asyncio.wait_for(
                    self._ws.receive(), timeout=2.0,
                )
                if config_msg.type == WSMsgType.TEXT:
                    config_response = json.loads(config_msg.data)
                    if config_response.get('status') == 'ready':
                        self._logger.info(
                            f"Legacy method successful. "
                            f"Model: {config_response.get('model')}",
                        )
                        return True
                    else:
                        # Might be an older service version without config
                        # confirmation, return success directly
                        self._logger.info(
                            'Legacy method successful (no config '
                            'confirmation)',
                        )
                        return True
            except asyncio.TimeoutError:
                # Older service versions may not send configuration
                # confirmation, this is normal
                self._logger.info(
                    'Legacy method successful (no config confirmation)',
                )
                return True
            return False

        except Exception as e:
            self._logger.warning(f"Legacy method failed: {e}")
            # If this is a 403 or token-related error,
            # raise specific error for upper layer handling
            if hasattr(e, 'status') and e.status == 403:
                self._logger.error(
                    f"Authentication failed (403). Token: "
                    f"{self.shared_token['access_token'][:20]}... "
                    f"API URL: {self.api_url}",
                )
                raise ConnectionError(
                    'Authentication failed - token may be expired',
                )
            elif any(
                keyword in str(e).lower()
                for keyword in ['expired', 'unauthorized', 'invalid token']
            ):
                raise ConnectionError(f"Token-related error: {e}")
            return False

    async def _detect_cloud_ws(self, frame: np.ndarray) -> list[list[float]]:
        """
        Perform object detection using WebSocket connection to remote server.

        Args:
            frame: Input image frame as numpy array for detection.

        Returns:
            List of detection results, where each detection is represented as
            [x1, y1, x2, y2, confidence, class_id].

        Note:
            Returns empty list if all retry attempts fail to prevent system
            crashes.
        """
        backoff: float = 1  # Faster initial backoff
        max_backoff: float = 15  # Smaller maximum backoff
        retry_count: int = 0
        max_retries: int = 3  # Reduced retry count

        while retry_count < max_retries:
            try:
                # Check if token is about to expire before sending request
                if self.token_manager.is_token_expired():
                    self._logger.info(
                        'Token expiring soon, preemptively refreshing...',
                    )
                    try:
                        await self.token_manager.refresh_token()
                        # Need to re-establish WebSocket connection after
                        # token refresh
                        await self.close()
                    except Exception as e:
                        self._logger.warning(
                            f"Preemptive token refresh failed: {e}",
                        )

                ws = await self._ensure_ws_connection()

                # Check WebSocket state, reconnect if closed
                if ws.closed:
                    self._logger.warning(
                        'WebSocket is closed before sending, '
                        'reconnecting...',
                    )
                    await self._close_and_retry()
                    retry_count += 1
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 1.5, max_backoff)
                    continue

                # Prepare frame for transmission
                frame_to_send = self._prepare_frame(frame)
                img_buf = self._encode_frame(frame_to_send)
                if img_buf is None:
                    return []

                # Send frame and get response
                response = await self._send_and_receive(
                    ws, img_buf, backoff, max_backoff,
                )
                if response is None:
                    retry_count += 1
                    continue

                return response

            except Exception as e:
                if await self._handle_exception(e):
                    await self.close()

                self._logger.error(
                    f"WS error: {e}, retrying "
                    f"({retry_count + 1}/{max_retries}) after {backoff}s.",
                )
                await self.close()
                retry_count += 1
                if retry_count < max_retries:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 1.5, max_backoff)

        # If all retries fail,
        # return empty results instead of raising exception
        self._logger.error(
            f"WebSocket connection failed after {max_retries} retries, "
            'returning empty results',
        )
        return []

    async def _close_and_retry(self) -> None:
        """Helper method to close connection and prepare for retry."""
        await self.close()

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """Prepare frame for transmission by resizing if needed."""
        if self.ws_frame_size:
            return cv2.resize(frame, self.ws_frame_size)
        return frame

    def _encode_frame(self, frame: np.ndarray) -> bytes | None:
        """Encode frame to bytes for WebSocket transmission."""
        if self.use_jpeg_ws:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
            success, img_buf = cv2.imencode('.jpg', frame, encode_params)
        else:
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 7]
            success, img_buf = cv2.imencode('.png', frame, encode_params)

        if not success:
            self._logger.error('Failed to encode frame.')
            return None

        return img_buf.tobytes()

    async def _send_and_receive(
        self, ws: ClientWebSocketResponse, img_buf: bytes,
        backoff: float, max_backoff: float,
    ) -> list[list[float]] | None:
        """Send frame and receive response from WebSocket."""
        try:
            # Send with timeout
            await asyncio.wait_for(ws.send_bytes(img_buf), timeout=10.0)
        except (
            aiohttp.ClientConnectionError, ConnectionResetError,
            RuntimeError, asyncio.TimeoutError,
        ) as send_err:
            self._logger.warning(
                f"[WebSocket] Send fail: {send_err}, reconnecting...",
            )
            await self.close()
            return None

        try:
            # Receive with timeout
            msg = await asyncio.wait_for(ws.receive(), timeout=15.0)
        except asyncio.TimeoutError:
            self._logger.warning(
                '[WebSocket] Receive timeout, reconnecting...',
            )
            await self.close()
            return None

        return await self._process_message(msg)

    async def _process_message(self, msg) -> list[list[float]] | None:
        """Process WebSocket message and extract detection results."""
        if msg.type == WSMsgType.CLOSE:
            self._logger.warning(
                '[WebSocket] Closed by server, reconnecting...',
            )
            await self.close()
            return None

        if msg.type in (WSMsgType.PING, WSMsgType.PONG):
            self._logger.debug('Received ping/pong, continuing...')
            return None

        if msg.type not in (WSMsgType.TEXT, WSMsgType.BINARY):
            self._logger.warning(f"Unexpected WS message type: {msg.type}")
            return []

        try:
            data = json.loads(
                msg.data if msg.type == WSMsgType.TEXT
                else msg.data.decode('utf-8', 'ignore'),
            )
            return await self._handle_response_data(data)
        except json.JSONDecodeError as e:
            self._logger.error(f"JSON decode error: {e}")
            return []

    async def _handle_response_data(self, data) -> list[list[float]] | None:
        """Handle different types of response data."""
        if not isinstance(data, dict):
            if isinstance(data, list):
                return data
            self._logger.warning(f"Unexpected data format: {type(data)}")
            return []

        # Handle ping messages
        if data.get('type') == 'ping':
            return None

        # Handle error messages
        if 'error' in data:
            return await self._handle_server_error(data['error'])

        # Handle status messages
        if data.get('status') == 'ready':
            self._logger.debug('Received ready status, continuing...')
            return None

        return []

    async def _handle_server_error(self, error_msg: str) -> list[list[float]]:
        """Handle server error messages."""
        self._logger.error(f"Server error: {error_msg}")

        if any(
            keyword in error_msg.lower() for keyword in [
                'expired', 'unauthorized', 'invalid token',
            ]
        ):
            self._logger.warning(
                'Token-related server error, attempting refresh...',
            )
            try:
                await self.token_manager.refresh_token()
                await self.close()
            except Exception as refresh_error:
                self._logger.error(f"Token refresh failed: {refresh_error}")

        return []

    async def _handle_exception(self, e: Exception) -> bool:
        """Handle exceptions and return True if token refresh was attempted."""
        error_message = str(e).lower()
        if any(
            keyword in error_message for keyword in [
                'expired', '401', 'unauthorized', 'invalid token',
            ]
        ):
            self._logger.warning(
                (
                    f"Token-related error in detection: {e}. "
                    'Attempting refresh...'
                ),
            )
            try:
                await self.token_manager.refresh_token()
                return True
            except Exception as refresh_error:
                self._logger.error(
                    f"Token refresh failed during detection: {refresh_error}",
                )
        return False

    async def _detect_local(self, frame: np.ndarray) -> list[list[float]]:
        """Perform object detection using local YOLO models.

        This method runs inference locally using either Ultralytics YOLO or
        SAHI AutoDetectionModel, depending on the configuration.

        Args:
            frame: Input image frame as numpy array for detection.

        Returns:
            List of detection results, where each detection is represented as
            [x1, y1, x2, y2, confidence, class_id].
        """
        if self.use_ultralytics:
            # Use Ultralytics YOLO for direct inference
            result = self.ultralytics_model(frame)
            boxes = result[0].boxes
            return [
                [
                    *map(float, boxes.xyxy[i].tolist()),
                    float(boxes.conf[i].item()),
                    int(boxes.cls[i].item()),
                ]
                for i in range(len(boxes))
            ]
        else:
            # Use SAHI for sliced inference on large images
            result = get_sliced_prediction(
                frame, self.model,
                slice_height=376, slice_width=376,
                overlap_height_ratio=0.3, overlap_width_ratio=0.3,
            )
            return [
                [
                    *map(int, obj.bbox.to_voc_bbox()),
                    float(obj.score.value),
                    int(obj.category.id),
                ]
                for obj in result.object_prediction_list
            ]

    async def generate_detections(
        self, frame: np.ndarray,
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Generate object detections with tracking information.

        This is the main detection method that coordinates between local and
        remote inference, applies object tracking, and manages frame counting.

        Args:
            frame: Input image frame as numpy array for detection.

        Returns:
            Tuple containing:
                - List of raw detection results
                  [x1, y1, x2, y2, confidence, class_id]
                - List of tracked detection results
                  [x1, y1, x2, y2, confidence, class_id, track_id, is_moving]
        """
        self.frame_count += 1
        if self.detect_with_server:
            datas = await self._detect_cloud_ws(frame)
            tracked = self._track_remote(datas)
        else:
            # Batch process detection results to improve efficiency
            results = self.ultralytics_model.track(
                frame, persist=True, verbose=False,
            )
            boxes = results[0].boxes

            if len(boxes) == 0:
                self._cleanup_prev_centers()
                return [], []

            ids = results[0].boxes.id if results[0].boxes.id is not None else [
                -1,
            ] * len(boxes)

            # Batch calculate all bounding box data
            xyxy_batch = boxes.xyxy.tolist()
            conf_batch = boxes.conf.tolist()
            cls_batch = boxes.cls.tolist()

            datas = []
            tracked = []

            for i in range(len(boxes)):
                xyxy = xyxy_batch[i]
                conf = float(conf_batch[i])
                cls = int(cls_batch[i])
                tid = (
                    int(ids[i]) if ids is not None and ids[i] is not None
                    else -1
                )

                # Calculate centre point and movement status
                cx, cy = (xyxy[0] + xyxy[2]) * 0.5, (xyxy[1] + xyxy[3]) * 0.5
                is_moving = 0

                if tid != -1:
                    prev_c = self.prev_centers.get(tid)
                    if prev_c:
                        # Use pre-computed square distance comparison
                        distance_sq = (
                            (cx - prev_c[0]) ** 2 + (cy - prev_c[1]) ** 2
                        )
                        is_moving = (
                            1 if distance_sq > self.movement_thr_sq else 0
                        )

                    self.prev_centers[tid] = (cx, cy)
                    self.prev_centers_last_seen[tid] = self.frame_count

                datas.append(xyxy + [conf, cls])
                tracked.append(xyxy + [conf, cls, tid, is_moving])
            self._cleanup_prev_centers()
            return datas, tracked
        self._cleanup_prev_centers()
        return datas, tracked

    def _cleanup_prev_centers(self) -> None:
        """Clean up tracking data for inactive object IDs.

        This method removes tracking information for objects that haven't been
        seen for more than max_id_keep frames to prevent memory leaks and
        maintain tracking performance.
        """
        # Clean up IDs that haven't appeared for more than max_id_keep frames
        if self.frame_count % 10 == 0:
            current_frame = self.frame_count
            expired_ids = [
                tid for tid, last_seen in self.prev_centers_last_seen.items()
                if current_frame - last_seen > self.max_id_keep
            ]
            for tid in expired_ids:
                self.prev_centers.pop(tid, None)
                self.prev_centers_last_seen.pop(tid, None)

    def _track_remote(self, dets: list[list[float]]) -> list[list[float]]:
        """Dispatch to the configured remote tracker implementation."""
        if self.remote_tracker == 'hungarian':
            return self._track_remote_hungarian(dets)
        # Default / fallback
        return self._track_remote_centroid(dets)

    # ------------------------------------------------------------------
    # Small utilities for tracking readability
    # ------------------------------------------------------------------
    # Centralized numeric constants for the Hungarian helpers
    _LARGE_COST: float = 1e6
    _ZERO_EPS: float = 1e-12

    def _bbox_center(
            self, x1: float, y1: float, x2: float, y2: float,
    ) -> tuple[float, float]:
        """
        Return the center point (cx, cy) of a bbox.

        Args:
            x1: The x1 coordinate of the bbox.
            y1: The y1 coordinate of the bbox.
            x2: The x2 coordinate of the bbox.
            y2: The y2 coordinate of the bbox.

        Returns:
            The center point (cx, cy) of the bbox.
        """
        return (x1 + x2) * 0.5, (y1 + y2) * 0.5

    def _bbox_iou(
        self,
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
    ) -> float:
        """
        Compute IoU for two boxes (x1,y1,x2,y2).

        Args:
            a: The first box (x1, y1, x2, y2).
            b: The second box (x1, y1, x2, y2).

        Returns:
            The IoU (Intersection over Union) of the two boxes.
        """
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        else:
            inter_area = 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    def _squared_distance(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
    ) -> float:
        """
        Return squared Euclidean distance between points p1 and p2.

        Args:
            p1: The first point (x, y).
            p2: The second point (x, y).

        Returns:
            The squared Euclidean distance.
        """
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx * dx + dy * dy

    def _set_remote_track(
        self,
        tid: int,
        bbox: tuple[float, float, float, float],
        cls_id: int,
        center: tuple[float, float],
    ) -> None:
        """
        Upsert a remote track's latest state.

        Args:
            tid: The track ID.
            bbox: The bounding box (x1, y1, x2, y2).
            cls_id: The class ID.
            center: The center point (cx, cy).

        Returns:
            None
        """
        self.remote_tracks[tid] = {
            'bbox': bbox,
            'center': center,
            'last_seen': self.frame_count,
            'cls': cls_id,
        }

    def _new_track_for_det(self, det: list[float]) -> list[float]:
        """
        Create a new track entry for a single detection.

        Args:
            det: A single detection [x1, y1, x2, y2, conf, cls_id].

        Returns:
            A tracked row [x1, y1, x2, y2, conf, cls_id, track_id, is_moving].
        """
        x1, y1, x2, y2, conf, cls_id = det
        cx, cy = self._bbox_center(x1, y1, x2, y2)
        tid = self.next_remote_id
        self.next_remote_id += 1
        self._set_remote_track(tid, (x1, y1, x2, y2), int(cls_id), (cx, cy))
        return [x1, y1, x2, y2, conf, cls_id, tid, 0]

    # ------------------------------------------------------------------
    # Centroid tracker (original simple implementation)
    # ------------------------------------------------------------------
    def _track_remote_centroid(
        self, dets: list[list[float]],
    ) -> list[list[float]]:
        """Simple centroid-based tracker for remote detections.

        Args:
            dets: List of detections [x1, y1, x2, y2, conf, cls].

        Returns:
            List with tracking info [x1, y1, x2, y2, conf, cls, track_id,
            is_moving].
        """
        if not dets:
            if self.frame_count % 10 == 0:
                self._prune_remote_tracks()
            return []

        assigned_tracks: list[list[float]] = []
        used_track_ids: set[int] = set()
        track_items = list(self.remote_tracks.items())
        track_centers = [info['center'] for _, info in track_items]
        track_cls = [info['cls'] for _, info in track_items]

        for det in dets:
            x1, y1, x2, y2, conf, cls_id = det
            cx, cy = self._bbox_center(x1, y1, x2, y2)
            best_tid = None
            best_dist_sq = float('inf')
            for (tid, _), (tcx, tcy), tcls in zip(
                track_items, track_centers, track_cls,
            ):
                if tid in used_track_ids or tcls != cls_id:
                    continue
                dist_sq = self._squared_distance((cx, cy), (tcx, tcy))
                if (
                    dist_sq < best_dist_sq and
                    dist_sq < (self.movement_thr_sq * 4)
                ):
                    best_dist_sq = dist_sq
                    best_tid = tid
            if best_tid is None:
                # new track
                tracked_row = self._new_track_for_det(det)
                assigned_tracks.append(tracked_row)
                continue
            else:
                tid = best_tid
                used_track_ids.add(tid)
                prev_center = self.remote_tracks[tid]['center']
                dist_sq_move = self._squared_distance((cx, cy), prev_center)
                moving_flag = 1 if dist_sq_move > self.movement_thr_sq else 0
            self._set_remote_track(
                tid, (x1, y1, x2, y2),
                int(cls_id), (cx, cy),
            )
            assigned_tracks.append(
                [x1, y1, x2, y2, conf, cls_id, tid, moving_flag],
            )
        if self.frame_count % 10 == 0:
            self._prune_remote_tracks()
        return assigned_tracks

    # ------------------------------------------------------------------
    # Hungarian (global) assignment tracker
    # ------------------------------------------------------------------
    def _track_remote_hungarian(
        self, dets: list[list[float]],
    ) -> list[list[float]]:
        """
        Global assignment tracker using Hungarian algorithm.

        Args:
            dets: List of detections [x1, y1, x2, y2, conf, cls].

        Returns:
            List with tracking info [x1, y1, x2, y2, conf, cls, track_id,
            is_moving].
        """
        if not dets:
            if self.frame_count % 10 == 0:
                self._prune_remote_tracks()
            return []

        # Build arrays for existing tracks
        track_items = list(self.remote_tracks.items())  # (tid, info)
        num_tracks = len(track_items)

        # If no existing tracks, create new ones directly
        if num_tracks == 0:
            return self._assign_new_tracks_for_all(dets)

        # Prepare cost matrix and solve assignment
        cost_matrix = self._build_cost_matrix(dets, track_items)
        matches, unmatched_dets, _ = self._hungarian_assign(
            cost_matrix, self.remote_cost_threshold,
        )

        assigned, used_track_ids = self._update_matched_tracks(
            dets, track_items, matches,
        )
        assigned += self._create_tracks_for_unmatched(dets, unmatched_dets)

        # (Optionally) we could retire unmatched tracks after grace period;
        # pruning handles this
        if self.frame_count % 10 == 0:
            self._prune_remote_tracks()
        return assigned

    def _assign_new_tracks_for_all(
        self, dets: list[list[float]],
    ) -> list[list[float]]:
        """
        Create brand-new tracks for each detection when no tracks exist.

        Args:
            dets: List of detections [x1, y1, x2, y2, conf, cls].

        Returns:
            List with tracking info [x1, y1, x2, y2, conf, cls, track_id,
            is_moving].
        """
        assigned: list[list[float]] = []
        for det in dets:
            assigned.append(self._new_track_for_det(det))
        return assigned

    def _build_cost_matrix(
        self,
        dets: list[list[float]],
        track_items: list[tuple[int, dict]],
    ) -> np.ndarray:
        """
        Compute the cost matrix for detections vs tracks.

        Args:
            dets: List of detections [x1, y1, x2, y2, conf, cls].
            track_items: List of track items [(track_id, info), ...].

        Returns:
            Cost matrix as a NumPy array.
        """
        num_dets = len(dets)
        num_tracks = len(track_items)
        cost_matrix = np.full(
            (num_dets, num_tracks),
            self._LARGE_COST, dtype=float,
        )

        for d_idx, det in enumerate(dets):
            for t_idx, (_tid, info) in enumerate(track_items):
                cost_matrix[d_idx, t_idx] = self._compute_pair_cost(det, info)
        return cost_matrix

    def _compute_pair_cost(self, det: list[float], info: dict) -> float:
        """
        Compute combined IoU and distance based cost for a pair.

        Args:
            det: Detection [x1, y1, x2, y2, conf, cls].
            info: Track info dictionary.

        Returns:
            Combined cost as a float.
        """
        x1, y1, x2, y2, _conf, cls_id = det
        if info['cls'] != cls_id:
            return self._LARGE_COST
        tx1, ty1, tx2, ty2 = info['bbox']
        tcx, tcy = info['center']
        # IoU
        iou = self._bbox_iou((x1, y1, x2, y2), (tx1, ty1, tx2, ty2))
        # Distance normalised
        cx, cy = self._bbox_center(x1, y1, x2, y2)
        dist_sq = self._squared_distance((cx, cy), (tcx, tcy))
        dist_norm = min(dist_sq / (self.movement_thr_sq * 4), 1.0)
        return 0.5 * (1 - iou) + 0.5 * dist_norm

    def _update_matched_tracks(
        self,
        dets: list[list[float]],
        track_items: list[tuple[int, dict]],
        matches: list[tuple[int, int]],
    ) -> tuple[list[list[float]], set[int]]:
        """
        Update matched tracks and compute moving flags.

        Args:
            dets:
                List of detections [x1, y1, x2, y2, conf, cls].
            track_items:
                List of track items [(track_id, info), ...].
            matches:
                List of matched pairs [(detection_index, track_index), ...].

        Returns:
            Tuple of (assigned_tracks, used_track_ids).
        """
        assigned: list[list[float]] = []
        used_track_ids: set[int] = set()

        # Update matched tracks
        for d_idx, t_idx in matches:
            x1, y1, x2, y2, conf, cls_id = dets[d_idx]
            tid, info = track_items[t_idx]
            cx, cy = self._bbox_center(x1, y1, x2, y2)
            prev_center = info['center']
            dist_sq_move = self._squared_distance((cx, cy), prev_center)
            moving_flag = 1 if dist_sq_move > self.movement_thr_sq else 0
            self._set_remote_track(
                tid, (x1, y1, x2, y2),
                int(cls_id), (cx, cy),
            )
            assigned.append([x1, y1, x2, y2, conf, cls_id, tid, moving_flag])
            used_track_ids.add(tid)
        return assigned, used_track_ids

    def _create_tracks_for_unmatched(
        self, dets: list[list[float]], unmatched_dets: list[int],
    ) -> list[list[float]]:
        """
        Create tracks for unmatched detections.

        Args:
            dets: List of detections [x1, y1, x2, y2, conf, cls].
            unmatched_dets: List of unmatched detection indices.

        Returns:
            List of new track representations.
        """
        assigned: list[list[float]] = []
        for d_idx in unmatched_dets:
            assigned.append(self._new_track_for_det(dets[d_idx]))
        return assigned

    # Hungarian assignment helper
    def _hungarian_assign(
        self, cost: np.ndarray, cost_threshold: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """
        Apply Hungarian algorithm on cost matrix and filter by threshold.

        Args:
            cost: Cost matrix as a 2D numpy array.
            cost_threshold: Cost threshold for filtering matches.

        Returns:
            (matches, unmatched_rows, unmatched_cols)
        """
        # Copy and square-pad
        cost_matrix = cost.copy()
        num_rows, num_cols = cost_matrix.shape
        cost_matrix, n = self._pad_to_square(cost_matrix)

        # Reduce
        self._row_col_reduce(cost_matrix)

        # Iteratively cover zeros and adjust
        while True:
            covered_rows, covered_cols = self._cover_zeros(cost_matrix)
            if len(covered_rows) + len(covered_cols) >= n:
                break
            if not self._adjust_matrix_with_min(
                cost_matrix, covered_rows, covered_cols,
            ):
                break

        # Extract assignment greedily from zeros (since matrix reduced)
        assigned_cols = set()
        matches_all: list[tuple[int, int]] = []
        for r in range(n):
            zero_cols = [
                c for c in range(n)
                if (
                    abs(cost_matrix[r, c]) < self._ZERO_EPS
                    and c not in assigned_cols
                )
            ]
            if zero_cols:
                c_sel = zero_cols[0]
                assigned_cols.add(c_sel)
                matches_all.append((r, c_sel))

        # Filter to original matrix size and cost threshold
        matches: list[tuple[int, int]] = []
        used_rows = set()
        used_cols = set()
        for r, c in matches_all:
            if r < num_rows and c < num_cols:
                original_cost = cost[r, c]
                if original_cost <= cost_threshold:
                    matches.append((r, c))
                    used_rows.add(r)
                    used_cols.add(c)

        unmatched_rows = [r for r in range(num_rows) if r not in used_rows]
        unmatched_cols = [c for c in range(num_cols) if c not in used_cols]
        return matches, unmatched_rows, unmatched_cols

    def _pad_to_square(self, mat: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Pad a rectangular matrix to square by filling with large cost.

        Args:
            mat: Input matrix to pad.

        Returns:
            Tuple of (padded_matrix, new_size).
        """
        num_rows, num_cols = mat.shape
        n = max(num_rows, num_cols)
        if num_rows == num_cols:
            return mat, n
        padded = np.full((n, n), self._LARGE_COST, dtype=float)
        padded[:num_rows, :num_cols] = mat
        return padded, n

    def _row_col_reduce(self, mat: np.ndarray) -> None:
        """Perform row and column reduction in-place."""
        mat -= mat.min(axis=1, keepdims=True)
        mat -= mat.min(axis=0, keepdims=True)

    def _cover_zeros(
        self, mat: np.ndarray,
    ) -> tuple[set[int], set[int]]:
        """
        Cover all zeros using minimum number of horizontal/vertical lines.

        Args:
            mat: Input matrix to cover zeros.

        Returns:
            Tuple of (covered_rows, covered_cols).
        """
        n_local = mat.shape[0]
        covered_rows: set[int] = set()
        covered_cols: set[int] = set()
        zero_locs = [
            (r, c) for r in range(n_local)
            for c in range(n_local) if abs(mat[r, c]) < self._ZERO_EPS
        ]
        row_zero_count = {r: 0 for r in range(n_local)}
        col_zero_count = {c: 0 for c in range(n_local)}
        for r, c in zero_locs:
            row_zero_count[r] += 1
            col_zero_count[c] += 1
        while zero_locs:
            row_counts = {r: 0 for r in range(n_local)}
            col_counts = {c: 0 for c in range(n_local)}
            for (r, c) in zero_locs:
                if r not in covered_rows and c not in covered_cols:
                    row_counts[r] += 1
                    col_counts[c] += 1
            if not row_counts and not col_counts:
                break
            if (
                row_counts and (
                    not col_counts or
                    max(row_counts.values()) >= max(col_counts.values())
                )
            ):
                r_sel = max(row_counts, key=lambda k: row_counts[k])
                covered_rows.add(r_sel)
            else:
                c_sel = max(col_counts, key=lambda k: col_counts[k])
                covered_cols.add(c_sel)
            zero_locs = [
                zc for zc in zero_locs
                if zc[0] not in covered_rows and zc[1] not in covered_cols
            ]
        return covered_rows, covered_cols

    def _adjust_matrix_with_min(
        self, mat: np.ndarray, covered_rows: set[int], covered_cols: set[int],
    ) -> bool:
        """
        Adjust the matrix by subtracting the minimum value from uncovered
        cells and adding it to the intersections of covered rows and columns.

        Args:
            mat: Input matrix to adjust.
            covered_rows: Set of covered row indices.
            covered_cols: Set of covered column indices.

        Returns:
            True if adjustment was applied, False otherwise.
        """
        n = mat.shape[0]
        uncovered = [
            mat[r, c] for r in range(n)
            if r not in covered_rows for c in range(n)
            if c not in covered_cols
        ]
        if not uncovered:
            return False
        m = min(uncovered)
        for r in range(n):
            for c in range(n):
                if r not in covered_rows and c not in covered_cols:
                    mat[r, c] -= m
                elif r in covered_rows and c in covered_cols:
                    mat[r, c] += m
        return True

    def _prune_remote_tracks(self) -> None:
        """
        Remove remote tracks that have not been updated
        within max_id_keep frames.
        """
        threshold = self.frame_count - self.max_id_keep
        stale = [
            tid for tid, info in self.remote_tracks.items()
            if info['last_seen'] < threshold
        ]
        for tid in stale:
            self.remote_tracks.pop(tid, None)

    async def run_detection(self, stream_url: str) -> None:
        """Run continuous object detection on a video stream.

        This method opens a video stream, performs real-time object detection
        with tracking, and displays the results in a window. The detection
        loop continues until the user presses 'q' to quit.

        Args:
            stream_url: URL or path to the video stream source.

        Raises:
            ValueError: If the stream cannot be opened.
        """
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            raise ValueError('Failed to open stream.')
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    await asyncio.sleep(1)
                    continue
                datas, tracked = await self.generate_detections(frame)
                disp = frame.copy()  # Use copy of original frame for display
                for d in tracked:
                    x1, y1, x2, y2, _, _, tid, mov = d
                    cv2.rectangle(
                        disp, (int(x1), int(y1)),
                        (int(x2), int(y2)), (0, 255, 0), 2,
                    )
                    cv2.putText(
                        disp, f"ID{tid} M{mov}", (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    )
                cv2.imshow('Stream', disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self._ws and not self._ws.closed:
                await self._ws.close()
            if self._session and not self._session.closed:
                await self._session.close()

    async def close(self) -> None:
        """Close WebSocket and aiohttp session to prevent resource leaks.

        This method properly closes all network connections and cleans up
        resources to prevent memory leaks and thread exceptions.
        """
        try:
            if self._ws:
                if not self._ws.closed:
                    await self._ws.close()
                self._ws = None
        except Exception as e:
            self._logger.error(f"Error closing WebSocket: {e}")
        try:
            if self._session:
                if not self._session.closed:
                    await self._session.close()
                self._session = None
        except Exception as e:
            self._logger.error(f"Error closing session: {e}")

    #######################################################################
    # Post-processing functions
    #######################################################################

    def remove_overlapping_labels(self, datas):
        """
        Removes overlapping labels for Hardhat and Safety Vest categories.

        Args:
            datas (list): A list of detection data in YOLO format.

        Returns:
            list: A list of detection data with overlapping labels removed.
        """
        # Indices of Hardhat detections
        hardhat_indices = [
            i for i, d in enumerate(
                datas,
            ) if d[5] == 0
        ]
        # Indices of NO-Hardhat detections
        no_hardhat_indices = [i for i, d in enumerate(datas) if d[5] == 2]
        # Indices of Safety Vest detections
        safety_vest_indices = [i for i, d in enumerate(datas) if d[5] == 7]
        # Indices of NO-Safety Vest detections
        no_safety_vest_indices = [i for i, d in enumerate(datas) if d[5] == 4]

        to_remove = set()
        for hardhat_index in hardhat_indices:
            for no_hardhat_index in no_hardhat_indices:
                overlap = self.overlap_percentage(
                    datas[hardhat_index][:4], datas[no_hardhat_index][:4],
                )
                if overlap > 0.8:
                    to_remove.add(no_hardhat_index)

        for safety_vest_index in safety_vest_indices:
            for no_safety_vest_index in no_safety_vest_indices:
                overlap = self.overlap_percentage(
                    datas[safety_vest_index][:4],
                    datas[no_safety_vest_index][:4],
                )
                if overlap > 0.8:
                    to_remove.add(no_safety_vest_index)

        for index in sorted(to_remove, reverse=True):
            datas.pop(index)

        gc.collect()
        return datas

    def overlap_percentage(self, bbox1, bbox2):
        """
        Calculates the percentage of overlap between two bounding boxes.

        Args:
            bbox1 (list): The first bounding box [x1, y1, x2, y2].
            bbox2 (list): The second bounding box [x1, y1, x2, y2].

        Returns:
            float: The percentage of overlap between the two bounding boxes.
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
        bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

        overlap_percentage = intersection_area / float(
            bbox1_area + bbox2_area - intersection_area,
        )
        gc.collect()

        return overlap_percentage

    def is_contained(self, inner_bbox, outer_bbox):
        """
        Determines if one bounding box is completely contained within another.

        Args:
            inner_bbox (list): The inner bounding box [x1, y1, x2, y2].
            outer_bbox (list): The outer bounding box [x1, y1, x2, y2].

        Returns:
            bool: Checks if inner box is fully within outer bounding box.
        """
        return (
            inner_bbox[0] >= outer_bbox[0]
            and inner_bbox[2] <= outer_bbox[2]
            and inner_bbox[1] >= outer_bbox[1]
            and inner_bbox[3] <= outer_bbox[3]
        )

    def remove_completely_contained_labels(self, datas):
        """
        Removes labels fully contained in Hardhat/Safety Vest categories.

        Args:
            datas (list): A list of detection data in YOLO format.

        Returns:
            list: Detection data with fully contained labels removed.
        """
        # Indices of Hardhat detections
        hardhat_indices = [
            i
            for i, d in enumerate(
                datas,
            )
            if d[5] == 0
        ]

        # Indices of NO-Hardhat detections
        no_hardhat_indices = [i for i, d in enumerate(datas) if d[5] == 2]

        # Indices of Safety Vest detections
        safety_vest_indices = [i for i, d in enumerate(datas) if d[5] == 7]

        # Indices of NO-Safety Vest detections
        no_safety_vest_indices = [i for i, d in enumerate(datas) if d[5] == 4]

        to_remove = set()
        # Check hardhats
        for hardhat_index in hardhat_indices:
            for no_hardhat_index in no_hardhat_indices:
                if self.is_contained(
                    datas[no_hardhat_index][:4],
                    datas[hardhat_index][:4],
                ):
                    to_remove.add(no_hardhat_index)
                elif self.is_contained(
                    datas[hardhat_index][:4],
                    datas[no_hardhat_index][:4],
                ):
                    to_remove.add(hardhat_index)

        # Check safety vests
        for safety_vest_index in safety_vest_indices:
            for no_safety_vest_index in no_safety_vest_indices:
                if self.is_contained(
                    datas[no_safety_vest_index][:4],
                    datas[safety_vest_index][:4],
                ):
                    to_remove.add(no_safety_vest_index)
                elif self.is_contained(
                    datas[safety_vest_index][:4],
                    datas[no_safety_vest_index][:4],
                ):
                    to_remove.add(safety_vest_index)

        for index in sorted(to_remove, reverse=True):
            datas.pop(index)

        return datas


async def main() -> None:
    """Main execution block for command-line interface.

    Args:
        None
    """
    parser = argparse.ArgumentParser(
        description='Live stream detection with WebSocket support',
    )
    parser.add_argument(
        '--url', type=str, required=True,
        help='Stream URL or video file path',
    )
    parser.add_argument(
        '--api_url', type=str,
        default=os.getenv('DETECT_API_URL', 'http://localhost:8000'),
        help='Base API URL for remote inference',
    )
    parser.add_argument(
        '--model_key', type=str,
        default='yolo11n', help='YOLO model identifier key',
    )
    parser.add_argument(
        '--detect_with_server',
        action='store_true', help='Enable remote WebSocket inference',
    )
    parser.add_argument(
        '--use_ultralytics', action='store_true',
        help='Use Ultralytics YOLO for local inference',
    )
    args = parser.parse_args()

    # Initialise shared token dictionary for authentication
    shared_token: SharedToken = SharedToken(
        access_token='', refresh_token='', is_refreshing=False,
    )

    # Create detector instance with parsed arguments
    detector = LiveStreamDetector(
        api_url=args.api_url,
        model_key=args.model_key,
        detect_with_server=args.detect_with_server,
        shared_token=shared_token,
        use_ultralytics=args.use_ultralytics,
    )

    # Run detection loop
    asyncio.run(detector.run_detection(args.url))  # pragma: no cover

if __name__ == '__main__':
    asyncio.run(main())  # pragma: no cover
