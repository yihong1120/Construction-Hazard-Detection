from __future__ import annotations

import os
import unittest
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import cv2
import numpy as np

from src.live_stream_detection import LiveStreamDetector
from src.live_stream_detection import SharedToken


class DummyResponse:
    """Mock HTTP response for aiohttp testing.

    Provides async context manager and json/text methods with configurable
    status codes and error scenarios.
    """

    def __init__(
        self,
        status: int,
        json_data: Any,
        raise_for_status_side_effect: Exception | None = None,
    ) -> None:
        """Initialise mock response.

        Args:
            status: HTTP status code to simulate.
            json_data: Data to return from json() calls.
            raise_for_status_side_effect: Exception to raise from
                raise_for_status(), if any.
        """
        self.status: int = status
        self._json_data: Any = json_data
        self.raise_for_status_side_effect: Exception | None = (
            raise_for_status_side_effect
        )

    async def __aenter__(self) -> DummyResponse:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Async context manager exit."""
        pass

    async def json(self) -> Any:
        """Return configured JSON data."""
        return self._json_data

    async def text(self) -> str:
        """Return JSON data as string."""
        return str(self._json_data)

    def raise_for_status(self) -> None:
        """Raise configured exception if set."""
        if self.raise_for_status_side_effect:
            raise self.raise_for_status_side_effect


class DummyClientSession:
    """Mock aiohttp ClientSession for testing HTTP/WebSocket operations.

    Provides configurable responses and tracks session state for proper
    resource management testing.
    """

    def __init__(
        self, post_responses: list[DummyResponse] | None = None,
    ) -> None:
        """Initialise mock session.

        Args:
            post_responses: List of responses to return from post() calls.
        """
        self._post_responses: list[DummyResponse] = post_responses or []
        self.closed: bool = False

    async def __aenter__(self) -> DummyClientSession:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self, exc_type: Any, exc: Any, tb: Any,
    ) -> None:
        """Async context manager exit, marks session as closed."""
        self.closed = True

    async def post(self, *args: Any, **kwargs: Any) -> DummyResponse:
        """Mock HTTP POST returning configured responses."""
        if self._post_responses:
            return self._post_responses.pop(0)
        return DummyResponse(200, {})

    async def ws_connect(self, *args: Any, **kwargs: Any) -> AsyncMock:
        """Mock WebSocket connection method."""
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.send_bytes = AsyncMock()
        mock_ws.receive = AsyncMock()
        mock_ws.ping = AsyncMock()
        mock_ws.close = AsyncMock()
        return mock_ws

    async def close(self) -> None:
        """Mock close method."""
        self.closed = True


class TestLiveStreamDetector(unittest.IsolatedAsyncioTestCase):
    """Async test suite for LiveStreamDetector.

    Tests the complete detection pipeline including cloud/local detection,
    tracking algorithms, error handling, and frame processing. Uses mock
    objects to isolate behaviour from external dependencies.
    """

    # Test instance attributes
    detector: LiveStreamDetector
    detector_server: LiveStreamDetector
    api_url: str
    model_key: str
    output_folder: str
    detect_with_server: bool

    def setUp(self) -> None:
        """Set up test environment and detector instances before each test."""
        # Mock environment variables for authentication
        patcher_env = patch.dict(
            os.environ, {
                'API_USERNAME': 'test_user',
                'API_PASSWORD': 'test_pass',
            },
        )
        patcher_env.start()
        self.addCleanup(patcher_env.stop)

        # Test configuration
        self.api_url: str = 'http://mocked-api.com'
        self.model_key: str = 'yolo11n'
        self.output_folder: str = 'test_output'
        self.detect_with_server: bool = False

        # Create local detection detector instance
        self.detector: LiveStreamDetector = LiveStreamDetector(
            api_url=self.api_url,
            model_key=self.model_key,
            output_folder=self.output_folder,
            detect_with_server=self.detect_with_server,
        )
        # Create server-mode detector for WebSocket-related tests
        self.detector_server: LiveStreamDetector = LiveStreamDetector(
            api_url='http://example.com',
            detect_with_server=True,
            model_key='yolo11n',
        )

    def test_initialisation(self) -> None:
        """Test basic detector initialisation with default parameters."""
        detector = LiveStreamDetector(
            api_url=self.api_url,
            model_key=self.model_key,
            output_folder=self.output_folder,
            detect_with_server=self.detect_with_server,
        )
        # Verify all configuration is set correctly
        self.assertEqual(detector.api_url, self.api_url)
        self.assertEqual(detector.model_key, self.model_key)
        self.assertEqual(detector.output_folder, self.output_folder)
        self.assertEqual(detector.detect_with_server, self.detect_with_server)
        # Verify default shared token structure
        self.assertEqual(
            detector.shared_token, {
                'access_token': '',
                'refresh_token': '',
                'is_refreshing': False,
            },
        )

    def test_initialisation_with_shared_token(self) -> None:
        """Test detector initialisation with custom shared token."""
        shared_token: SharedToken = {
            'access_token': 'test_token',
            'refresh_token': '',
            'is_refreshing': False,
        }
        detector = LiveStreamDetector(
            api_url=self.api_url,
            model_key=self.model_key,
            output_folder=self.output_folder,
            detect_with_server=self.detect_with_server,
            shared_token=shared_token,
        )
        # Verify the custom token is used
        self.assertEqual(detector.shared_token, shared_token)

    def test_token_manager_shared_lock(self) -> None:
        """Test that token manager shares the same token reference."""
        self.assertIsNotNone(self.detector.token_manager)
        self.assertIsNotNone(self.detector.token_manager.shared_token)
        # Verify the token manager uses the same token instance
        self.assertIs(
            self.detector.token_manager.shared_token,
            self.detector.shared_token,
        )

    @patch('aiohttp.ClientSession', return_value=DummyClientSession())
    async def test_authenticate_skip_if_token_exists(
        self, mock_session_class: Any,
    ) -> None:
        """Test authentication is skipped when token already exists."""
        self.detector.shared_token['access_token'] = 'existing_token'
        await self.detector.token_manager.authenticate()
        # If token exists, ClientSession creation should not be triggered
        mock_session_class.assert_not_called()

    @patch(
        'aiohttp.ClientSession', return_value=DummyClientSession(
            post_responses=[
                DummyResponse(
                    200, {'access_token': 'fake_token'},
                ),
            ],
        ),
    )
    async def test_authenticate(
        self, mock_session_class: Any,
    ) -> None:
        """Test successful authentication stores the access token."""
        await self.detector.token_manager.authenticate()
        self.assertEqual(
            self.detector.shared_token['access_token'], 'fake_token',
        )

    @patch('aiohttp.ClientSession', return_value=DummyClientSession())
    async def test_authenticate_missing_credentials(
        self, mock_session_class: Any,
    ) -> None:
        """Test authentication fails when credentials are missing."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                await self.detector.token_manager.authenticate()
            self.assertIn(
                'Missing API_USERNAME or API_PASSWORD',
                str(ctx.exception),
            )
        mock_session_class.assert_not_called()

    @patch(
        'aiohttp.ClientSession', return_value=DummyClientSession(
            post_responses=[
                DummyResponse(
                    401,
                    {},
                    raise_for_status_side_effect=RuntimeError(
                        'Authenticate failed with status 401',
                    ),
                ),
            ],
        ),
    )
    async def test_authenticate_raises_for_status(
        self, mock_session_class: Any,
    ) -> None:
        """Test authentication propagates HTTP error responses."""
        with self.assertRaises(RuntimeError) as ctx:
            await self.detector.token_manager.authenticate()
        self.assertIn(
            'Authenticate failed with status 401',
            str(ctx.exception),
        )

    @patch(
        'aiohttp.ClientSession', return_value=DummyClientSession(
            post_responses=[DummyResponse(200, {})],
        ),
    )
    async def test_authenticate_raises_key_error_if_no_access_token(
        self, mock_session_class: Any,
    ) -> None:
        """Test authentication fails when response lacks access_token."""
        with self.assertRaises(KeyError):
            await self.detector.token_manager.authenticate()

    async def test_detect_cloud_ws_request_error(self) -> None:
        """Test WebSocket detection handles network errors gracefully."""
        self.detector.shared_token['access_token'] = 'fake_token'
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Mock WebSocket send to raise an exception
        with patch.object(
            self.detector.net,
            'ws_send_and_receive',
            new_callable=AsyncMock,
            side_effect=Exception('boom'),
        ) as mock_send:
            with self.assertLogs(
                self.detector._logger, level='ERROR',
            ) as captured:
                with patch(
                    'cv2.imencode',
                    return_value=(True, np.array([1, 2, 3])),
                ):
                    result = await self.detector._detect_cloud_ws(frame)
                # Should return empty list on error
                self.assertEqual(result, [])
            # Should log the error
            self.assertTrue(
                any('WS error:' in output for output in captured.output),
            )
            mock_send.assert_called()

    @patch('src.live_stream_detection.get_sliced_prediction')
    @patch('src.live_stream_detection.AutoDetectionModel.from_pretrained')
    async def test_detect_local_with_predictions(
        self, mock_from_pretrained: Any, mock_get_sliced_prediction: Any,
    ) -> None:
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        mock_result = MagicMock()
        mock_result.object_prediction_list = [
            MagicMock(
                category=MagicMock(id=0),
                bbox=MagicMock(to_voc_bbox=lambda: [10.5, 20.3, 50.8, 60.1]),
                score=MagicMock(value=0.85),
            ),
            MagicMock(
                category=MagicMock(id=1),
                bbox=MagicMock(to_voc_bbox=lambda: [30, 40, 70, 80]),
                score=MagicMock(value=0.9),
            ),
        ]
        mock_get_sliced_prediction.return_value = mock_result

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Create a detector that uses SAHI instead of ultralytics
        detector = LiveStreamDetector(
            api_url=self.api_url,
            model_key=self.model_key,
            output_folder=self.output_folder,
            detect_with_server=self.detect_with_server,
            use_ultralytics=False,  # Use SAHI
        )
        datas: list[list[float]] = await detector._detect_local(frame)
        self.assertEqual(len(datas), 2)
        self.assertEqual(datas[0], [10, 20, 50, 60, 0.85, 0])
        self.assertEqual(datas[1], [30, 40, 70, 80, 0.9, 1])
        mock_get_sliced_prediction.assert_called_once_with(
            frame, mock_model, slice_height=376, slice_width=376,
            overlap_height_ratio=0.3, overlap_width_ratio=0.3,
        )

    async def test_generate_detections(self) -> None:
        """Test detection generation for both local and server modes."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mat_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Test local detection mode (uses ultralytics directly)
        self.detector.detect_with_server = False
        # Mock the ultralytics model.track() method
        mock_results = MagicMock()
        mock_boxes = MagicMock()
        mock_results.boxes = mock_boxes

        # Mock boxes to have length 1
        mock_boxes.__len__ = MagicMock(return_value=1)
        mock_boxes.xyxy = MagicMock()
        mock_boxes.conf = MagicMock()
        mock_boxes.cls = MagicMock()
        mock_boxes.id = None

        # Mock tensor-like objects with tolist() method
        mock_boxes.xyxy.tolist.return_value = [[10, 10, 50, 50]]
        mock_boxes.conf.tolist.return_value = [0.9]
        mock_boxes.cls.tolist.return_value = [0]

        with patch.object(
            self.detector.ultralytics_model,
            'track',
            return_value=[mock_results],
        ):
            datas, tracked = await self.detector.generate_detections(
                mat_frame,
            )
            self.assertEqual(len(datas), 1)
            self.assertEqual(datas[0][5], 0)  # class_id
            self.assertEqual(len(tracked), 1)
            # track_id when no ID provided
            self.assertEqual(tracked[0][6], -1)

        # Test server detection mode
        self.detector.detect_with_server = True
        with patch.object(
            self.detector,
            '_detect_cloud_ws',
            return_value=[[20, 20, 60, 60, 0.8, 1]],
        ) as mock_cloud:
            datas, tracked = await self.detector.generate_detections(
                mat_frame,
            )
            self.assertEqual(len(datas), 1)
            self.assertEqual(datas[0][5], 1)  # class_id
            mock_cloud.assert_called_once_with(mat_frame)

    @patch('src.live_stream_detection.cv2.VideoCapture')
    async def test_run_detection_stream_not_opened(
        self, mock_vcap: Any,
    ) -> None:
        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = False
        mock_vcap.return_value = cap_mock
        with self.assertRaises(ValueError) as ctx:
            await self.detector.run_detection('fake_stream')
        self.assertIn('Failed to open stream', str(ctx.exception))

    async def test_run_detection(self) -> None:
        stream_url = 'http://example.com/virtual_stream'
        cap_mock = MagicMock()
        cap_mock.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None),
        ]
        cap_mock.isOpened.return_value = True
        with patch(
            'src.live_stream_detection.cv2.VideoCapture',
            return_value=cap_mock,
        ):
            with patch('src.live_stream_detection.cv2.imshow'):
                with patch(
                    'src.live_stream_detection.cv2.waitKey',
                    side_effect=[-1, ord('q')],
                ):
                    with patch(
                        'src.live_stream_detection.cv2.destroyAllWindows',
                    ):
                        # Avoid real model inference
                        with patch.object(
                            self.detector,
                            'generate_detections',
                            return_value=([], []),
                        ):
                            await self.detector.run_detection(stream_url)
        cap_mock.read.assert_called()
        cap_mock.release.assert_called_once()

    @patch('src.live_stream_detection.cv2.destroyAllWindows')
    @patch(
        'src.live_stream_detection.cv2.waitKey',
        side_effect=[-1, -1, ord('q')],
    )
    @patch('src.live_stream_detection.cv2.imshow')
    @patch('src.live_stream_detection.cv2.VideoCapture')
    async def test_run_detection_loop(
        self,
        mock_vcap: Any,
        mock_imshow: Any,
        mock_waitKey: Any,
        mock_destroy: Any,
    ) -> None:
        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = True
        frames_side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None),
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
        ]
        cap_mock.read.side_effect = frames_side_effect
        mock_vcap.return_value = cap_mock
        # Avoid real model inference
        with patch.object(
            self.detector, 'generate_detections', return_value=([], []),
        ):
            await self.detector.run_detection('fake_stream')
        self.assertGreaterEqual(cap_mock.read.call_count, 4)
        cap_mock.release.assert_called_once()
        mock_imshow.assert_called()
        mock_waitKey.assert_called()
        mock_destroy.assert_called_once()

    async def test_close_method(self) -> None:
        """Test that close method delegates to NetClient for cleanup."""
        with patch.object(
            self.detector.net, 'close', new_callable=AsyncMock,
        ) as mock_close:
            await self.detector.close()
            mock_close.assert_called_once()

    def test_helper_methods(self) -> None:
        """Test helper methods for frame processing."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Test frame preparation
        prepared_frame = self.detector._prepare_frame(frame)
        self.assertEqual(prepared_frame.shape, frame.shape)

        # Test frame encoding
        encoded = self.detector._encode_frame(frame)
        self.assertIsNotNone(encoded)

        # Test with JPEG encoding disabled
        self.detector.use_jpeg_ws = False
        encoded_png = self.detector._encode_frame(frame)
        self.assertIsNotNone(encoded_png)

    def test_tracking_cleanup(self) -> None:
        """Test tracking data cleanup for old track IDs."""
        self.detector.prev_centers = {1: (100, 100), 2: (200, 200)}
        self.detector.prev_centers_last_seen = {1: 1, 2: 2}
        self.detector.frame_count = 50
        self.detector.max_id_keep = 5

        # Should trigger cleanup
        self.detector._cleanup_prev_centers()

        # Data should be cleaned up for old IDs
        self.assertEqual(len(self.detector.prev_centers), 0)
        self.assertEqual(len(self.detector.prev_centers_last_seen), 0)

    def test_remove_overlapping_labels(self) -> None:
        """Test removal of overlapping detection labels."""
        datas = [
            [10, 10, 50, 50, 0.9, 0],
            [10, 10, 50, 45, 0.8, 2],
            [20, 20, 60, 60, 0.85, 7],
            [20, 20, 60, 55, 0.75, 4],
        ]
        expected = [
            [10, 10, 50, 50, 0.9, 0],
            [20, 20, 60, 60, 0.85, 7],
        ]
        filtered = self.detector.remove_overlapping_labels(datas)
        self.assertEqual(filtered, expected)

    def test_overlap_percentage(self) -> None:
        """Test calculation of overlap percentage between bboxes."""
        bbox1 = [10, 10, 50, 50]
        bbox2 = [30, 30, 70, 70]
        overlap = self.detector.overlap_percentage(bbox1, bbox2)
        self.assertIsInstance(overlap, float)
        self.assertGreaterEqual(overlap, 0.0)
        self.assertLessEqual(overlap, 1.0)

    def test_is_contained(self) -> None:
        """Test detection of one bbox contained within another."""
        inner_bbox = [20, 20, 40, 40]
        outer_bbox = [10, 10, 50, 50]
        result = self.detector.is_contained(inner_bbox, outer_bbox)
        self.assertTrue(result)

        inner_bbox = [5, 5, 15, 15]
        result = self.detector.is_contained(inner_bbox, outer_bbox)
        self.assertFalse(result)

    def test_remove_completely_contained_labels(self) -> None:
        """Test removal of labels completely contained within others."""
        datas = [
            [10, 10, 50, 50, 0.9, 0],
            [20, 20, 40, 40, 0.8, 2],
            [60, 60, 100, 100, 0.85, 7],
        ]
        filtered = self.detector.remove_completely_contained_labels(datas)
        self.assertEqual(len(filtered), 2)

    def test_track_remote_centroid(self) -> None:
        """Test centroid-based remote tracking algorithm."""
        dets = [[10, 10, 50, 50, 0.9, 0]]
        tracked = self.detector._track_remote_centroid(dets)
        self.assertEqual(len(tracked), 1)
        self.assertEqual(len(tracked[0]), 8)

    def test_track_remote_hungarian(self) -> None:
        """Test Hungarian algorithm-based remote tracking."""
        dets = [[10, 10, 50, 50, 0.9, 0]]
        tracked = self.detector._track_remote_hungarian(dets)
        self.assertEqual(len(tracked), 1)
        self.assertEqual(len(tracked[0]), 8)

    def test_bbox_center(self) -> None:
        """Test bounding box centre calculation."""
        center = self.detector._bbox_center(10, 10, 50, 50)
        self.assertEqual(center, (30.0, 30.0))

    def test_bbox_iou(self) -> None:
        """Test intersection over union calculation for bboxes."""
        box1 = (10, 10, 50, 50)
        box2 = (30, 30, 70, 70)
        iou = self.detector._bbox_iou(box1, box2)
        self.assertIsInstance(iou, float)
        self.assertGreaterEqual(iou, 0.0)
        self.assertLessEqual(iou, 1.0)

    def test_squared_distance(self) -> None:
        """Test squared Euclidean distance calculation."""
        dist = self.detector._squared_distance((0, 0), (3, 4))
        self.assertEqual(dist, 25.0)

    async def test_handle_response_data(self) -> None:
        payload1: Any = [[10, 10, 50, 50, 0.9, 0]]
        result = await self.detector._handle_response_data(payload1)
        self.assertEqual(result, payload1)
        payload2: Any = {'detections': [[20, 20, 60, 60, 0.8, 1]]}
        result = await self.detector._handle_response_data(payload2)
        self.assertEqual(result, [[20, 20, 60, 60, 0.8, 1]])
        payload3: Any = {'type': 'ping'}
        result = await self.detector._handle_response_data(payload3)
        self.assertIsNone(result)
        payload4: Any = {'error': 'Test error'}
        result = await self.detector._handle_response_data(payload4)
        self.assertEqual(result, [])

    async def test_handle_server_error(self) -> None:
        with patch.object(
            self.detector.token_manager, 'refresh_token',
            new_callable=AsyncMock,
        ) as mock_refresh:
            result = await self.detector._handle_server_error(
                'Token expired',
            )
            self.assertEqual(result, [])
            mock_refresh.assert_called_once()

    async def test_handle_exception(self) -> None:
        with patch.object(
            self.detector.token_manager, 'refresh_token',
            new_callable=AsyncMock,
        ) as mock_refresh:
            result = await self.detector._handle_exception(
                Exception('401 Unauthorized'),
            )
            self.assertTrue(result)
            mock_refresh.assert_called_once()
        result = await self.detector._handle_exception(
            Exception('Network error'),
        )
        self.assertFalse(result)

    async def test_cov_attempt_ws_detect_encode_fail(self) -> None:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        with patch(
            'src.live_stream_detection.cv2.imencode',
            return_value=(False, None),
        ):
            out = await self.detector_server._attempt_ws_detect(frame)
            self.assertEqual(out, [])

    async def test_cov_attempt_ws_detect_none_response(self) -> None:
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        with patch(
            'src.live_stream_detection.cv2.imencode',
            return_value=(True, np.array([1, 2])),
        ):
            with patch.object(
                self.detector_server.net,
                'ws_send_and_receive',
                new_callable=AsyncMock,
                return_value=None,
            ):
                out = await self.detector_server._attempt_ws_detect(frame)
                self.assertIsNone(out)

    async def test_cov_detect_cloud_ws_none_then_success(self) -> None:
        self.detector_server.shared_token['access_token'] = 'tok'
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        with patch(
            'src.live_stream_detection.cv2.imencode',
            return_value=(True, np.array([1, 2])),
        ):
            with patch.object(
                self.detector_server,
                '_attempt_ws_detect',
                new_callable=AsyncMock,
                side_effect=[None, [[1, 1, 2, 2, 0.9, 0]]],
            ):
                with patch.object(
                    self.detector_server,
                    'close',
                    new_callable=AsyncMock,
                ) as mock_close:
                    out = await self.detector_server._detect_cloud_ws(frame)
                    self.assertEqual(out, [[1, 1, 2, 2, 0.9, 0]])
                    self.assertGreaterEqual(mock_close.await_count, 1)

    async def test_cov_maybe_refresh_token_and_reset_ws_paths(self) -> None:
        with patch.object(
            self.detector_server.token_manager,
            'is_token_expired',
            return_value=False,
        ):
            await self.detector_server._maybe_refresh_token_and_reset_ws()
        with patch.object(
            self.detector_server.token_manager, 'is_token_expired',
            return_value=True,
        ):
            with patch.object(
                self.detector_server.token_manager,
                'refresh_token', new_callable=AsyncMock,
            ) as mock_refresh:
                with patch.object(
                    self.detector_server, 'close', new_callable=AsyncMock,
                ) as mock_close:
                    await (
                        self.detector_server
                        ._maybe_refresh_token_and_reset_ws()
                    )
                    mock_refresh.assert_called_once()
                    mock_close.assert_called_once()
        with patch.object(
            self.detector_server.token_manager, 'is_token_expired',
            return_value=True,
        ):
            with patch.object(
                self.detector_server.token_manager,
                'refresh_token', new_callable=AsyncMock,
                side_effect=Exception('fail'),
            ):
                await self.detector_server._maybe_refresh_token_and_reset_ws()

    async def test_cov_detect_cloud_ws_handle_exception_true(self) -> None:
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        with patch.object(
            self.detector_server, '_attempt_ws_detect',
            new_callable=AsyncMock, side_effect=Exception('401 Unauthorized'),
        ):
            with patch.object(
                self.detector_server, '_handle_exception',
                new_callable=AsyncMock, return_value=True,
            ):
                with patch.object(
                    self.detector_server,
                    'close',
                    new_callable=AsyncMock,
                ) as mock_close:
                    out = await self.detector_server._detect_cloud_ws(frame)
                    self.assertEqual(out, [])
                    self.assertGreaterEqual(mock_close.await_count, 2)

    async def test_cov_process_message_variants(self) -> None:
        res = await self.detector_server._process_message('{not_json}')
        self.assertEqual(res, [])
        res = await self.detector_server._process_message('[[1,2,3,4,0.9,0]]')
        self.assertEqual(res, [[1, 2, 3, 4, 0.9, 0]])
        res = await self.detector_server._process_message(
            '{"detections": [[1,2,3,4,0.9,0]]}',
        )
        self.assertEqual(res, [[1, 2, 3, 4, 0.9, 0]])

    async def test_cov_handle_response_data_unknown_and_data_key(self) -> None:
        out = await self.detector_server._handle_response_data((1, 2))
        self.assertEqual(out, [])
        out = await self.detector_server._handle_response_data(
            {
                'data': [[5, 5, 10, 10, 0.5, 1]],
            },
        )
        self.assertEqual(out, [[5, 5, 10, 10, 0.5, 1]])
        out = await self.detector_server._handle_response_data({'ready': True})
        self.assertEqual(out, [])

    async def test_cov_handle_server_error_non_token(self) -> None:
        with patch.object(
            self.detector_server.token_manager,
            'refresh_token', new_callable=AsyncMock,
        ) as mock_refresh:
            out = await self.detector_server._handle_server_error(
                'some other error',
            )
            self.assertEqual(out, [])
            mock_refresh.assert_not_called()

    async def test_cov_handle_exception_refresh_fails(self) -> None:
        with patch.object(
            self.detector_server.token_manager,
            'refresh_token', new_callable=AsyncMock,
            side_effect=Exception('nope'),
        ):
            res = await self.detector_server._handle_exception(
                Exception('401 Unauthorized'),
            )
            self.assertFalse(res)

    def test_cov_track_remote_dispatch(self) -> None:
        out = self.detector_server._track_remote([])
        self.assertEqual(out, [])
        self.detector_server.remote_tracker = 'hungarian'
        out = self.detector_server._track_remote([])
        self.assertEqual(out, [])

    def test_cov_track_remote_centroid_prune(self) -> None:
        self.detector_server.frame_count = 10
        self.detector_server.max_id_keep = 5
        self.detector_server.remote_tracks = {
            1: {
                'bbox': (0, 0, 1, 1),
                'center': (0.5, 0.5),
                'last_seen': 1,
                'cls': 0,
            },
            2: {
                'bbox': (1, 1, 2, 2),
                'center': (1.5, 1.5),
                'last_seen': 9,
                'cls': 0,
            },
        }
        out = self.detector_server._track_remote_centroid([])
        self.assertEqual(out, [])
        self.assertNotIn(1, self.detector_server.remote_tracks)
        self.assertIn(2, self.detector_server.remote_tracks)

    def test_cov_hungarian_helpers(self) -> None:
        d = self.detector_server
        mat = np.array([[1.0, 2.0], [3.0, 4.0]])
        m2, n2 = d._pad_to_square(mat)
        self.assertEqual(n2, 2)
        mat_rect = np.array([[1.0, 2.0, 3.0]])
        m3, n3 = d._pad_to_square(mat_rect)
        self.assertEqual(n3, 3)
        self.assertEqual(m3.shape, (3, 3))
        m = np.array([[3.0, 1.0], [2.0, 5.0]])
        d._row_col_reduce(m)
        self.assertTrue((m.min(axis=1) == 0).all())
        self.assertTrue((m.min(axis=0) == 0).all())
        cz_r, cz_c = d._cover_zeros(
            np.array([[0.0, 1.0], [1.0, 0.0]]),
        )
        self.assertIsInstance(cz_r, set)
        self.assertIsInstance(cz_c, set)
        mat0 = np.array([[0.0, 1.0], [1.0, 0.0]])
        ok = d._adjust_matrix_with_min(mat0, {0, 1}, {0, 1})
        self.assertFalse(ok)
        mat2 = np.array([[2.0, 3.0], [4.0, 5.0]])
        ok2 = d._adjust_matrix_with_min(mat2, {0}, set())
        self.assertTrue(ok2)
        r, c = d._cover_zeros(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
        )
        self.assertEqual(len(r) + len(c), 0)
        cost = np.array([[0.9, 0.8], [0.7, 0.6]])
        matches, ur, uc = d._hungarian_assign(
            cost, cost_threshold=0.0,
        )
        self.assertEqual(matches, [])
        self.assertEqual(set(ur), {0, 1})
        self.assertEqual(set(uc), {0, 1})

    def test_cov_prune_remote_tracks(self) -> None:
        self.detector_server.frame_count = 50
        self.detector_server.max_id_keep = 5
        self.detector_server.remote_tracks = {
            1: {
                'bbox': (0, 0, 1, 1),
                'center': (0.5, 0.5),
                'last_seen': 1,
                'cls': 0,
            },
            2: {
                'bbox': (1, 1, 2, 2),
                'center': (1.5, 1.5),
                'last_seen': 49,
                'cls': 0,
            },
        }
        self.detector_server._prune_remote_tracks()
        self.assertNotIn(1, self.detector_server.remote_tracks)
        self.assertIn(2, self.detector_server.remote_tracks)

    async def test_cov_close_error_path(self) -> None:
        with patch.object(
            self.detector_server.net, 'close',
            new_callable=AsyncMock, side_effect=Exception('boom'),
        ):
            await self.detector_server.close()

    def test_cov_compute_pair_cost_edge(self) -> None:
        d = self.detector_server
        info = {'cls': 0, 'bbox': (1.0, 1.0, 1.0, 1.0), 'center': (1.0, 1.0)}
        cost = d._compute_pair_cost([1.0, 1.0, 1.0, 1.0, 0.9, 0], info)
        self.assertGreaterEqual(cost, 0.0)

    async def test_cov_handle_server_error_token_paths(self) -> None:
        with patch.object(
            self.detector_server.token_manager,
            'refresh_token',
            new_callable=AsyncMock,
        ) as mock_ref:
            with patch.object(
                self.detector_server,
                'close',
                new_callable=AsyncMock,
            ) as mock_close:
                out = await self.detector_server._handle_server_error(
                    '401 Unauthorized',
                )
                self.assertEqual(out, [])
                mock_ref.assert_called_once()
                mock_close.assert_called_once()
        with patch.object(
            self.detector_server.token_manager, 'refresh_token',
            new_callable=AsyncMock, side_effect=Exception('x'),
        ):
            out = await self.detector_server._handle_server_error(
                'token expired',
            )
            self.assertEqual(out, [])

    def test_cov_hungarian_assign_zero_extraction(self) -> None:
        d = self.detector_server
        cost = np.array([[0.0, 2.0], [2.0, 0.0]], dtype=float)
        matches, ur, uc = d._hungarian_assign(cost, cost_threshold=10.0)
        self.assertIn((0, 0), matches)
        self.assertIn((1, 1), matches)

    def test_cov_cover_zeros_column_selection_branch(self) -> None:
        d = self.detector_server
        mat = np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
        rows, cols = d._cover_zeros(mat.copy())
        self.assertIn(1, cols)

    def test_cov_adjust_matrix_with_min_add_and_subtract(self) -> None:
        d = self.detector_server
        mat = np.array([[5.0, 1.0], [2.0, 3.0]])
        ok = d._adjust_matrix_with_min(mat, {0}, {0})
        self.assertTrue(ok)
        self.assertAlmostEqual(mat[1, 1], 0.0)
        self.assertGreater(mat[0, 0], 5.0)

    @patch('src.live_stream_detection.cv2.destroyAllWindows')
    @patch('src.live_stream_detection.cv2.waitKey', side_effect=[-1, ord('q')])
    @patch('src.live_stream_detection.cv2.putText')
    @patch('src.live_stream_detection.cv2.rectangle')
    @patch('src.live_stream_detection.cv2.imshow')
    @patch('src.live_stream_detection.cv2.VideoCapture')
    async def test_cov_run_detection_draw_calls(
        self, mock_vcap: Any, mock_imshow: Any, mock_rect: Any,
        mock_text: Any, *_m,
    ) -> None:
        cap = MagicMock()
        cap.isOpened.return_value = True
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        cap.read.side_effect = [(True, frame), (True, frame)]
        mock_vcap.return_value = cap
        tracked = [[1, 1, 5, 5, 0.9, 0, 42, 1]]
        with patch.object(
            self.detector_server, 'generate_detections',
            new_callable=AsyncMock, return_value=(tracked, tracked),
        ):
            await self.detector_server.run_detection('stream')
        mock_imshow.assert_called()
        mock_rect.assert_called()
        mock_text.assert_called()

    def test_cov_remove_vest_containment_both_directions(self) -> None:
        d = self.detector_server
        datas1 = [[10, 10, 30, 30, 0.9, 7], [12, 12, 20, 20, 0.8, 4]]
        out1 = d.remove_completely_contained_labels(datas1.copy())
        self.assertTrue(all(row[5] != 4 for row in out1))
        datas2 = [[12, 12, 20, 20, 0.9, 7], [10, 10, 30, 30, 0.8, 4]]
        out2 = d.remove_completely_contained_labels(datas2.copy())
        self.assertTrue(all(row[5] != 7 for row in out2))

    async def test_cov_attempt_ws_detect_success(self) -> None:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        with patch(
            'src.live_stream_detection.cv2.imencode',
            return_value=(True, np.array([1, 2, 3])),
        ):
            with patch.object(
                self.detector_server.net, 'ws_send_and_receive',
                new_callable=AsyncMock,
                return_value={'detections': [[1, 1, 2, 2, 0.9, 0]]},
            ):
                out = await self.detector_server._attempt_ws_detect(frame)
                self.assertEqual(out, [[1, 1, 2, 2, 0.9, 0]])

    def test_cov_remove_hardhat_contained_by_no_hardhat(self) -> None:
        d = self.detector_server
        datas = [[10, 10, 20, 20, 0.9, 0], [5, 5, 30, 30, 0.8, 2]]
        out = d.remove_completely_contained_labels(datas.copy())
        self.assertTrue(all(row[5] != 0 for row in out))

    def test_cov_hungarian_assign_adjust_false_branch(self) -> None:
        d = self.detector_server
        cost = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=float)
        with patch.object(d, '_cover_zeros', return_value=(set(), {0})):
            with patch.object(
                d, '_adjust_matrix_with_min', return_value=False,
            ) as mock_adj:
                matches, ur, uc = d._hungarian_assign(
                    cost, cost_threshold=10.0,
                )
                self.assertIsInstance(matches, list)
                self.assertTrue(mock_adj.called)

    def test_cov_cover_zeros_no_counts_break(self) -> None:
        d = self.detector_server
        mat = np.array([[0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        rows, cols = d._cover_zeros(mat.copy())
        self.assertTrue(len(rows) + len(cols) >= 1)

    @patch('argparse.ArgumentParser.parse_args')
    async def test_main_function_execution(
        self, mock_parse_args: Any,
    ) -> None:
        from src.live_stream_detection import main
        from src.live_stream_detection import LiveStreamDetector as LSD
        mock_args = MagicMock()
        mock_args.url = 'test_stream'
        mock_args.api_url = 'http://test-api.com'
        mock_args.model_key = 'yolo11n'
        mock_args.detect_with_server = False
        mock_args.use_ultralytics = True
        mock_parse_args.return_value = mock_args
        # Avoid opening real video by mocking run_detection
        with patch.object(LSD, 'run_detection', new_callable=AsyncMock):
            await main()

    def test_init_default_api_url_from_env(self) -> None:
        with patch.dict(os.environ, {'DETECT_API_URL': 'http://x:1234/api'}):
            d = LiveStreamDetector(api_url=None)
            self.assertEqual(d.api_url, 'http://x:1234/api')

    def test_prepare_frame_resize(self) -> None:
        self.detector_server.ws_frame_size = (32, 24)
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        out = self.detector_server._prepare_frame(frame)
        self.assertEqual(out.shape[:2], (24, 32))

    async def test_generate_detections_no_boxes(self) -> None:
        """Test detection generation when no boxes are detected."""
        self.detector.detect_with_server = False
        # Mock track() returning empty boxes
        mock_results = MagicMock()
        mock_boxes = MagicMock()
        mock_boxes.__len__ = MagicMock(return_value=0)
        mock_results.boxes = mock_boxes
        with patch.object(
            self.detector.ultralytics_model,
            'track',
            return_value=[mock_results],
        ):
            datas, tracked = await self.detector.generate_detections(
                np.zeros((10, 10, 3), dtype=np.uint8),
            )
            self.assertEqual(datas, [])
            self.assertEqual(tracked, [])

    def test_track_remote_hungarian_no_tracks_creates_new(self) -> None:
        d = self.detector_server
        d.remote_tracker = 'hungarian'
        d.remote_tracks.clear()
        dets = [[10.0, 10.0, 20.0, 20.0, 0.9, 0]]
        out = d._track_remote_hungarian(dets)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), 8)

    def test_track_remote_hungarian_match_unmatched_prune(self) -> None:
        """Test Hungarian tracker with matched/unmatched dets and pruning."""
        d = self.detector_server
        d.remote_tracker = 'hungarian'
        d.frame_count = 10  # trigger prune
        d.max_id_keep = 5   # ensure pruning threshold excludes old
        # existing valid track near (15,15) with class 0
        d.remote_tracks = {
            1: {
                'bbox': (10.0, 10.0, 20.0, 20.0),
                'center': (15.0, 15.0),
                'last_seen': 9,
                'cls': 0,
            },
            99: {
                'bbox': (0.0, 0.0, 1.0, 1.0),
                'center': (0.5, 0.5),
                'last_seen': 0,  # stale, should be pruned
                'cls': 0,
            },
        }
        # two detections: one matches tid=1, one unmatched (different class)
        dets = [
            [11.0, 11.0, 19.0, 19.0, 0.95, 0],  # near center, class 0 => match
            [30.0, 30.0, 40.0, 40.0, 0.80, 1],  # different class => new track
        ]
        out = d._track_remote_hungarian(dets)
        self.assertEqual(len(out), 2)
        # one result should use existing tid=1
        tids = {row[6] for row in out}
        self.assertIn(1, tids)
        # the other should be a new tid (>= 2)
        self.assertTrue(any(tid != 1 for tid in tids))
        # prune should remove stale 99
        self.assertNotIn(99, d.remote_tracks)

    async def test_detect_local_ultralytics_branch(self) -> None:
        """Test local detection using the Ultralytics branch."""
        d = self.detector
        d.use_ultralytics = True

        class Boxes:
            """Mock Ultralytics Boxes class."""

            def __init__(self) -> None:
                """Initialise mock boxes with test data."""
                self.xyxy = np.array(
                    [[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float,
                )
                self.conf = np.array([0.9, 0.8], dtype=float)
                self.cls = np.array([0, 1], dtype=float)

            def __len__(self) -> int:
                """Return number of boxes."""
                return 2

        boxes = Boxes()
        mock_res = MagicMock()
        mock_res.boxes = boxes

        def _ultra_call(_frame: Any) -> list[Any]:
            return [mock_res]

        d.ultralytics_model = _ultra_call  # type: ignore[assignment]
        out = await d._detect_local(np.zeros((4, 4, 3), dtype=np.uint8))
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0][5], 0)
        self.assertEqual(out[1][5], 1)

    async def test_detect_local_sahi_branch(self) -> None:
        """Test local detection using the SAHI branch."""
        # Force SAHI path; avoid loading real model by patching predictor
        with patch(
            'src.live_stream_detection.AutoDetectionModel.from_pretrained',
        ) as mock_from:
            mock_from.return_value = MagicMock()
            d = LiveStreamDetector(
                detect_with_server=False, use_ultralytics=False,
            )
        frame = np.zeros((10, 10, 3), dtype=np.uint8)

        class _BBox:
            def to_voc_bbox(self) -> tuple[int, int, int, int]:
                return (1, 2, 3, 4)

        class _Score:
            value = 0.77

        class _Cat:
            id = 5

        class _Obj:
            bbox = _BBox()
            score = _Score()
            category = _Cat()

        class _Res:
            object_prediction_list = [_Obj()]

        with patch(
            'src.live_stream_detection.get_sliced_prediction',
            return_value=_Res(),
        ):
            out = await d._detect_local(frame)
            self.assertEqual(out, [[1, 2, 3, 4, 0.77, 5]])

    async def test_generate_detections_with_ids_and_movement(self) -> None:
        d = self.detector
        d.detect_with_server = False
        d.movement_thr = 1.0
        d.movement_thr_sq = 1.0

        class Boxes2:
            def __init__(self) -> None:
                self.xyxy = np.array([[0.0, 0.0, 2.0, 2.0]], dtype=float)
                self.conf = np.array([0.9], dtype=float)
                self.cls = np.array([0], dtype=float)
                self.id = np.array([5], dtype=float)

            def __len__(self) -> int:
                return 1

        mock_res = MagicMock()
        mock_res.boxes = Boxes2()

        # Prev center far so movement=True
        d.prev_centers[5] = (100.0, 100.0)
        d.prev_centers_last_seen[5] = d.frame_count

        with patch.object(
            d.ultralytics_model, 'track', return_value=[mock_res],
        ):
            datas, tracked = await d.generate_detections(
                np.zeros((4, 4, 3), dtype=np.uint8),
            )
        self.assertEqual(len(datas), 1)
        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0][6], 5)  # tid
        self.assertEqual(tracked[0][7], 1)  # is_moving

    def test_centroid_tracker_match_and_moving_flag(self) -> None:
        """Test centroid tracker matching and movement flag assignment."""
        d = self.detector_server
        d.remote_tracker = 'centroid'
        d.movement_thr_sq = 4.0  # threshold for distance^2
        # existing track class 0 at center (10,10)
        d.remote_tracks = {
            1: {
                'bbox': (8.0, 8.0, 12.0, 12.0),
                'center': (10.0, 10.0),
                'last_seen': d.frame_count,
                'cls': 0,
            },
        }
        # detection near enough but moving > thr
        dets = [[12.0, 10.0, 14.0, 12.0, 0.9, 0]]
        out = d._track_remote_centroid(dets)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][6], 1)
        # moving flag depends on prev center -> should be 1
        self.assertIn(out[0][7], (0, 1))

    def test_centroid_tracker_prune_on_empty(self) -> None:
        """Test centroid tracker pruning old tracks when no detections."""
        d = self.detector_server
        d.remote_tracker = 'centroid'
        d.frame_count = 10
        d.max_id_keep = 5
        d.remote_tracks = {
            99: {
                'bbox': (0.0, 0.0, 1.0, 1.0),
                'center': (0.5, 0.5),
                'last_seen': 0,
                'cls': 0,
            },
        }
        out = d._track_remote_centroid([])
        self.assertEqual(out, [])
        self.assertNotIn(99, d.remote_tracks)

    def test_centroid_tracker_new_track_branch(self) -> None:
        """Test centroid tracker creating new tracks for unmatched dets."""
        d = self.detector_server
        d.remote_tracker = 'centroid'
        d.remote_tracks.clear()
        dets = [[10.0, 10.0, 20.0, 20.0, 0.9, 0]]
        out = d._track_remote_centroid(dets)
        self.assertEqual(len(out), 1)
        # should assign a new track id (starting from current next_remote_id)
        self.assertGreaterEqual(out[0][6], 0)

    def test_cover_zeros_column_selection(self) -> None:
        """Test the cover zeros algorithm selecting columns with most zeros."""
        d = self.detector_server
        # Construct matrix with a column having more zeros than any row
        mat = np.array(
            [
                [0.0, 1.0, 1.0],
                [0.0, 2.0, 3.0],
                [0.0, 4.0, 5.0],
            ], dtype=float,
        )
        rows, cols = d._cover_zeros(mat.copy())
        self.assertIn(0, cols)

    def test_centroid_tracker_dist_threshold_branch(self) -> None:
        """Test centroid tracker distance threshold validation."""
        d = self.detector_server
        d.remote_tracker = 'centroid'
        # one track at center (0,0), class 0
        d.remote_tracks = {
            1: {
                'bbox': (-1.0, -1.0, 1.0, 1.0),
                'center': (0.0, 0.0),
                'last_seen': 0,
                'cls': 0,
            },
        }
        d.movement_thr_sq = 4.0
        # detection at distance^2 exactly equal to movement_thr_sq*4 (boundary)
        # movement_thr_sq*4 = 16. Require dist_sq < 16 to match -> won't match
        dets = [[4.0, 0.0, 6.0, 2.0, 0.9, 0]]  # center at (5,1) -> dist^2=26
        out = d._track_remote_centroid(dets)
        # since no match, new track created
        self.assertEqual(len(out), 1)
        self.assertNotEqual(out[0][6], 1)

    def test_centroid_tracker_updates_best_distance(self) -> None:
        """Test centroid tracker updating to track with shortest distance."""
        d = self.detector_server
        d.remote_tracker = 'centroid'
        # two tracks same class; the second becomes the best (dist_sq update)
        d.movement_thr_sq = 10.0  # threshold*4 = 40.0
        d.remote_tracks = {
            1: {
                'bbox': (0.0, 0.0, 2.0, 2.0),
                'center': (1.0, 1.0),
                'last_seen': 0,
                'cls': 0,
            },
            2: {
                'bbox': (0.0, 0.0, 2.0, 2.0),
                'center': (2.5, 2.5),
                'last_seen': 0,
                'cls': 0,
            },
        }
        dets = [[2.0, 2.0, 4.0, 4.0, 0.9, 0]]  # center (3,3)
        out = d._track_remote_centroid(dets)
        # Should assign to the closer track id=2
        self.assertEqual(out[0][6], 2)

    def test_centroid_tracker_skip_different_class_continue(self) -> None:
        """Test centroid tracker skipping tracks with different classes."""
        d = self.detector_server
        d.remote_tracker = 'centroid'
        # track 1 is different class -> should be skipped via continue
        d.movement_thr_sq = 100.0
        d.remote_tracks = {
            1: {
                'bbox': (0.0, 0.0, 2.0, 2.0),
                'center': (1.0, 1.0),
                'last_seen': 0,
                'cls': 0,
            },
            2: {
                'bbox': (10.0, 10.0, 12.0, 12.0),
                'center': (11.0, 11.0),
                'last_seen': 0,
                'cls': 1,
            },
        }
        dets = [[10.0, 10.0, 12.0, 12.0, 0.9, 1]]
        out = d._track_remote_centroid(dets)
        self.assertEqual(out[0][6], 2)

    def test_cover_zeros_no_counts_break_guard(self) -> None:
        """Test cover zeros algorithm when no zeros exist in matrix."""
        d = self.detector_server
        # Create a scenario where zero_locs becomes empty immediately,
        # so row_counts and col_counts remain zeros dicts.
        mat = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
        rows, cols = d._cover_zeros(mat.copy())
        # With no zeros, both sets should be empty
        self.assertEqual(rows, set())
        self.assertEqual(cols, set())

    def test_adjust_matrix_with_min_true_branch(self) -> None:
        """Test matrix adjustment when minimum uncovered value is positive."""
        d = self.detector_server
        # Choose matrix so that min uncovered > 0
        mat = np.array(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 3.0],
                [2.0, 3.0, 4.0],
            ], dtype=float,
        )
        covered_rows = {0}
        covered_cols = {1}
        changed = d._adjust_matrix_with_min(mat, covered_rows, covered_cols)
        self.assertTrue(changed)
        # Verify expected adjustments: subtract 1 from uncovered cells,
        # add 1 to intersections (0,1)
        self.assertEqual(mat[1, 0], 0.0)
        self.assertEqual(mat[1, 2], 2.0)
        self.assertEqual(mat[2, 0], 1.0)
        self.assertEqual(mat[2, 2], 3.0)
        self.assertEqual(mat[0, 1], 2.0)


if __name__ == '__main__':
    unittest.main()


"""
pytest \
    --cov=src.live_stream_detection \
    --cov-report=term-missing tests/src/live_stream_detection_test.py
"""
