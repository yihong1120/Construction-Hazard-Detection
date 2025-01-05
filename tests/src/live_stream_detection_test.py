from __future__ import annotations

import os
import sys
import unittest
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import aiohttp
import cv2
import numpy as np
import yarl
from multidict import CIMultiDict
from multidict import CIMultiDictProxy

from src.live_stream_detection import LiveStreamDetector
from src.live_stream_detection import main


class TestLiveStreamDetector(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for the LiveStreamDetector class methods.
    """

    def setUp(self) -> None:
        """
        Set up the LiveStreamDetector instance for tests.
        """
        # Mock environment variables to avoid missing credentials
        patcher_env = patch.dict(
            os.environ,
            {'API_USERNAME': 'test_user', 'API_PASSWORD': 'test_pass'},
        )
        patcher_env.start()
        self.addCleanup(patcher_env.stop)

        # Initialise default detector parameters for testing
        self.api_url: str = 'http://mocked-api.com'
        self.model_key: str = 'yolo11n'
        self.output_folder: str = 'test_output'
        self.detect_with_server: bool = False

        # Create an instance of LiveStreamDetector for use in tests
        self.detector: LiveStreamDetector = LiveStreamDetector(
            api_url=self.api_url,
            model_key=self.model_key,
            output_folder=self.output_folder,
            detect_with_server=self.detect_with_server,
        )

    ########################################################################
    # Initialisation tests
    ########################################################################

    def test_initialisation(self) -> None:
        """
        Test the initialisation of the LiveStreamDetector instance.
        """
        detector = LiveStreamDetector(
            api_url=self.api_url,
            model_key=self.model_key,
            output_folder=self.output_folder,
            detect_with_server=self.detect_with_server,
        )

        # Assert initialisation values
        self.assertEqual(detector.api_url, self.api_url)
        self.assertEqual(detector.model_key, self.model_key)
        self.assertEqual(detector.output_folder, self.output_folder)
        self.assertEqual(detector.detect_with_server, self.detect_with_server)
        self.assertEqual(detector.shared_token, {'access_token': ''})

    def test_initialisation_with_shared_token(self) -> None:
        """
        Test initialisation when shared_token is provided.
        """
        shared_token = {
            'access_token': 'test_token',
        }

        # Initialise with a shared token
        detector = LiveStreamDetector(
            api_url=self.api_url,
            model_key=self.model_key,
            output_folder=self.output_folder,
            detect_with_server=self.detect_with_server,
            shared_token=shared_token,
        )

        # Validate that the shared token is set correctly
        self.assertEqual(detector.shared_token, shared_token)

    def test_shared_lock(self):
        """
        Test shared lock acquire and release methods.
        """
        # Mock a shared lock for testing
        shared_lock = MagicMock()
        shared_lock.acquire = MagicMock()
        shared_lock.release = MagicMock()

        # Initialise the detector with a shared lock
        detector = LiveStreamDetector(
            api_url=self.api_url,
            model_key=self.model_key,
            shared_lock=shared_lock,
        )

        # Test acquiring the lock
        detector.acquire_shared_lock()
        shared_lock.acquire.assert_called_once()

        # Test releasing the lock
        detector.release_shared_lock()
        shared_lock.release.assert_called_once()

    ########################################################################
    # Authentication tests
    ########################################################################

    @patch('aiohttp.ClientSession.post')
    async def test_authenticate_skip_if_token_exists(
        self,
        mock_post: MagicMock,
    ) -> None:
        """
        Test authenticate skips re-authentication if token exists.
        """
        # Set an existing token to bypass authentication
        self.detector.shared_token['access_token'] = 'existing_token'

        # Call authenticate and ensure no network requests are made
        await self.detector.authenticate()
        mock_post.assert_not_called()

    @patch('aiohttp.ClientSession.post')
    async def test_authenticate(
        self,
        mock_post: MagicMock,
    ) -> None:
        """
        Test successful authentication flow.
        """
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(
            return_value={'access_token': 'fake_token'},
        )
        mock_post.return_value.__aenter__.return_value = mock_response

        await self.detector.authenticate()
        self.assertEqual(
            self.detector.shared_token['access_token'], 'fake_token',
        )

    @patch('aiohttp.ClientSession.post')
    async def test_authenticate_missing_credentials(
        self,
        mock_post: MagicMock,
    ) -> None:
        """
        Test the authenticate method
        when credentials are missing in environment variables.
        """
        with patch.dict(os.environ, {}, clear=True):
            # API_USERNAME / API_PASSWORD not exist
            with self.assertRaises(ValueError) as ctx:
                await self.detector.authenticate()
            self.assertIn(
                'Missing API_USERNAME or API_PASSWORD',
                str(ctx.exception),
            )

    @patch('aiohttp.ClientSession.post')
    async def test_authenticate_raises_for_status(
        self,
        mock_post: MagicMock,
    ) -> None:
        """
        Test the authenticate method raising ClientResponseError
        if response.raise_for_status fails.
        """
        # Create typed headers and mock RequestInfo
        headers: CIMultiDict[str] = CIMultiDict()
        mock_request_info = aiohttp.RequestInfo(
            url=yarl.URL('http://mock.com/auth'),
            method='POST',
            headers=CIMultiDictProxy(headers),
            real_url=yarl.URL('http://mock.com/auth'),
        )

        # Mock a response with status 401
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = (
            aiohttp.ClientResponseError(
                request_info=mock_request_info,
                history=(),
                status=401,
                message='Unauthorized',
            )
        )
        mock_post.return_value.__aenter__.return_value = mock_response

        # Ensure a ClientResponseError is raised by the authenticate call
        with self.assertRaises(aiohttp.ClientResponseError):
            await self.detector.authenticate()

    @patch('aiohttp.ClientSession.post')
    async def test_authenticate_raises_key_error_if_no_access_token(
        self,
        mock_post: MagicMock,
    ) -> None:
        """
        If the server returns a JSON response without 'access_token',
        a KeyError should occur.
        """
        mock_response: MagicMock = MagicMock()
        mock_response.raise_for_status = MagicMock()

        # Return an empty JSON response
        mock_response.json = AsyncMock(return_value={})
        mock_post.return_value.__aenter__.return_value = mock_response

        # Expect a KeyError to be raised
        with self.assertRaises(KeyError):
            await self.detector.authenticate()

    ########################################################################
    # Detection tests
    ########################################################################

    @patch('cv2.imencode', return_value=(False, None))
    async def test_generate_detections_cloud_encode_fail(
        self,
        mock_imencode: MagicMock,
    ) -> None:
        """
        Test generate_detections_cloud raises ValueError
        if frame encoding fails.
        """
        # Simulate a frame encoding failure
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with self.assertRaises(ValueError) as ctx:
            await self.detector.generate_detections_cloud(frame)
        self.assertIn(
            'Failed to encode frame as PNG bytes.',
            str(ctx.exception),
        )

    @patch('aiohttp.ClientSession.post')
    async def test_generate_detections_cloud(
        self,
        mock_post: MagicMock,
    ) -> None:
        """
        Test cloud detection generation.
        """
        frame: np.ndarray = np.zeros((480, 640, 3), dtype=np.uint8)

        # Simulate the token response
        mock_token_response = MagicMock()
        mock_token_response.raise_for_status = MagicMock()
        mock_token_response.json = AsyncMock(
            return_value={'access_token': 'fake_token'},
        )

        # Simulate the detection response
        mock_detection_response = MagicMock()
        mock_detection_response.raise_for_status = MagicMock()
        mock_detection_response.json = AsyncMock(
            return_value=[
                [10, 10, 50, 50, 0.9, 0],
                [20, 20, 60, 60, 0.8, 1],
            ],
        )

        # Simulate the two responses from the server
        mock_post.return_value.__aenter__.side_effect = [
            mock_token_response,       # First call: /token
            mock_detection_response,   # Second call: /detect
        ]

        datas: list[list[Any]] = (
            await self.detector.generate_detections_cloud(frame)
        )

        # Validate the response from the server
        self.assertIsInstance(datas, list)
        self.assertEqual(len(datas), 2)

        # Validate the structure and types of the detection data
        for data in datas:
            self.assertIsInstance(data, list)
            self.assertEqual(len(data), 6)
            self.assertIsInstance(data[0], int)
            self.assertIsInstance(data[1], int)
            self.assertIsInstance(data[2], int)
            self.assertIsInstance(data[3], int)
            self.assertIsInstance(data[4], float)
            self.assertIsInstance(data[5], int)

    @patch('aiohttp.ClientSession.post')
    async def test_generate_detections_cloud_retry_on_token_expiry(
        self,
        mock_post: MagicMock,
    ) -> None:
        """
        Test generate_detections_cloud retries on token expiry.
        """
        # Assume the token already exists, so authenticate() is skipped
        self.detector.shared_token['access_token'] = 'old_token'

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Simulate the first detection response with 401 Unauthorised
        mock_unauthorized_detection = MagicMock()
        mock_unauthorized_detection.status = 401
        headers: CIMultiDict[str] = CIMultiDict()
        mock_unauthorized_detection.raise_for_status.side_effect = (
            aiohttp.ClientResponseError(
                request_info=aiohttp.RequestInfo(
                    url=yarl.URL('http://mock.com/detect'),
                    method='POST',
                    headers=CIMultiDictProxy(headers),
                ),
                history=(),
                status=401,
                message='Unauthorized',
            )
        )

        # Simulate the token response with the new token
        mock_token_response = MagicMock()
        mock_token_response.status = 200
        mock_token_response.raise_for_status = MagicMock()
        mock_token_response.json = AsyncMock(
            return_value={'access_token': 'new_token'},
        )

        # Simulate the successful detection response
        mock_success_detection = MagicMock()
        mock_success_detection.status = 200
        mock_success_detection.raise_for_status = MagicMock()
        mock_success_detection.json = AsyncMock(
            return_value=[
                [10, 10, 50, 50, 0.9, 0],
                [20, 20, 60, 60, 0.8, 1],
            ],
        )

        mock_post.return_value.__aenter__.side_effect = [
            # First detection attempt: 401 Unauthorised
            mock_unauthorized_detection,
            # Token refresh response
            mock_token_response,
            # Second detection attempt: Success
            mock_success_detection,
        ]

        datas = await self.detector.generate_detections_cloud(frame)
        self.assertEqual(len(datas), 2)
        self.assertEqual(
            self.detector.shared_token['access_token'], 'new_token',
        )

        # Validate the structure and types of the detection data
        for data in datas:
            self.assertIsInstance(data, list)
            self.assertEqual(len(data), 6)
            self.assertIsInstance(data[0], int)
            self.assertIsInstance(data[1], int)
            self.assertIsInstance(data[2], int)
            self.assertIsInstance(data[3], int)
            self.assertIsInstance(data[4], float)
            self.assertIsInstance(data[5], int)

    @patch('aiohttp.ClientSession.post')
    async def test_generate_detections_cloud_request_error(
        self,
        mock_post: MagicMock,
    ) -> None:
        """
        Test generate_detections_cloud handles request errors properly.
        """
        # Assume the token already exists, so authenticate() is skipped
        self.detector.shared_token['access_token'] = 'fake_token'

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Simulate the detection response with 500 Internal Server Error
        headers: CIMultiDict[str] = CIMultiDict()
        request_info = aiohttp.RequestInfo(
            url=yarl.URL('http://mock.com/detect'),
            method='POST',
            headers=CIMultiDictProxy(headers),
            real_url=yarl.URL('http://mock.com/detect'),
        )

        # Simulate the detection response with 500 Internal Server Error
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = (
            aiohttp.ClientResponseError(
                request_info=request_info,
                history=(),
                status=500,
                message='Internal Server Error',
            )
        )
        mock_post.return_value.__aenter__.return_value = mock_response

        with self.assertLogs(self.detector.logger, level='ERROR') as captured:
            with self.assertRaises(aiohttp.ClientResponseError):
                await self.detector.generate_detections_cloud(frame)

        # Validate the error message in the logs
        combined_logs = '\n'.join(captured.output)
        self.assertIn('Failed to send detection request:', combined_logs)

    @patch('src.live_stream_detection.get_sliced_prediction')
    @patch('src.live_stream_detection.AutoDetectionModel.from_pretrained')
    async def test_generate_detections_local_with_predictions(
        self,
        mock_from_pretrained: MagicMock,
        mock_get_sliced_prediction: MagicMock,
    ) -> None:
        """
        Test the generate_detections_local method with predictions.
        """
        # Mock the model
        mock_model: MagicMock = MagicMock()
        mock_from_pretrained.return_value = mock_model

        # Mock the object predictions
        mock_result: MagicMock = MagicMock()
        mock_result.object_prediction_list = [
            MagicMock(
                category=MagicMock(id=0),  # Set category ID is 0
                bbox=MagicMock(
                    # Set the bounding box
                    to_voc_bbox=lambda: [10.5, 20.3, 50.8, 60.1],
                ),
                score=MagicMock(value=0.85),  # Set the score
            ),
            MagicMock(
                category=MagicMock(id=1),  # Set category ID to 1
                bbox=MagicMock(
                    to_voc_bbox=lambda: [30, 40, 70, 80],
                ),
                score=MagicMock(value=0.9),
            ),
        ]
        mock_get_sliced_prediction.return_value = mock_result

        # Set up the input frame
        frame: np.ndarray = np.zeros((480, 640, 3), dtype=np.uint8)

        # Generate the detections
        datas: list[list[Any]] = await self.detector.generate_detections_local(
            frame,
        )

        # Validate the structure and types of the detection data
        self.assertIsInstance(datas, list)

        # Ensure two detections are returned
        self.assertEqual(len(datas), 2)

        # Validate the structure and types of the detection data
        for data in datas:
            self.assertIsInstance(data, list)
            self.assertEqual(len(data), 6)
            self.assertIsInstance(data[0], int)
            self.assertIsInstance(data[1], int)
            self.assertIsInstance(data[2], int)
            self.assertIsInstance(data[3], int)
            self.assertIsInstance(data[4], float)
            self.assertIsInstance(data[5], int)

        # Ensure the first detection is correct
        self.assertEqual(datas[0], [10, 20, 50, 60, 0.85, 0])
        # Ensure the second detection is correct
        self.assertEqual(datas[1], [30, 40, 70, 80, 0.9, 1])

        # Validate the calls to the model
        mock_get_sliced_prediction.assert_called_once_with(
            frame,
            mock_model,
            slice_height=376,
            slice_width=376,
            overlap_height_ratio=0.3,
            overlap_width_ratio=0.3,
        )

    async def test_generate_detections(self) -> None:
        """
        Test the generate_detections method.
        """
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mat_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.detector.detect_with_server = False
        with patch.object(
            self.detector, 'generate_detections_local',
            return_value=[[10, 10, 50, 50, 0.9, 0]],
        ) as mock_local:
            datas, _ = await self.detector.generate_detections(mat_frame)
            self.assertEqual(len(datas), 1)
            self.assertEqual(datas[0][5], 0)
            mock_local.assert_called_once_with(mat_frame)

        self.detector.detect_with_server = True
        with patch.object(
            self.detector, 'generate_detections_cloud',
            return_value=[[20, 20, 60, 60, 0.8, 1]],
        ) as mock_cloud:
            datas, _ = await self.detector.generate_detections(mat_frame)
            self.assertEqual(len(datas), 1)
            self.assertEqual(datas[0][5], 1)
            mock_cloud.assert_called_once_with(mat_frame)

    ########################################################################
    # run_detection method tests
    ########################################################################

    @patch('src.live_stream_detection.cv2.VideoCapture')
    async def test_run_detection_stream_not_opened(
        self,
        mock_vcap: MagicMock,
    ) -> None:
        """
        If the stream cannot be opened, ValueError is raised.
        """
        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = False
        mock_vcap.return_value = cap_mock

        with self.assertRaises(ValueError) as ctx:
            await self.detector.run_detection('fake_stream')

        # Validate the error message
        self.assertIn('Failed to open stream', str(ctx.exception))

    async def test_run_detection(self) -> None:
        """
        Test the run_detection method.
        """
        stream_url: str = 'http://example.com/virtual_stream'
        cap_mock: MagicMock = MagicMock()
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
                    await self.detector.run_detection(stream_url)

        cap_mock.read.assert_called()
        cap_mock.release.assert_called_once()

    @patch(
        'src.live_stream_detection.cv2.waitKey',
        side_effect=[-1, -1, ord('q')],
    )
    @patch('src.live_stream_detection.cv2.imshow')
    @patch('src.live_stream_detection.cv2.VideoCapture')
    async def test_run_detection_loop(
        self,
        mock_vcap: MagicMock,
        mock_imshow: MagicMock,
        mock_waitKey: MagicMock,
    ) -> None:
        """
        Test run_detection loop with valid frames then user presses 'q'.
        """
        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = True

        # Mock the frames read from the stream
        frames_side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None),
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
        ]
        cap_mock.read.side_effect = frames_side_effect
        mock_vcap.return_value = cap_mock

        # Execute run_detection
        await self.detector.run_detection('fake_stream')

        # Validate the calls
        self.assertGreaterEqual(cap_mock.read.call_count, 4, 'read()至少被呼叫4次')
        cap_mock.release.assert_called_once()
        mock_imshow.assert_called()
        mock_waitKey.assert_called()

    ########################################################################
    # Post-processing function tests
    ########################################################################

    def test_remove_overlapping_labels(self) -> None:
        """
        Test the remove_overlapping_labels method.
        """
        datas = [
            [10, 10, 50, 50, 0.9, 0],  # Hardhat
            [10, 10, 50, 45, 0.8, 2],  # NO-Hardhat (overlap > 0.8)
            [20, 20, 60, 60, 0.85, 7],  # Safety Vest
            [20, 20, 60, 55, 0.75, 4],  # NO-Safety Vest (overlap > 0.8)
        ]
        expected_datas = [
            [10, 10, 50, 50, 0.9, 0],  # Hardhat
            [20, 20, 60, 60, 0.85, 7],  # Safety Vest
        ]
        filtered_datas = self.detector.remove_overlapping_labels(datas)
        self.assertEqual(len(filtered_datas), len(expected_datas))
        # for fd, ed in zip(filtered_datas, expected_datas):
        self.assertEqual(filtered_datas, expected_datas)

        # Add more test cases to cover different overlap scenarios
        datas = [
            [10, 10, 50, 50, 0.9, 0],  # Hardhat
            [30, 30, 70, 70, 0.8, 2],  # NO-Hardhat (no overlap)
            [10, 10, 50, 50, 0.85, 7],  # Safety Vest (same as Hardhat)
            [60, 60, 100, 100, 0.75, 4],  # NO-Safety Vest (no overlap)
        ]
        expected_datas = [
            [10, 10, 50, 50, 0.9, 0],  # Hardhat
            [10, 10, 50, 50, 0.85, 7],  # Safety Vest
            [30, 30, 70, 70, 0.8, 2],  # NO-Hardhat
            [60, 60, 100, 100, 0.75, 4],  # NO-Safety Vest
        ]
        filtered_datas = self.detector.remove_overlapping_labels(datas)
        self.assertEqual(len(filtered_datas), len(expected_datas))

        # Sorting both lists before comparing
        filter_datas_sorted = sorted(filtered_datas)
        expected_datas_sorted = sorted(expected_datas)

        for fd, ed in zip(filter_datas_sorted, expected_datas_sorted):
            self.assertEqual(fd, ed)

    def test_overlap_percentage(self) -> None:
        """
        Test the overlap_percentage method.
        """
        bbox1 = [10, 10, 50, 50]
        bbox2 = [20, 20, 40, 40]
        overlap = self.detector.overlap_percentage(bbox1, bbox2)
        self.assertAlmostEqual(overlap, 0.262344, places=6)

    def test_is_contained(self) -> None:
        """
        Test the is_contained method.
        """
        outer_bbox = [10, 10, 50, 50]
        inner_bbox = [20, 20, 40, 40]
        self.assertTrue(self.detector.is_contained(inner_bbox, outer_bbox))

        inner_bbox = [5, 5, 40, 40]
        self.assertFalse(self.detector.is_contained(inner_bbox, outer_bbox))

        inner_bbox = [10, 10, 50, 50]
        self.assertTrue(self.detector.is_contained(inner_bbox, outer_bbox))

        inner_bbox = [0, 0, 60, 60]
        self.assertFalse(self.detector.is_contained(inner_bbox, outer_bbox))

    def test_remove_completely_contained_labels(self) -> None:
        """
        Test the remove_completely_contained_labels method.
        """
        datas = [
            [10, 10, 50, 50, 0.9, 0],   # Hardhat
            [20, 20, 40, 40, 0.8, 2],   # NO-Hardhat (contained within Hardhat)
            [20, 20, 60, 60, 0.85, 7],  # Safety Vest
            # NO-Safety Vest (contained within Safety Vest)
            [25, 25, 35, 35, 0.75, 4],
        ]
        expected_datas = [
            [10, 10, 50, 50, 0.9, 0],   # Hardhat
            [20, 20, 60, 60, 0.85, 7],  # Safety Vest
        ]
        filtered_datas = self.detector.remove_completely_contained_labels(
            datas,
        )
        self.assertEqual(filtered_datas, expected_datas)

        # Add more test cases to cover different containment scenarios
        datas = [
            [10, 10, 50, 50, 0.9, 0],   # Hardhat
            [30, 30, 70, 70, 0.8, 2],   # NO-Hardhat (not contained)
            [10, 10, 50, 50, 0.85, 7],  # Safety Vest (same as Hardhat)
            [60, 60, 100, 100, 0.75, 4],  # NO-Safety Vest (not contained)
        ]
        expected_datas = [
            [10, 10, 50, 50, 0.9, 0],   # Hardhat
            [30, 30, 70, 70, 0.8, 2],   # NO-Hardhat
            [10, 10, 50, 50, 0.85, 7],  # Safety Vest
            [60, 60, 100, 100, 0.75, 4],  # NO-Safety Vest
        ]
        filtered_datas = self.detector.remove_completely_contained_labels(
            datas,
        )
        self.assertEqual(filtered_datas, expected_datas)

        # Sorting both lists before comparing
        filter_datas_sorted = sorted(filtered_datas)
        expected_datas_sorted = sorted(expected_datas)

        for fd, ed in zip(filter_datas_sorted, expected_datas_sorted):
            self.assertEqual(fd, ed)

    def test_remove_hardhat_in_no_hardhat(self) -> None:
        """
        Test the remove_completely_contained_labels method
        with a Hardhat bounding box.
        """
        # Input data with bounding boxes
        datas = [
            [10, 10, 50, 50, 0.8, 2],  # No-Hardhat
            [20, 20, 30, 30, 0.9, 0],  # Hardhat
        ]

        # Expected result: Only the No-Hardhat bounding box remains
        expected_datas = [
            [10, 10, 50, 50, 0.8, 2],
        ]

        # Call the method being tested
        filtered_datas = self.detector.remove_completely_contained_labels(
            datas,
        )

        # Assert that the filtered data matches the expected output
        self.assertEqual(
            filtered_datas,
            expected_datas,
            'Hardhat should be removed '
            "when it's completely contained within No-Hardhat",
        )

    def test_remove_safety_vest_in_no_vest(self) -> None:
        """
        Test the remove_completely_contained_labels method
        with a Safety Vest bounding box.
        """
        # Input data with bounding boxes
        datas: list[list[float]] = [
            # No-Safety Vest bounding box (large box)
            [10, 10, 50, 50, 0.85, 4],
            # Safety Vest bounding box (contained within the first)
            [20, 20, 30, 30, 0.9, 7],
        ]

        # Expected filtered output: Safety Vest box is removed
        expected_datas: list[list[float]] = [
            [10, 10, 50, 50, 0.85, 4],
        ]

        # Perform filtering using the method under test
        filtered_datas = self.detector.remove_completely_contained_labels(
            datas,
        )

        # Assert the output matches the expected result
        self.assertEqual(
            filtered_datas,
            expected_datas,
            'Safety Vest should be removed '
            "when it's contained by No-Safety Vest",
        )

    ########################################################################
    # Test main()
    ########################################################################

    @patch.object(
        sys, 'argv', [
            'python',  # 模擬 Python 執行檔
            '--url', 'http://example.com/virtual_stream',
            '--api_url', 'http://mocked-api.com',
            '--detect_with_server',
        ],
    )
    @patch(
        'src.live_stream_detection.LiveStreamDetector.run_detection',
        new_callable=AsyncMock,
    )
    @patch(
        'src.live_stream_detection.LiveStreamDetector.__init__',
        return_value=None,
    )
    async def test_main(
        self,
        mock_init: MagicMock,
        mock_run_detection: AsyncMock,
    ) -> None:
        """
        Test the main function with valid arguments.
        """
        # Execute the main function
        await main()

        # Ensure the detector was initialised with the correct arguments
        mock_init.assert_called_once_with(
            api_url='http://mocked-api.com',
            model_key='yolo11n',
            output_folder=None,
            detect_with_server=True,
            shared_token={'access_token': ''},
        )

        # Ensure the run_detection method was called with the expected URL
        mock_run_detection.assert_called_once_with(
            'http://example.com/virtual_stream',
        )


if __name__ == '__main__':
    unittest.main()
