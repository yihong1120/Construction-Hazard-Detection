from __future__ import annotations

import time
import unittest
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import cv2
import numpy as np

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
        self.api_url: str = 'http://127.0.0.1:8001'
        self.model_key: str = 'yolo11n'
        self.output_folder: str = 'test_output'
        self.detect_with_server: bool = False
        self.detector: LiveStreamDetector = LiveStreamDetector(
            api_url=self.api_url,
            model_key=self.model_key,
            output_folder=self.output_folder,
            detect_with_server=self.detect_with_server,
        )

    @patch('PIL.ImageFont.truetype')
    def test_initialisation(
        self,
        mock_truetype: MagicMock,
    ) -> None:
        """
        Test the initialisation of the LiveStreamDetector instance.

        Args:
            mock_truetype (MagicMock): Mock for PIL.ImageFont.truetype.
        """
        mock_truetype.return_value = MagicMock()

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

        # Assert the shared token is initialised correctly with default values
        self.assertEqual(
            detector.shared_token,
            {'access_token': None, 'token_expiry': 0},
        )

    def test_initialisation_with_shared_token(self) -> None:
        """
        Test initialisation when shared_token is provided.
        """
        shared_token = {
            'access_token': 'test_token',
            'token_expiry': time.time() + 1000,
        }
        detector = LiveStreamDetector(
            api_url=self.api_url,
            model_key=self.model_key,
            output_folder=self.output_folder,
            detect_with_server=self.detect_with_server,
            shared_token=shared_token,
        )

        self.assertEqual(detector.shared_token, shared_token)

    def test_token_expired_with_no_expiry(self) -> None:
        """
        Test token_expired when token_expiry is None.
        """
        self.detector.shared_token['token_expiry'] = None
        self.assertTrue(self.detector.token_expired())

    async def test_ensure_authenticated_with_lock(self) -> None:
        """
        Test ensure_authenticated when shared_lock is used.
        """
        # Mock the shared lock to test the lock acquisition
        shared_lock = MagicMock()

        detector = LiveStreamDetector(
            api_url=self.api_url,
            model_key=self.model_key,
            output_folder=self.output_folder,
            detect_with_server=self.detect_with_server,
            shared_token={'access_token': None, 'token_expiry': 0},
            shared_lock=shared_lock,
        )

        # If token is expired, ensure_authenticated should acquire the lock
        with patch.object(detector, 'token_expired', return_value=True):
            # Mock authenticate method
            with patch.object(
                detector, 'authenticate', AsyncMock(),
            ) as mock_auth:
                await detector.ensure_authenticated()

                # Validate the lock is acquired and released
                shared_lock.acquire.assert_called_once()
                shared_lock.release.assert_called_once()

                # Validate the authenticate method is called
                mock_auth.assert_called_once()

    @patch('src.live_stream_detection.cv2.VideoCapture')
    @patch('src.live_stream_detection.AutoDetectionModel.from_pretrained')
    async def test_generate_detections_local(
        self,
        mock_from_pretrained: MagicMock,
        mock_video_capture: MagicMock,
    ) -> None:
        """
        Test local detection generation.

        Args:
            mock_from_pretrained (MagicMock): Mock for
                AutoDetectionModel.from_pretrained.
            mock_video_capture (MagicMock): Mock for cv2.VideoCapture.
        """
        mock_model: MagicMock = MagicMock()
        mock_from_pretrained.return_value = mock_model

        frame: np.ndarray = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_result: MagicMock = MagicMock()
        mock_result.object_prediction_list = [
            MagicMock(
                category=MagicMock(id=0), bbox=MagicMock(
                    to_voc_bbox=lambda: [10, 10, 50, 50],
                ), score=MagicMock(value=0.9),
            ),
            MagicMock(
                category=MagicMock(id=1), bbox=MagicMock(
                    to_voc_bbox=lambda: [20, 20, 60, 60],
                ), score=MagicMock(value=0.8),
            ),
        ]
        mock_model.predict.return_value = mock_result

        datas: list[list[Any]] = await self.detector.generate_detections_local(
            frame,
        )

        # Assert the structure and types of the detection dataㄋ
        self.assertIsInstance(datas, list)
        for data in datas:
            self.assertIsInstance(data, list)
            self.assertEqual(len(data), 6)
            self.assertIsInstance(data[0], int)
            self.assertIsInstance(data[1], int)
            self.assertIsInstance(data[2], int)
            self.assertIsInstance(data[3], int)
            self.assertIsInstance(data[4], float)
            self.assertIsInstance(data[5], int)

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

    @patch('aiohttp.ClientSession.post')
    async def test_generate_detections_cloud(
        self, mock_post: MagicMock,
    ) -> None:
        """
        Test cloud detection generation.
        """
        frame: np.ndarray = np.zeros((480, 640, 3), dtype=np.uint8)

        # Simulate the token response
        mock_token_response = MagicMock()
        mock_token_response.json = AsyncMock(
            return_value={'access_token': 'fake_token'},
        )

        # Simulate the detection response
        mock_detection_response = MagicMock()
        mock_detection_response.json = AsyncMock(
            return_value=[[10, 10, 50, 50, 0.9, 0], [20, 20, 60, 60, 0.8, 1]],
        )

        # Simulate the two responses from the server
        mock_post.return_value.__aenter__.side_effect = [
            mock_token_response,      # First call: /token
            mock_detection_response,  # Second call: /detect
        ]

        datas: list[list[Any]] = (
            await self.detector.generate_detections_cloud(frame)
        )

        # Validate the structure and types of the detection data
        self.assertIsInstance(datas, list)
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
    async def test_authenticate(self, mock_post: MagicMock) -> None:
        mock_response: MagicMock = MagicMock()
        mock_response.json = AsyncMock(
            return_value={'access_token': 'fake_token'},
        )
        mock_post.return_value.__aenter__.return_value = mock_response

        await self.detector.authenticate()

        # Validate the shared token data after authentication is successful
        assert self.detector.shared_token['access_token'] == 'fake_token'
        token_expiry = self.detector.shared_token['token_expiry']
        assert token_expiry is not None
        self.assertGreater(token_expiry, time.time())

    def test_token_expired(self) -> None:
        """
        Test the token_expired method.
        """
        # Test expired token
        self.detector.shared_token['token_expiry'] = int(time.time() - 1)
        self.assertTrue(self.detector.token_expired())

        # Test non-expired token
        self.detector.shared_token['token_expiry'] = int(time.time() + 1000)
        self.assertFalse(self.detector.token_expired())

    async def test_ensure_authenticated(self) -> None:
        """
        Test the ensure_authenticated method.
        """
        with patch.object(self.detector, 'token_expired', return_value=True):
            with patch.object(
                self.detector, 'authenticate',
            ) as mock_authenticate:
                await self.detector.ensure_authenticated()
                mock_authenticate.assert_called_once()

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

    @patch('aiohttp.ClientSession.post')
    async def test_authenticate_error(self, mock_post: MagicMock) -> None:
        """
        Test the authenticate method with error response.

        Args:
            mock_post (MagicMock): Mock for aiohttp.ClientSession.post.
        """
        mock_response: MagicMock = MagicMock()
        mock_response.json = AsyncMock(
            return_value={'msg': 'Authentication failed'},
        )
        mock_post.return_value.__aenter__.return_value = mock_response

        with self.assertRaises(Exception) as context:
            await self.detector.authenticate()

        self.assertIn('Authentication failed', str(context.exception))

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

    @patch('src.live_stream_detection.cv2.imshow')
    @patch('src.live_stream_detection.cv2.waitKey', return_value=ord('q'))
    @patch('src.live_stream_detection.cv2.VideoCapture')
    async def test_run_detection_fail_read_frame(
        self,
        mock_VideoCapture: MagicMock,
        mock_waitKey: MagicMock,
        mock_imshow: MagicMock,
    ) -> None:
        """
        Test the run_detection method when failing to read a frame.
        """
        stream_url: str = 'https://example.com/fake_stream'

        # Mock VideoCapture behaviour
        cap_mock = MagicMock()
        cap_mock.read.side_effect = [
            # Failed to read frame
            (False, None),
            (False, None),

            # Successful to read frame
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
        ]
        cap_mock.isOpened.return_value = True
        mock_VideoCapture.return_value = cap_mock

        # Execute the method under test
        await self.detector.run_detection(stream_url)

        # Assert expected calls
        cap_mock.read.assert_called()
        cap_mock.release.assert_called_once()
        mock_waitKey.assert_called()
        mock_imshow.assert_called()

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

    @patch('aiohttp.ClientSession.post')
    async def test_authenticate_missing_token_data(
        self, mock_post: MagicMock,
    ) -> None:
        """
        Test the authenticate method when the token data
        is missing 'msg' and 'access_token'.

        Args:
            mock_post (MagicMock): Mock for aiohttp.ClientSession.post.
        """
        mock_response: MagicMock = MagicMock()
        mock_response.json = AsyncMock(return_value={})
        mock_post.return_value.__aenter__.return_value = mock_response

        with self.assertRaises(Exception) as context:
            await self.detector.authenticate()

        self.assertIn(
            "Token data does not contain 'msg' or 'access_token'", str(
                context.exception,
            ),
        )
        mock_post.assert_called_once()

    @patch('src.live_stream_detection.get_sliced_prediction')
    @patch('src.live_stream_detection.AutoDetectionModel.from_pretrained')
    async def test_generate_detections_local_with_predictions(
        self,
        mock_from_pretrained: MagicMock,
        mock_get_sliced_prediction: MagicMock,
    ) -> None:
        """
        Test the generate_detections_local method with predictions.

        Args:
            mock_from_pretrained (MagicMock): Mock for
                AutoDetectionModel.from_pretrained.
            mock_get_sliced_prediction (MagicMock): Mock for
                get_sliced_prediction

        Raises:
            AssertionError: If the detection data
                does not match the expected output.
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

    def test_remove_hardhat_in_no_hardhat(self) -> None:
        """
        Test the remove_completely_contained_labels method

        Args:
            self: Instance of the test class.

        Raises:
            AssertionError: If the filtered data
                does not match the expected output.

        Test Scenario:
            - A `No-Hardhat` bounding box (large box) contains a
            `Hardhat` bounding box (smaller box).
            - Expected behaviour: The `Hardhat` bounding box
            should be removed.
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
        with a Safety Vest bounding box

        Args:
            self: Instance of the test class.

        Raises:
            AssertionError: If the filtered data
                does not match the expected output.

        Test Scenario:
            - A `No-Safety Vest` bounding box (large box) contains a
            `Safety Vest` bounding box (smaller box).
            - Expected behaviour: The `Safety Vest` bounding box
            should be removed.

        Example:
            Input:
                datas = [
                    [10, 10, 50, 50, 0.85, 4],  # No-Safety Vest (large box)
                    [20, 20, 30, 30, 0.9, 7],   # Safety Vest (contained box)
                ]
            Output:
                filtered_datas = [
                    [10, 10, 50, 50, 0.85, 4],
                ]
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

    @patch('src.live_stream_detection.cv2.VideoCapture')
    async def test_run_detection_stream_not_opened(
        self, mock_VideoCapture: MagicMock,
    ) -> None:
        """
        Test the run_detection method when the stream cannot be opened.

        Args:
            mock_VideoCapture (MagicMock): Mock for cv2.VideoCapture

        Raises:
            AssertionError: If the exception message
                does not match the expected value.

        Test Cases:
            1. Simulates a failure to open the video stream.
            2. Ensures the method raises a ValueError
                with the correct error message.
            3. Validates that `cv2.VideoCapture`
                and its `isOpened` method are called as expected.
        """
        # Simulate the behaviour of VideoCapture
        cap_mock = MagicMock()
        cap_mock.isOpened.return_value = False  # Simulate stream open failure
        mock_VideoCapture.return_value = cap_mock

        # Test URL for the stream
        stream_url: str = 'http://example.com/fake_stream'

        # Ensure ValueError is raised when the stream cannot be opened
        with self.assertRaises(ValueError) as context:
            await self.detector.run_detection(stream_url)

        # Verify the exception message is as expected
        self.assertEqual(
            str(context.exception),
            'Failed to open stream.',
            'The exception message should match the expected value.',
        )

        # Validate that VideoCapture was called with the correct URL
        mock_VideoCapture.assert_called_once_with(stream_url)

        # Validate that `isOpened` was called on the mocked VideoCapture object
        cap_mock.isOpened.assert_called_once()

    @patch(
        'sys.argv', [
            'main', '--url',
            'http://example.com/virtual_stream', '--detect_with_server',
            '--api_url', 'http://127.0.0.1:8001',
        ],
    )
    @patch('src.live_stream_detection.LiveStreamDetector.run_detection')
    async def test_main(self, mock_run_detection: MagicMock) -> None:
        """
        Test the main function.

        Args:
            mock_run_detection (MagicMock): Mock for
                LiveStreamDetector.run_detection.
        """
        with patch(
            'src.live_stream_detection.LiveStreamDetector.__init__',
            return_value=None,
        ) as mock_init:
            mock_init.return_value = None
            await main()
            mock_init.assert_called_once_with(
                api_url='http://127.0.0.1:8001',
                model_key='yolo11n',
                output_folder=None,
                detect_with_server=True,
                shared_token={'access_token': None, 'token_expiry': 0},
            )
            mock_run_detection.assert_called_once_with(
                'http://example.com/virtual_stream',
            )


if __name__ == '__main__':
    unittest.main()
