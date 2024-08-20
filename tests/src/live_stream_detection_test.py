from __future__ import annotations

import time
import unittest
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import cv2
import numpy as np

from src.live_stream_detection import LiveStreamDetector
from src.live_stream_detection import main


class TestLiveStreamDetector(unittest.TestCase):
    """
    Unit tests for the LiveStreamDetector class methods.
    """

    def setUp(self) -> None:
        """
        Set up the LiveStreamDetector instance for tests.
        """
        self.api_url: str = 'http://localhost:5000'
        self.model_key: str = 'yolov8n'
        self.output_folder: str = 'test_output'
        self.run_local: bool = True
        self.detector: LiveStreamDetector = LiveStreamDetector(
            api_url=self.api_url,
            model_key=self.model_key,
            output_folder=self.output_folder,
            run_local=self.run_local,
        )

    @patch(
        'src.live_stream_detection.LiveStreamDetector.requests_retry_session',
    )
    @patch('PIL.ImageFont.truetype')
    def test_initialisation(
        self,
        mock_truetype: MagicMock,
        mock_requests_retry_session: MagicMock,
    ) -> None:
        """
        Test the initialisation of the LiveStreamDetector instance.

        Args:
            mock_truetype (MagicMock): Mock for PIL.ImageFont.truetype.
            mock_requests_retry_session (MagicMock): Mock for
                requests_retry_session.
        """
        mock_requests_retry_session.return_value = MagicMock()
        mock_truetype.return_value = MagicMock()

        detector = LiveStreamDetector(
            api_url=self.api_url,
            model_key=self.model_key,
            output_folder=self.output_folder,
            run_local=self.run_local,
        )

        # Assert initialisation values
        self.assertEqual(detector.api_url, self.api_url)
        self.assertEqual(detector.model_key, self.model_key)
        self.assertEqual(detector.output_folder, self.output_folder)
        self.assertEqual(detector.run_local, self.run_local)
        self.assertIsNotNone(detector.session)
        self.assertEqual(detector.access_token, None)
        self.assertEqual(detector.token_expiry, 0.0)

    @patch('src.live_stream_detection.cv2.VideoCapture')
    @patch('src.live_stream_detection.AutoDetectionModel.from_pretrained')
    def test_generate_detections_local(
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
        # mat_frame = cv2.Mat(frame)
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

        datas: list[list[Any]] = self.detector.generate_detections_local(
            frame,
        )

        # Assert the structure and types of the detection data
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

    @patch('src.live_stream_detection.cv2.destroyAllWindows')
    @patch('src.live_stream_detection.LiveStreamDetector.generate_detections')
    def test_run_detection(
        self,
        mock_generate_detections: MagicMock,
        mock_destroyAllWindows: MagicMock,
    ) -> None:
        """
        Test the run_detection method.

        Args:
            mock_generate_detections (MagicMock): Mock for generate_detections.
            mock_destroyAllWindows (MagicMock): Mock for cv2.destroyAllWindows.
        """
        mock_generate_detections.return_value = (
            [], np.zeros((480, 640, 3), dtype=np.uint8),
        )
        stream_url: str = 'https://cctv6.kctmc.nat.gov.tw/ea05668e/'
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
                    self.detector.run_detection(stream_url)

        cap_mock.read.assert_called()
        cap_mock.release.assert_called_once()
        mock_destroyAllWindows.assert_called()

    @patch('src.live_stream_detection.requests.Session.post')
    @patch('src.live_stream_detection.LiveStreamDetector.authenticate')
    def test_generate_detections_cloud(
        self,
        mock_authenticate: MagicMock,
        mock_post: MagicMock,
    ) -> None:
        """
        Test cloud detection generation.

        Args:
            mock_authenticate (MagicMock): Mock for authenticate.
            mock_post (MagicMock): Mock for requests.Session.post.
        """
        frame: np.ndarray = np.zeros((480, 640, 3), dtype=np.uint8)
        # mat_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        mock_response: MagicMock = MagicMock()
        mock_response.json.return_value = [
            [10, 10, 50, 50, 0.9, 0], [20, 20, 60, 60, 0.8, 1],
        ]
        mock_post.return_value = mock_response

        datas: list[list[Any]] = self.detector.generate_detections_cloud(frame)

        # Assert the structure and types of the detection data
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

    @patch('src.live_stream_detection.LiveStreamDetector.ensure_authenticated')
    @patch('src.live_stream_detection.requests.Session.post')
    def test_authenticate(
        self,
        mock_post: MagicMock,
        mock_ensure_authenticated: MagicMock,
    ) -> None:
        """
        Test the authentication process.

        Args:
            mock_post (MagicMock): Mock for requests.Session.post.
            mock_ensure_authenticated (MagicMock): Mock
                for ensure_authenticated.
        """
        mock_response: MagicMock = MagicMock()
        mock_response.json.return_value = {'access_token': 'fake_token'}
        mock_post.return_value = mock_response

        self.detector.authenticate()

        # Assert the access token and token expiry
        self.assertEqual(self.detector.access_token, 'fake_token')
        self.assertGreater(self.detector.token_expiry, time.time())

    def test_token_expired(self) -> None:
        """
        Test the token_expired method.
        """
        self.detector.token_expiry = time.time() - 1
        self.assertTrue(self.detector.token_expired())

        self.detector.token_expiry = time.time() + 1000
        self.assertFalse(self.detector.token_expired())

    def test_ensure_authenticated(self) -> None:
        """
        Test the ensure_authenticated method.
        """
        with patch.object(self.detector, 'token_expired', return_value=True):
            with patch.object(
                self.detector, 'authenticate',
            ) as mock_authenticate:
                self.detector.ensure_authenticated()
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

    @patch('src.live_stream_detection.requests.Session.post')
    def test_authenticate_error(self, mock_post: MagicMock) -> None:
        """
        Test the authenticate method with error response.

        Args:
            mock_post (MagicMock): Mock for requests.Session.post.
        """
        mock_response: MagicMock = MagicMock()
        mock_response.json.return_value = {'msg': 'Authentication failed'}
        mock_post.return_value = mock_response

        with self.assertRaises(Exception) as context:
            self.detector.authenticate()

        self.assertIn('Authentication failed', str(context.exception))

    def test_generate_detections(self) -> None:
        """
        Test the generate_detections method.
        """
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mat_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.detector.run_local = True
        with patch.object(
            self.detector, 'generate_detections_local',
            return_value=[[10, 10, 50, 50, 0.9, 0]],
        ) as mock_local:
            datas, _ = self.detector.generate_detections(mat_frame)
            self.assertEqual(len(datas), 1)
            self.assertEqual(datas[0][5], 0)
            mock_local.assert_called_once_with(mat_frame)

        self.detector.run_local = False
        with patch.object(
            self.detector, 'generate_detections_cloud',
            return_value=[[20, 20, 60, 60, 0.8, 1]],
        ) as mock_cloud:
            datas, _ = self.detector.generate_detections(mat_frame)
            self.assertEqual(len(datas), 1)
            self.assertEqual(datas[0][5], 1)
            mock_cloud.assert_called_once_with(mat_frame)

    def test_run_detection_fail_read_frame(self) -> None:
        """
        Test the run_detection method when failing to read a frame.
        """
        stream_url: str = 'https://cctv6.kctmc.nat.gov.tw/ea05668e/'
        cap_mock: MagicMock = MagicMock()
        # Simulate repeated failure to read frame
        cap_mock.read.side_effect = [(False, None)] * 10
        cap_mock.isOpened.return_value = True

        with patch(
            'src.live_stream_detection.cv2.VideoCapture',
            return_value=cap_mock,
        ):
            with patch(
                'src.live_stream_detection.cv2.waitKey',
                side_effect=[-1, -1, ord('q')],
            ):
                try:
                    self.detector.run_detection(stream_url)
                except StopIteration:
                    pass

        cap_mock.read.assert_called()
        cap_mock.release.assert_called_once()

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

    @patch(
        'sys.argv', [
            'main', '--url',
            'https://cctv6.kctmc.nat.gov.tw/ea05668e/', '--run_local',
        ],
    )
    @patch('src.live_stream_detection.LiveStreamDetector.run_detection')
    def test_main(self, mock_run_detection: MagicMock) -> None:
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
            main()
            mock_init.assert_called_once_with(
                api_url='http://localhost:5000',
                model_key='yolov8n',
                output_folder=None,
                run_local=True,
            )
            mock_run_detection.assert_called_once_with(
                'https://cctv6.kctmc.nat.gov.tw/ea05668e/',
            )


if __name__ == '__main__':
    unittest.main()
