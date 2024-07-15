from __future__ import annotations

import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

from src.live_stream_detection import LiveStreamDetector


class TestLiveStreamDetector(unittest.TestCase):
    def setUp(self):
        self.api_url = 'http://localhost:5000'
        self.model_key = 'yolov8n'
        self.output_folder = 'test_output'
        self.run_local = True
        self.detector = LiveStreamDetector(
            api_url=self.api_url,
            model_key=self.model_key,
            output_folder=self.output_folder,
            run_local=self.run_local,
        )

    @patch(
        'src.live_stream_detection.LiveStreamDetector.requests_retry_session',
    )
    @patch('src.live_stream_detection.ImageFont.truetype')
    def test_initialization(self, mock_truetype, mock_requests_retry_session):
        mock_requests_retry_session.return_value = MagicMock()
        detector = LiveStreamDetector(
            api_url=self.api_url,
            model_key=self.model_key,
            output_folder=self.output_folder,
            run_local=self.run_local,
        )
        self.assertEqual(detector.api_url, self.api_url)
        self.assertEqual(detector.model_key, self.model_key)
        self.assertEqual(detector.output_folder, self.output_folder)
        self.assertEqual(detector.run_local, self.run_local)
        self.assertIsNotNone(detector.session)
        self.assertIsNotNone(detector.font)
        self.assertEqual(detector.access_token, None)
        self.assertEqual(detector.token_expiry, 0.0)

    @patch('src.live_stream_detection.cv2.VideoCapture')
    @patch(
        'src.live_stream_detection.AutoDetectionModel.from_pretrained',
    )
    def test_generate_detections_local(
        self, mock_from_pretrained, mock_video_capture,
    ):
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_result = MagicMock()
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

        datas = self.detector.generate_detections_local(frame)

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
        self, mock_generate_detections, mock_destroyAllWindows,
    ):
        mock_generate_detections.return_value = (
            [], np.zeros((480, 640, 3), dtype=np.uint8),
        )
        stream_url = 'https://cctv6.kctmc.nat.gov.tw/ea05668e/'
        cap_mock = MagicMock()
        cap_mock.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)), (
                True, np.zeros((480, 640, 3), dtype=np.uint8),
            ), (False, None),
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
    def test_generate_detections_cloud(self, mock_authenticate, mock_post):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_response = MagicMock()
        mock_response.json.return_value = [
            [10, 10, 50, 50, 0.9, 0], [20, 20, 60, 60, 0.8, 1],
        ]
        mock_post.return_value = mock_response

        datas = self.detector.generate_detections_cloud(frame)

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
    def test_authenticate(self, mock_post, mock_ensure_authenticated):
        mock_response = MagicMock()
        mock_response.json.return_value = {'access_token': 'fake_token'}
        mock_post.return_value = mock_response

        self.detector.authenticate()

        self.assertEqual(self.detector.access_token, 'fake_token')
        self.assertGreater(self.detector.token_expiry, time.time())

    def test_token_expired(self):
        self.detector.token_expiry = time.time() - 1
        self.assertTrue(self.detector.token_expired())

        self.detector.token_expiry = time.time() + 1000
        self.assertFalse(self.detector.token_expired())

    def test_ensure_authenticated(self):
        with patch.object(self.detector, 'token_expired', return_value=True):
            with patch.object(
                self.detector, 'authenticate',
            ) as mock_authenticate:
                self.detector.ensure_authenticated()
                mock_authenticate.assert_called_once()

    def test_draw_detections_on_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        datas = [[10, 10, 50, 50, 0.9, 0], [20, 20, 60, 60, 0.8, 1]]

        result_frame = self.detector.draw_detections_on_frame(frame, datas)

        self.assertIsInstance(result_frame, np.ndarray)

    def test_save_frame(self):
        frame_bytes = bytearray(
            np.zeros((480, 640, 3), dtype=np.uint8).tobytes(),
        )
        output_filename = 'test_frame'

        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            self.detector.save_frame(frame_bytes, output_filename)
            mock_file.assert_called_once_with(
                Path('detected_frames/test_output/test_frame.png'), 'wb',
            )
            mock_file().write.assert_called_once_with(frame_bytes)


if __name__ == '__main__':
    unittest.main()
