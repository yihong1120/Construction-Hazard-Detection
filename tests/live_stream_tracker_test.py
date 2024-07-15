from __future__ import annotations

import datetime
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

from src.live_stream_tracker import LiveStreamDetector


class TestLiveStreamDetector(unittest.TestCase):
    def setUp(self):
        self.stream_url = 'tests/videos/test.mp4'
        self.model_path = 'models/pt/best_yolov8n.pt'
        self.detector = LiveStreamDetector(self.stream_url, self.model_path)

    @patch('src.live_stream_tracker.YOLO')
    @patch('src.live_stream_tracker.cv2.VideoCapture')
    def test_initialisation(self, mock_video_capture, mock_yolo):
        mock_cap_instance = MagicMock()
        mock_video_capture.return_value = mock_cap_instance

        detector = LiveStreamDetector(self.stream_url, self.model_path)

        self.assertEqual(detector.stream_url, self.stream_url)
        self.assertEqual(detector.model_path, self.model_path)
        self.assertEqual(detector.model, mock_yolo.return_value)
        self.assertEqual(detector.cap, mock_cap_instance)

    @patch('src.live_stream_tracker.YOLO')
    @patch('src.live_stream_tracker.cv2.VideoCapture')
    @patch('src.live_stream_tracker.datetime')
    def test_generate_detections(
        self, mock_datetime, mock_video_capture, mock_yolo,
    ):
        mock_cap_instance = MagicMock()
        mock_video_capture.return_value = mock_cap_instance
        mock_cap_instance.isOpened.side_effect = [True, True, False]
        mock_cap_instance.read.side_effect = [
            (
                True, np.zeros(
                    (480, 640, 3), dtype=np.uint8,
                ),
            ), (True, np.zeros((480, 640, 3), dtype=np.uint8)), (False, None),
        ]

        mock_yolo_instance = MagicMock()
        mock_yolo.return_value = mock_yolo_instance
        mock_results = MagicMock()
        mock_boxes = MagicMock()
        mock_boxes.id = MagicMock()
        mock_boxes.data = MagicMock()
        mock_boxes.id.cpu.return_value = mock_boxes.id
        mock_boxes.data.cpu.return_value = mock_boxes.data
        mock_boxes.id.numpy.return_value = [1, 2, 3]
        mock_boxes.data.numpy.return_value = [
            [0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8],
        ]
        mock_results[0].boxes = mock_boxes
        mock_yolo_instance.track.return_value = mock_results

        mock_now = datetime.datetime(2023, 1, 1, 0, 0, 0)
        mock_datetime.datetime.now.side_effect = [mock_now, mock_now]

        frame_generator = self.detector.generate_detections()

        ids, datas, frame, timestamp = next(frame_generator)
        self.assertIsInstance(ids, list)
        self.assertIsInstance(datas, list)
        self.assertIsInstance(frame, np.ndarray)
        self.assertIsInstance(timestamp, float)

    @patch('src.live_stream_tracker.cv2.VideoCapture')
    @patch('src.live_stream_tracker.cv2.destroyAllWindows')
    def test_release_resources(
        self, mock_destroy_all_windows, mock_video_capture,
    ):
        mock_cap_instance = MagicMock()
        mock_video_capture.return_value = mock_cap_instance
        self.detector.cap = mock_cap_instance

        self.detector.release_resources()

        mock_cap_instance.release.assert_called_once()
        mock_destroy_all_windows.assert_called_once()

    @patch('src.live_stream_tracker.LiveStreamDetector.generate_detections')
    def test_run_detection(self, mock_generate_detections):
        mock_generate_detections.return_value = iter(
            [(
                [1, 2, 3], [[0.1, 0.2, 0.3, 0.4]], np.zeros(
                    (480, 640, 3), dtype=np.uint8,
                ), 1234567890.0,
            )],
        )

        with patch('builtins.print') as mock_print:
            self.detector.run_detection()
            self.assertTrue(
                any(
                    'Timestamp:' in str(call) and '2009-02-14' in str(call)
                    for call in mock_print.call_args_list
                ),
            )
            self.assertTrue(
                any(
                    'IDs:' in str(call)
                    for call in mock_print.call_args_list
                ),
            )
            self.assertTrue(
                any(
                    'Data (xyxy format):' in str(call)
                    for call in mock_print.call_args_list
                ),
            )


if __name__ == '__main__':
    unittest.main()
