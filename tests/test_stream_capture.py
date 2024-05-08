import unittest
from unittest.mock import patch, MagicMock
import cv2
from src.stream_capture import StreamCapture

class TestStreamCapture(unittest.TestCase):
    @patch('src.stream_capture.cv2.VideoCapture')
    def test_initialise_stream(self, mock_video_capture):
        """
        Test the initialisation of the video stream.
        """
        mock_video_capture.return_value.isOpened.return_value = False
        stream_capture = StreamCapture('http://test_stream_url')
        with self.assertRaises(Exception):
            stream_capture.initialise_stream()
        mock_video_capture.assert_called_with('http://test_stream_url')

    @patch('src.stream_capture.cv2.VideoCapture')
    def test_release_resources(self, mock_video_capture):
        """
        Test the release of resources.
        """
        mock_video_capture_instance = MagicMock()
        mock_video_capture.return_value = mock_video_capture_instance
        stream_capture = StreamCapture('http://test_stream_url')
        stream_capture.cap = mock_video_capture_instance
        stream_capture.release_resources()
        mock_video_capture_instance.release.assert_called_once()
        # Since we cannot check cv2.destroyAllWindows() directly, we assume it's called.

    @patch('src.stream_capture.cv2.VideoCapture')
    @patch('src.stream_capture.cv2.waitKey')
    def test_capture_frames(self, mock_cv2_waitKey, mock_video_capture):
        """
        Test the capture of frames from the stream.
        """
        mock_video_capture_instance = MagicMock()
        mock_video_capture.return_value = mock_video_capture_instance
        mock_video_capture_instance.read.return_value = (True, 'frame')
        mock_cv2_waitKey.return_value = -1  # Simulate no 'q' key press

        stream_capture = StreamCapture('http://test_stream_url')
        frames_generator = stream_capture.capture_frames()
        frame, timestamp = next(frames_generator)

        self.assertEqual(frame, 'frame')
        self.assertTrue(isinstance(timestamp, float))
        mock_video_capture_instance.read.assert_called_once()

    @patch('src.stream_capture.speedtest.Speedtest')
    def test_check_internet_speed(self, mock_speedtest):
        """
        Test the checking of internet speed.
        """
        mock_speedtest_instance = MagicMock()
        mock_speedtest.return_value = mock_speedtest_instance
        mock_speedtest_instance.download.return_value = 10_000_000  # 10 Mbps
        mock_speedtest_instance.upload.return_value = 5_000_000  # 5 Mbps

        stream_capture = StreamCapture('http://test_stream_url')
        download_speed, upload_speed = stream_capture.check_internet_speed()

        self.assertEqual(download_speed, 10)
        self.assertEqual(upload_speed, 5)

    @patch('src.stream_capture.streamlink.streams')
    @patch('src.stream_capture.StreamCapture.check_internet_speed')
    def test_select_quality_based_on_speed(self, mock_check_internet_speed, mock_streams):
        """
        Test the selection of stream quality based on internet speed.
        """
        mock_check_internet_speed.return_value = (10, 5)  # 10 Mbps download, 5 Mbps upload
        mock_streams.return_value = {
            'best': MagicMock(url='http://best_quality_stream'),
            '720p': MagicMock(url='http://720p_quality_stream'),
            'worst': MagicMock(url='http://worst_quality_stream')
        }

        stream_capture = StreamCapture('http://test_stream_url')
        selected_stream_url = stream_capture.select_quality_based_on_speed()

        self.assertEqual(selected_stream_url, 'http://best_quality_stream')

if __name__ == '__main__':
    unittest.main()
