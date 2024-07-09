from __future__ import annotations

import datetime
import gc
import time
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import cv2

from src.stream_capture import StreamCapture


class TestStreamCapture(unittest.TestCase):
    def setUp(self):
        self.stream_url = 'https://www.youtube.com/watch?v=mf-1VZ6ewlE'
        self.capture_interval = 15
        self.stream_capture = StreamCapture(
            self.stream_url, self.capture_interval,
        )

    @patch('src.stream_capture.cv2.VideoCapture')
    def test_initialise_stream(self, mock_video_capture):
        mock_cap_instance = MagicMock()
        mock_video_capture.return_value = mock_cap_instance

        self.stream_capture.initialise_stream()

        mock_video_capture.assert_called_with(self.stream_url)
        mock_cap_instance.set.assert_any_call(cv2.CAP_PROP_BUFFERSIZE, 1)
        mock_cap_instance.set.assert_any_call(
            cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'),
        )
        self.assertTrue(mock_cap_instance.isOpened.called)

    @patch('src.stream_capture.cv2.VideoCapture')
    def test_release_resources(self, mock_video_capture):
        mock_cap_instance = MagicMock()
        mock_video_capture.return_value = mock_cap_instance
        self.stream_capture.cap = mock_cap_instance

        self.stream_capture.release_resources()

        mock_cap_instance.release.assert_called_once()
        self.assertIsNone(self.stream_capture.cap)
        gc.collect()

    @patch('src.stream_capture.cv2.VideoCapture')
    @patch('src.stream_capture.datetime')
    @patch('src.stream_capture.gc.collect')
    def test_capture_frames(
        self, mock_gc_collect, mock_datetime, mock_video_capture,
    ):
        mock_cap_instance = MagicMock()
        mock_video_capture.return_value = mock_cap_instance
        self.stream_capture.cap = mock_cap_instance

        mock_now = datetime.datetime(2023, 1, 1, 0, 0, 0)
        mock_later = mock_now + \
            datetime.timedelta(seconds=self.capture_interval + 1)
        mock_datetime.datetime.now.side_effect = [
            mock_now,
            mock_now,
            mock_later,
        ]

        mock_cap_instance.read.side_effect = [
            (True, MagicMock()), (True, MagicMock()),
        ]

        frame_generator = self.stream_capture.capture_frames()

        frame, timestamp = next(frame_generator)
        self.assertIsNotNone(frame)
        self.assertEqual(timestamp, mock_now.timestamp())

        frame, timestamp = next(frame_generator)
        self.assertIsNotNone(frame)
        self.assertEqual(timestamp, mock_later.timestamp())

        self.stream_capture.release_resources()
        mock_gc_collect.assert_called()

    @patch('src.stream_capture.speedtest.Speedtest')
    def test_check_internet_speed(self, mock_speedtest):
        mock_st_instance = MagicMock()
        mock_speedtest.return_value = mock_st_instance

        mock_st_instance.download.return_value = 100_000_000
        mock_st_instance.upload.return_value = 50_000_000

        download_speed, upload_speed = (
            self.stream_capture.check_internet_speed()
        )

        self.assertEqual(download_speed, 100.0)
        self.assertEqual(upload_speed, 50.0)

    @patch('src.stream_capture.streamlink.streams')
    @patch('src.stream_capture.StreamCapture.check_internet_speed')
    def test_select_quality_based_on_speed(
        self, mock_check_internet_speed, mock_streams,
    ):
        mock_check_internet_speed.return_value = (15.0, 5.0)
        mock_streams.return_value = {
            'best': MagicMock(url=self.stream_url),
            '1080p': MagicMock(url=self.stream_url),
            '720p': MagicMock(url=self.stream_url),
        }

        selected_quality = self.stream_capture.select_quality_based_on_speed()
        self.assertEqual(selected_quality, self.stream_url)

    @patch('src.stream_capture.streamlink.streams')
    @patch('src.stream_capture.StreamCapture.check_internet_speed')
    def test_select_quality_based_on_speed_no_quality(
        self, mock_check_internet_speed, mock_streams,
    ):
        mock_check_internet_speed.return_value = (2.0, 1.0)
        mock_streams.return_value = {
            '720p': MagicMock(url=self.stream_url),
            '480p': MagicMock(url=self.stream_url),
        }

        selected_quality = self.stream_capture.select_quality_based_on_speed()
        self.assertEqual(selected_quality, self.stream_url)

    @patch('src.stream_capture.cv2.VideoCapture')
    @patch('src.stream_capture.streamlink.streams')
    @patch('src.stream_capture.StreamCapture.check_internet_speed')
    def test_capture_youtube_frames(
        self, mock_check_internet_speed, mock_streams, mock_video_capture,
    ):
        mock_check_internet_speed.return_value = (15.0, 5.0)
        mock_streams.return_value = {
            'best': MagicMock(url=self.stream_url),
            '1080p': MagicMock(url=self.stream_url),
            '720p': MagicMock(url=self.stream_url),
        }
        mock_cap_instance = MagicMock()
        mock_video_capture.return_value = mock_cap_instance
        self.stream_capture.cap = mock_cap_instance

        mock_cap_instance.read.side_effect = [
            (True, MagicMock()), (True, MagicMock()),
        ]

        frame_generator = self.stream_capture.capture_youtube_frames()

        try:
            frame, timestamp = next(frame_generator)
            self.assertIsNotNone(frame)
            self.assertIsInstance(timestamp, float)
        except StopIteration:
            self.fail('Generator stopped unexpectedly')

        mock_now_later = datetime.datetime.now(
        ) + datetime.timedelta(seconds=self.capture_interval + 1)
        with patch(
            'src.stream_capture.datetime.datetime.now',
            return_value=mock_now_later,
        ):
            frame, timestamp = next(frame_generator)
            self.assertIsNotNone(frame)
            self.assertIsInstance(timestamp, float)

        self.stream_capture.release_resources()

    @patch('src.stream_capture.StreamCapture.capture_frames')
    def test_execute_capture(self, mock_capture_frames):
        expected_timestamp = time.time()  # 使用当前时间戳
        mock_capture_frames.return_value = iter(
            [(MagicMock(), expected_timestamp)],
        )
        frames_generator = self.stream_capture.execute_capture()

        frame, timestamp = next(frames_generator)
        self.assertIsNotNone(frame)
        self.assertAlmostEqual(
            timestamp, expected_timestamp, delta=60,
        )  # 允许更大的差异

    @patch('src.stream_capture.StreamCapture.capture_youtube_frames')
    def test_execute_capture_youtube(self, mock_capture_youtube_frames):
        self.stream_capture.stream_url = (
            'https://www.youtube.com/watch?v=mf-1VZ6ewlE'
        )
        mock_capture_youtube_frames.return_value = iter(
            [(MagicMock(), 1234567890.0)],
        )
        frames_generator = self.stream_capture.execute_capture()

        frame, timestamp = next(frames_generator)
        self.assertIsNotNone(frame)
        self.assertAlmostEqual(timestamp, 1234567890.0, delta=60)


if __name__ == '__main__':
    unittest.main()
