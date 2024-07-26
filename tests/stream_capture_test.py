from __future__ import annotations

import unittest

from src.stream_capture import StreamCapture


class TestStreamCapture(unittest.TestCase):
    """
    Unit tests for the StreamCapture class methods.
    """

    stream_capture: StreamCapture

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the StreamCapture instance for tests.
        """
        url: str = 'tests/videos/test.mp4'
        capture_interval: int = 15
        cls.stream_capture = StreamCapture(url, capture_interval)

    def test_initialisation(self) -> None:
        """
        Test if the StreamCapture instance is initialised correctly.
        """
        self.assertIsInstance(self.stream_capture, StreamCapture)

    def test_capture_interval_update(self) -> None:
        """
        Test updating the capture interval.
        """
        new_interval: int = 10
        self.stream_capture.update_capture_interval(new_interval)
        self.assertEqual(self.stream_capture.capture_interval, new_interval)

    def test_select_quality_based_on_speed(self) -> None:
        """
        Test selecting stream quality based on internet speed.
        """
        stream_url: str | None = (
            self.stream_capture.select_quality_based_on_speed()
        )
        self.assertIsInstance(stream_url, (str, type(None)))

    def test_check_internet_speed(self) -> None:
        """
        Test checking internet speed.
        """
        download_speed: float
        upload_speed: float
        download_speed, upload_speed = (
            self.stream_capture.check_internet_speed()
        )
        self.assertIsInstance(download_speed, float)
        self.assertIsInstance(upload_speed, float)
        self.assertGreaterEqual(download_speed, 0)
        self.assertGreaterEqual(upload_speed, 0)

    def test_execute_capture(self) -> None:
        """
        Test executing the capture process.
        """
        generator = self.stream_capture.execute_capture()
        self.assertTrue(hasattr(generator, '__iter__'))

        # Test the first frame and timestamp
        frame, timestamp = next(generator)
        self.assertIsNotNone(frame)
        self.assertIsInstance(timestamp, float)

        # Release resources
        del frame
        self.stream_capture.release_resources()

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Teardown or release resources if needed.
        """
        cls.stream_capture.release_resources()


if __name__ == '__main__':
    unittest.main()
