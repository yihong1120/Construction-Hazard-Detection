from __future__ import annotations

import argparse
import sys
import time
import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

from src.stream_capture import main as stream_capture_main
from src.stream_capture import StreamCapture


class TestStreamCapture(IsolatedAsyncioTestCase):
    """
    Tests for the StreamCapture class.
    """

    def setUp(self) -> None:
        """Set up a StreamCapture instance for use in tests."""
        # Initialise StreamCapture instance with a presumed stream URL
        self.stream_capture: StreamCapture = StreamCapture(
            'http://example.com/stream',
        )

    @patch('cv2.VideoCapture')
    async def test_initialise_stream_success(
        self,
        mock_video_capture: MagicMock,
    ) -> None:
        """
        Test that the stream is successfully initialised.

        Args:
            mock_video_capture (MagicMock): Mock for cv2.VideoCapture.
        """
        # Mock VideoCapture object's isOpened method to
        # return True, indicating the stream opened successfully
        mock_video_capture.return_value.isOpened.return_value = True

        # Call initialise_stream method to initialise the stream
        await self.stream_capture.initialise_stream(
            self.stream_capture.stream_url,
        )

        # Assert that the cap object is successfully initialised
        self.assertIsNotNone(self.stream_capture.cap)

        # Verify that VideoCapture was called correctly
        mock_video_capture.assert_called_once_with(
            self.stream_capture.stream_url,
        )

        # Release resources
        await self.stream_capture.release_resources()

    @patch('cv2.VideoCapture')
    async def test_execute_capture_cap_is_none(
        self,
        mock_video_capture: MagicMock,
    ) -> None:
        """
        Test that the generator reinitialises `self.cap` if it is `None`.

        Args:
            mock_video_capture (MagicMock): Mock object for `cv2.VideoCapture`.

        Raises:
            AssertionError: If the test conditions are not met.
        """
        # Mock `isOpened()` to always return True
        mock_video_capture.return_value.isOpened.return_value = True

        # Set `read()` to return two frames successfully
        mock_frame = MagicMock(name='MockFrame')
        mock_video_capture.return_value.read.side_effect = [
            (True, mock_frame),  # First iteration
            (True, mock_frame),  # Second iteration
        ]

        # Set `capture_interval` to 0
        # to allow immediate yielding during each iteration
        self.stream_capture.capture_interval = 0

        # Start the asynchronous generator for `execute_capture`
        generator = self.stream_capture.execute_capture()

        # First call to `__anext__`: `self.cap` should not be `None`
        frame1, ts1 = await generator.__anext__()
        self.assertIsNotNone(frame1, 'First frame should not be None')

        # Manually set `self.cap` to `None`
        # to trigger the `if self.cap is None` branch
        self.stream_capture.cap = None

        # Second call to `__anext__`: The branch `if self.cap is None`
        # should execute and reinitialise `self.cap`
        frame2, ts2 = await generator.__anext__()
        self.assertIsNotNone(
            frame2, 'Second frame should not be None after reinitialisation',
        )

        # Close the generator to avoid warnings
        await generator.aclose()

        # Verify that `cv2.VideoCapture` was called twice
        mock_video_capture.assert_called_with(self.stream_capture.stream_url)

    @patch('cv2.VideoCapture')
    @patch.object(StreamCapture, 'capture_generic_frames')
    async def test_execute_capture_switch_to_generic(
        self,
        mock_capture_generic: MagicMock,
        mock_video_capture: MagicMock,
    ) -> None:
        """
        Test that the generator switches to `capture_generic_frames`
        after 5 consecutive failures.

        Args:
            mock_capture_generic (MagicMock): Mock for
                the `capture_generic_frames` method.
            mock_video_capture (MagicMock): Mock for
                the `cv2.VideoCapture` object.

        Returns:
            None
        """
        # Mock VideoCapture object's read method to return False 5 times
        mock_video_capture.return_value.read.side_effect = [(False, None)] * 6
        mock_video_capture.return_value.isOpened.return_value = True

        # Set capture interval to 0 to avoid delays during the test.
        self.stream_capture.successfully_captured = False

        # Start the coroutine generator.
        async def mock_generic():
            for i in range(2):
                frame_mock = MagicMock()
                # Explicitly set frame name
                frame_mock.name = f"GenericFrame{i}"
                yield (frame_mock, float(i))
        mock_capture_generic.side_effect = mock_generic

        # Set capture interval to 0 to avoid delays during the test.
        self.stream_capture.capture_interval = 0

        # Start the coroutine generator.
        gen = self.stream_capture.execute_capture()

        # First frame: Validate first frame
        # from `capture_generic_frames`.
        generic_frame_0, ts_0 = await gen.__anext__()
        self.assertEqual(generic_frame_0.name, 'GenericFrame0')

        # Second frame: Validate subsequent
        # frame from `generic_frames`.
        generic_frame_1, ts_1 = await gen.__anext__()
        self.assertEqual(generic_frame_1.name, 'GenericFrame1')

        # After `generic_frames` iteration completes,
        # the generator should raise `StopAsyncIteration`.
        with self.assertRaises(StopAsyncIteration):
            await gen.__anext__()

    @patch('cv2.VideoCapture')
    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_initialise_stream_retry(
        self,
        mock_sleep: AsyncMock,
        mock_video_capture: MagicMock,
    ) -> None:
        """
        Test that the stream initialisation retries if it fails initially.

        Args:
            mock_sleep (AsyncMock): Mock for asyncio.sleep.
            mock_video_capture (MagicMock): Mock for cv2.VideoCapture.
        """
        # Mock VideoCapture object's isOpened method to
        # return False on the first call and True on the second
        instance = mock_video_capture.return_value
        instance.isOpened.side_effect = [False, True]

        # Call initialise_stream method to simulate retry mechanism
        await self.stream_capture.initialise_stream(
            self.stream_capture.stream_url,
        )

        # Assert that the cap object is eventually successfully initialised
        self.assertIsNotNone(self.stream_capture.cap)

        # Verify that sleep method was called once to wait before retrying
        mock_sleep.assert_called_once_with(5)

    async def test_release_resources(self) -> None:
        """
        Test that resources are released correctly.
        """
        # Initialise StreamCapture instance and mock cap object
        stream_capture: StreamCapture = StreamCapture('test_stream_url')
        stream_capture.cap = MagicMock()

        # Call release_resources method to release resources
        await stream_capture.release_resources()

        # Assert that cap object is set to None
        self.assertIsNone(stream_capture.cap)

    @patch('cv2.VideoCapture')
    @patch('cv2.Mat')
    @patch('time.sleep', return_value=None)
    async def test_execute_capture(
        self,
        mock_sleep: MagicMock,
        mock_mat: MagicMock,
        mock_video_capture: MagicMock,
    ) -> None:
        """
        Test that frames are captured and returned with a timestamp.

        Args:
            mock_sleep (MagicMock): Mock for time.sleep.
            mock_video_capture (MagicMock): Mock for cv2.VideoCapture.
        """
        # Mock VideoCapture object's read method to
        # return a frame and True indicating successful read
        mock_video_capture.return_value.read.return_value = (True, mock_mat)
        mock_video_capture.return_value.isOpened.return_value = True

        # Execute capture frame generator and get the first frame and timestamp
        generator = self.stream_capture.execute_capture()
        frame, timestamp = await generator.__anext__()

        # Assert that the captured frame is not None
        # and the timestamp is a float
        self.assertIsNotNone(frame)
        self.assertIsInstance(timestamp, float)

        # Release resources
        await self.stream_capture.release_resources()

    @patch('speedtest.Speedtest')
    def test_check_internet_speed(self, mock_speedtest: MagicMock) -> None:
        """
        Test that internet speed is correctly checked and returned.

        Args:
            mock_speedtest (MagicMock): Mock for speedtest.Speedtest.
        """
        # Mock Speedtest object's download and upload methods
        # to return download and upload speeds
        mock_speedtest.return_value.download.return_value = 50_000_000
        mock_speedtest.return_value.upload.return_value = 10_000_000

        # Check internet speed and assert that
        # the returned speeds are correct
        download_speed, upload_speed = (
            self.stream_capture.check_internet_speed()
        )
        self.assertEqual(download_speed, 50.0)
        self.assertEqual(upload_speed, 10.0)

    @patch('streamlink.streams')
    def test_select_quality_based_on_speed_high_speed(
        self,
        mock_streams: MagicMock,
    ) -> None:
        """
        Test that the highest quality stream is selected
        for high internet speed.

        Args:
            mock_streams (MagicMock): Mock for streamlink.streams.
        """
        # Mock streamlink to return different quality streams
        mock_streams.return_value = {
            'best': MagicMock(url='http://best.stream'),
            '1080p': MagicMock(url='http://1080p.stream'),
            '720p': MagicMock(url='http://720p.stream'),
            '480p': MagicMock(url='http://480p.stream'),
        }

        # Mock internet speed check result
        with patch.object(
            self.stream_capture,
            'check_internet_speed',
            return_value=(20, 5),
        ):
            # Select the best stream quality based on internet speed
            selected_quality = (
                self.stream_capture.select_quality_based_on_speed()
            )
            self.assertEqual(selected_quality, 'http://best.stream')

    @patch('streamlink.streams')
    def test_select_quality_based_on_speed_medium_speed(
        self,
        mock_streams: MagicMock,
    ) -> None:
        """
        Test that an appropriate quality stream is selected
        for medium internet speed.

        Args:
            mock_streams (MagicMock): Mock for streamlink.streams.
        """
        # Mock streamlink to return medium quality streams
        mock_streams.return_value = {
            '720p': MagicMock(url='http://720p.stream'),
            '480p': MagicMock(url='http://480p.stream'),
            '360p': MagicMock(url='http://360p.stream'),
        }

        # Mock internet speed check result
        with patch.object(
            self.stream_capture,
            'check_internet_speed',
            return_value=(7, 5),
        ):
            # Select the appropriate stream quality based on internet speed
            selected_quality = (
                self.stream_capture.select_quality_based_on_speed()
            )
            self.assertEqual(selected_quality, 'http://720p.stream')

    @patch('streamlink.streams')
    def test_select_quality_based_on_speed_low_speed(
        self,
        mock_streams: MagicMock,
    ) -> None:
        """
        Test that a lower quality stream is selected for low internet speed.

        Args:
            mock_streams (MagicMock): Mock for streamlink.streams.
        """
        # Mock streamlink to return low quality streams
        mock_streams.return_value = {
            '480p': MagicMock(url='http://480p.stream'),
            '360p': MagicMock(url='http://360p.stream'),
            '240p': MagicMock(url='http://240p.stream'),
        }

        # Mock internet speed check result
        with patch.object(
            self.stream_capture,
            'check_internet_speed',
            return_value=(3, 5),
        ):
            # Select the lower quality stream based on internet speed
            selected_quality = (
                self.stream_capture.select_quality_based_on_speed()
            )
            self.assertEqual(selected_quality, 'http://480p.stream')

    @patch('streamlink.streams', return_value={})
    @patch.object(StreamCapture, 'check_internet_speed', return_value=(20, 5))
    def test_select_quality_based_on_speed_no_quality(
        self,
        mock_check_speed: MagicMock,
        mock_streams: MagicMock,
    ) -> None:
        """
        Test that None is returned if no suitable stream quality is available.

        Args:
            mock_check_speed (MagicMock): Mock for check_internet_speed method.
            mock_streams (MagicMock): Mock for streamlink.streams.
        """
        # Mock internet speed and stream quality check result to be empty
        selected_quality = self.stream_capture.select_quality_based_on_speed()
        self.assertIsNone(selected_quality)

    @patch(
        'streamlink.streams', return_value={
            'best': MagicMock(url='http://best.stream'),
            '720p': MagicMock(url='http://720p.stream'),
            '480p': MagicMock(url='http://480p.stream'),
        },
    )
    @patch.object(StreamCapture, 'check_internet_speed', return_value=(20, 5))
    @patch('cv2.VideoCapture')
    @patch('time.sleep', return_value=None)
    async def test_capture_generic_frames(
        self,
        mock_sleep: MagicMock,
        mock_video_capture: MagicMock,
        mock_check_speed: MagicMock,
        mock_streams: MagicMock,
    ) -> None:
        """
        Test that generic frames are captured and returned with a timestamp.

        Args:
            mock_sleep (MagicMock): Mock for time.sleep.
            mock_video_capture (MagicMock): Mock for cv2.VideoCapture.
            mock_check_speed (MagicMock): Mock for check_internet_speed method.
            mock_streams (MagicMock): Mock for streamlink.streams.
        """
        # Mock VideoCapture object's behaviour
        mock_video_capture.return_value.read.return_value = (True, MagicMock())
        mock_video_capture.return_value.isOpened.return_value = True

        # Execute capture frame generator
        generator = self.stream_capture.capture_generic_frames()
        frame, timestamp = await generator.__anext__()

        # Verify the returned frame and timestamp
        self.assertIsNotNone(frame)
        self.assertIsInstance(timestamp, float)

        # Release resources
        await self.stream_capture.release_resources()

    def test_update_capture_interval(self) -> None:
        """
        Test that the capture interval is updated correctly.
        """
        # Update capture interval and verify
        self.stream_capture.update_capture_interval(20)
        self.assertEqual(self.stream_capture.capture_interval, 20)

    @patch('argparse.ArgumentParser.parse_args')
    async def test_main_function(
        self,
        mock_parse_args: MagicMock,
    ) -> None:
        """
        Test that the main function correctly initialises
        and executes StreamCapture.

        Args:
            mock_parse_args (MagicMock):
                Mock for argparse.ArgumentParser.parse_args.
        """
        # Mock command line argument parsing
        mock_parse_args.return_value = argparse.Namespace(
            url='test_stream_url',
        )

        # Mock command line argument parsing
        mock_capture_instance = MagicMock()
        with patch(
            'src.stream_capture.StreamCapture',
            return_value=mock_capture_instance,
        ):
            with patch.object(
                sys, 'argv', ['stream_capture.py', '--url', 'test_stream_url'],
            ):
                await stream_capture_main()
            mock_capture_instance.execute_capture.assert_called_once()

    @patch('cv2.VideoCapture')
    @patch('time.sleep', return_value=None)
    async def test_execute_capture_failures(
        self,
        mock_sleep: MagicMock,
        mock_video_capture: MagicMock,
    ) -> None:
        """
        Test that execute_capture handles multiple failures before success.

        Args:
            mock_sleep (MagicMock): Mock for time.sleep.
            mock_video_capture (MagicMock): Mock for cv2.VideoCapture.
        """
        # Mock VideoCapture object's multiple failures and one success read
        instance: MagicMock = mock_video_capture.return_value
        instance.read.side_effect = [(False, None)] * 5 + [(True, MagicMock())]
        instance.isOpened.return_value = True

        # Mock capture_generic_frames method and execute
        async def mock_generic_frames():
            for _ in range(3):
                yield np.zeros((480, 640, 3), dtype=np.uint8), time.time()

        with patch.object(
            self.stream_capture,
            'capture_generic_frames',
            side_effect=mock_generic_frames,
        ):
            generator = self.stream_capture.execute_capture()
            frame, timestamp = await generator.__anext__()

            # Assert that a frame and timestamp were returned
            self.assertIsInstance(frame, np.ndarray)
            self.assertIsInstance(timestamp, float)

    @patch.object(
        StreamCapture,
        'select_quality_based_on_speed',
        return_value=None,
    )
    async def test_capture_generic_frames_no_quality(
        self, mock_quality: MagicMock,
    ) -> None:
        """
        Test that capture_generic_frames handles no suitable quality.

        Args:
            mock_quality (MagicMock): Mock for
                select_quality_based_on_speed method.
        """
        async for _ in self.stream_capture.capture_generic_frames():
            self.fail('No frame should be yielded when quality is None.')

    @patch('speedtest.Speedtest')
    @patch('streamlink.streams', side_effect=Exception('Streamlink error'))
    def test_select_quality_based_on_speed_exception(
        self,
        mock_streams: MagicMock,
        mock_speedtest: MagicMock,
    ) -> None:
        # Mock Speedtest object's download and upload methods
        mock_speedtest.return_value.get_best_server.return_value = {}
        mock_speedtest.return_value.download.return_value = 20_000_000
        mock_speedtest.return_value.upload.return_value = 5_000_000

        selected_quality = self.stream_capture.select_quality_based_on_speed()
        self.assertIsNone(selected_quality)

    @patch('cv2.VideoCapture')
    @patch.object(
        StreamCapture,
        'select_quality_based_on_speed',
        return_value='http://example.com/stream',
    )
    async def test_generic_frame_reinitialisation_logic(
        self,
        mock_quality: MagicMock,
        mock_video_capture: MagicMock,
    ) -> None:
        """
        Test that the generic frame capture handles reinitialisation correctly
        after multiple consecutive failures.
        """
        # Set up the StreamCapture instance with a capture interval of 0
        self.stream_capture = StreamCapture(
            'http://example.com/stream', capture_interval=0,
        )

        # Mock VideoCapture object's read method to
        # return False 5 times and then True
        mock_video_capture.return_value.read.side_effect = (
            [(False, None)] * 5 + [(True, MagicMock())] * 5
        )
        mock_video_capture.return_value.isOpened.return_value = True

        # Use the generic frame capture method to
        # get the first frame and timestamp
        generator = self.stream_capture.capture_generic_frames()
        frame, timestamp = await generator.__anext__()

        self.assertIsNotNone(
            frame, 'Frame should not be None after reinitialisation',
        )
        self.assertIsInstance(timestamp, float, 'Timestamp should be a float')

    @patch('cv2.VideoCapture')
    @patch.object(
        StreamCapture,
        'select_quality_based_on_speed',
        return_value=None,
    )
    async def test_generic_frame_no_quality(
        self,
        mock_quality: MagicMock,
        mock_video_capture: MagicMock,
    ) -> None:
        """
        Test that capture_generic_frames skips iterations
        when no quality is available.

        Args:
            mock_quality (MagicMock): Mock for select_quality_based_on_speed.
            mock_video_capture (MagicMock): Mock for cv2.VideoCapture.
        """
        # Iterate over the generator and ensure no frames are yielded
        async for _ in self.stream_capture.capture_generic_frames():
            self.fail('No frames should be yielded when quality is None')

        # Verify that VideoCapture was not called
        mock_video_capture.assert_not_called()


if __name__ == '__main__':
    unittest.main()
