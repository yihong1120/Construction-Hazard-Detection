from __future__ import annotations

import argparse
import asyncio
import datetime
import gc
import os
import time
from collections.abc import AsyncGenerator
from typing import TypedDict
from typing import TypeGuard
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import cv2
import numpy as np
import speedtest  # type: ignore[import-untyped]
import streamlink


def _is_rtsp_url(value: str) -> bool:
    """Return whether a URL uses an RTSP scheme."""
    return value.lower().startswith(('rtsp://', 'rtsps://'))


def _redact_stream_url(value: str) -> str:
    """Redact credentials from a stream URL for logs.

    Args:
        value: Stream URL to sanitise.

    Returns:
        URL with credentials removed.
    """
    try:
        parts = urlsplit(value)
    except ValueError:
        return '<invalid-url>'
    if not parts.username and not parts.password:
        return value
    host = parts.hostname or ''
    if parts.port:
        host = f'{host}:{parts.port}'
    return urlunsplit((
        parts.scheme,
        f'<credentials>@{host}',
        parts.path,
        parts.query,
        parts.fragment,
    ))


def _nonnegative_float_env(name: str, default: float) -> float:
    """Read a non-negative floating-point environment setting."""
    try:
        return max(0.0, float(os.getenv(name, str(default))))
    except ValueError:
        return default


class InputData(TypedDict):
    """Input payload for a stream capture task."""

    stream_url: str
    capture_interval: float


class ResultData(TypedDict):
    """Captured frame payload emitted by the stream capture task."""

    frame: np.ndarray
    timestamp: float


class StreamCapture:
    """Capture frames from a live or local video stream."""

    def __init__(
        self,
        stream_url: str,
        capture_interval: float = 15,
    ) -> None:
        """Initialise the stream capture.

        Args:
            stream_url: URL or local path of the video stream.
            capture_interval: Delay between captured frames in seconds.
        """
        self.stream_url = stream_url

        # Set OpenCV FFMPEG options for RTSP streams to use TCP transport.
        os.environ.setdefault(
            'OPENCV_FFMPEG_CAPTURE_OPTIONS',
            'rtsp_transport;tcp|stimeout;5000000|max_delay;5000000',
        )

        self.cap: cv2.VideoCapture | None = None
        self.capture_interval: float = capture_interval
        self.successfully_captured = False
        self.reopen_delay = float(
            os.getenv('STREAM_CAPTURE_REOPEN_DELAY_SECONDS', '5.0'),
        )
        self.max_reopen_delay = float(
            os.getenv(
                'STREAM_CAPTURE_MAX_REOPEN_DELAY_SECONDS',
                '60.0',
            ),
        )
        self.freeze_reconnect_seconds = _nonnegative_float_env(
            'STREAM_CAPTURE_FREEZE_RECONNECT_SECONDS',
            0.0,
        )
        self.freeze_sample_seconds = max(
            0.1,
            _nonnegative_float_env(
                'STREAM_CAPTURE_FREEZE_SAMPLE_SECONDS',
                1.0,
            ),
        )
        self.freeze_frame_delta = _nonnegative_float_env(
            'STREAM_CAPTURE_FREEZE_FRAME_DELTA',
            0.2,
        )
        self.timestamp_reconnect_seconds = _nonnegative_float_env(
            'STREAM_CAPTURE_TIMESTAMP_RECONNECT_SECONDS',
            30.0,
        )
        self.timestamp_sample_seconds = max(
            0.1,
            _nonnegative_float_env(
                'STREAM_CAPTURE_TIMESTAMP_SAMPLE_SECONDS',
                1.0,
            ),
        )
        self.reconnect_event = asyncio.Event()
        self._reconnecting = False
        self._freeze_last_sample: np.ndarray | None = None
        self._freeze_last_sample_at: float | None = None
        self._freeze_last_motion_at: float | None = None
        self._timestamp_last_value_ms: float | None = None
        self._timestamp_last_sample_at: float | None = None
        self._timestamp_last_progress_at: float | None = None
        self._source_timestamp_available = False

    async def initialise_stream(self, stream_url: str) -> None:
        """
        Initialises the video stream.

        Args:
            stream_url (str): The URL of the stream to initialise.
        """
        self._reset_frozen_frame_watchdog()
        self.cap = self._create_capture(stream_url)

        if not self.cap.isOpened():
            await asyncio.sleep(self.reopen_delay)
            self.cap.release()
            self.cap = self._create_capture(stream_url)

    @staticmethod
    def _create_capture(stream_url: str) -> cv2.VideoCapture:
        """Create a configured OpenCV capture object."""
        # OpenCV's FFmpeg open/read timeouts are open-only properties. Passing
        # them after VideoCapture() has connected leaves the backend's 30-second
        # defaults active during a source outage.
        cap = cv2.VideoCapture(
            stream_url,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                int(
                    float(
                        os.getenv(
                            'STREAM_CAPTURE_OPEN_TIMEOUT_MS',
                            '5000',
                        ),
                    ),
                ),
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                int(
                    float(
                        os.getenv(
                            'STREAM_CAPTURE_READ_TIMEOUT_MS',
                            '5000',
                        ),
                    ),
                ),
            ],
        )
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _reset_frozen_frame_watchdog(self) -> None:
        """Forget visual and timestamp samples from the prior connection."""
        self._freeze_last_sample = None
        self._freeze_last_sample_at = None
        self._freeze_last_motion_at = None
        self._timestamp_last_value_ms = None
        self._timestamp_last_sample_at = None
        self._timestamp_last_progress_at = None
        self._source_timestamp_available = False

    def _begin_reconnect(self) -> None:
        """Broadcast one reconnect transition for the current source outage."""
        if self._reconnecting:
            return
        self._reconnecting = True
        self.reconnect_event.set()

    def _mark_connected(self) -> None:
        """Mark the source healthy after a usable decoded frame arrives."""
        self._reconnecting = False

    @staticmethod
    def _is_usable_frame(frame: object) -> TypeGuard[np.ndarray]:
        """Return whether a decoded frame has a complete BGR image layout."""
        return (
            isinstance(frame, np.ndarray)
            and frame.dtype == np.uint8
            and frame.ndim == 3
            and frame.shape[0] > 0
            and frame.shape[1] > 0
            and frame.shape[2] == 3
            and frame.size == frame.shape[0] * frame.shape[1] * 3
        )

    def _should_reconnect_after_stalled_source_timestamp(self) -> bool:
        """Reconnect only when a usable source timestamp stops advancing.

        A decoded frame can be visually static while the camera and RTSP
        transport are healthy.  Source timestamps distinguish that case from
        a decoder that repeatedly returns a stale frame.  Some OpenCV/RTSP
        backends do not expose timestamps; those sources intentionally skip
        this watchdog instead of guessing from scene motion.
        """
        cap = self.cap
        if self.timestamp_reconnect_seconds <= 0 or cap is None:
            return False
        now = time.monotonic()
        if (
            self._timestamp_last_sample_at is not None
            and now - self._timestamp_last_sample_at
            < self.timestamp_sample_seconds
        ):
            return False
        try:
            timestamp_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
        except (AttributeError, TypeError, ValueError, cv2.error):
            self._source_timestamp_available = False
            self._timestamp_last_value_ms = None
            self._timestamp_last_sample_at = None
            self._timestamp_last_progress_at = None
            return False
        if not np.isfinite(timestamp_ms) or timestamp_ms <= 0:
            self._source_timestamp_available = False
            self._timestamp_last_value_ms = None
            self._timestamp_last_sample_at = None
            self._timestamp_last_progress_at = None
            return False

        self._source_timestamp_available = True
        previous_timestamp_ms = self._timestamp_last_value_ms
        self._timestamp_last_value_ms = timestamp_ms
        self._timestamp_last_sample_at = now
        if (
            previous_timestamp_ms is None
            or timestamp_ms != previous_timestamp_ms
        ):
            self._timestamp_last_progress_at = now
            return False
        return (
            self._timestamp_last_progress_at is not None
            and now - self._timestamp_last_progress_at
            >= self.timestamp_reconnect_seconds
        )

    def _should_reconnect_after_frozen_frame(self, frame: np.ndarray) -> bool:
        """Return whether a valid-but-frozen source frame needs a reconnect."""
        if (
            self.freeze_reconnect_seconds <= 0
            or self._source_timestamp_available
            or not isinstance(frame, np.ndarray)
            or frame.ndim < 2
        ):
            return False
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            return False
        now = time.monotonic()
        if (
            self._freeze_last_sample_at is not None
            and now - self._freeze_last_sample_at < self.freeze_sample_seconds
        ):
            return False
        sample = cv2.resize(
            frame,
            (min(64, width), min(36, height)),
            interpolation=cv2.INTER_AREA,
        )
        previous_sample = self._freeze_last_sample
        self._freeze_last_sample = sample
        self._freeze_last_sample_at = now
        if previous_sample is None or previous_sample.shape != sample.shape:
            self._freeze_last_motion_at = now
            return False
        delta = float(
            np.asarray(
                cv2.absdiff(sample, previous_sample),
                dtype=np.float64,
            ).mean(),
        )
        if delta > self.freeze_frame_delta:
            self._freeze_last_motion_at = now
            return False
        return (
            self._freeze_last_motion_at is not None
            and now - self._freeze_last_motion_at
            >= self.freeze_reconnect_seconds
        )

    async def release_resources(self) -> None:
        """
        Releases resources like the capture object.
        """
        if self.cap:
            self.cap.release()
            self.cap = None

    async def execute_capture(
        self,
    ) -> AsyncGenerator[tuple[np.ndarray, float]]:
        """
        Captures frames from the stream and yields them with timestamps.

        Yields:
            Tuple[np.ndarray, float]: The captured frame and the timestamp.
        """
        await self.initialise_stream(self.stream_url)
        last_process_time = datetime.datetime.now() - datetime.timedelta(
            seconds=self.capture_interval,
        )
        fail_count = 0  # Counter for consecutive failures
        backoff_seconds = self.reopen_delay

        while True:
            if self.cap is None:
                await self.initialise_stream(self.stream_url)

            ret, frame = (
                self.cap.read() if self.cap is not None else (False, None)
            )

            if not ret or not self._is_usable_frame(frame):
                fail_count += 1
                self._begin_reconnect()
                print(
                    'Failed to read frame, trying to reinitialise stream. '
                    f"Fail count: {fail_count}, "
                    f"source={_redact_stream_url(self.stream_url)}",
                    flush=True,
                )
                await self.release_resources()
                await asyncio.sleep(backoff_seconds)
                backoff_seconds = min(
                    self.max_reopen_delay,
                    max(self.reopen_delay, backoff_seconds * 1.5),
                )
                await self.initialise_stream(self.stream_url)
                # Switch to generic frame capture after 5 consecutive failures
                if (
                    fail_count >= 5
                    and not self.successfully_captured
                    and not _is_rtsp_url(self.stream_url)
                ):
                    print('Switching to generic frame capture method.')
                    async for generic_frame, generic_timestamp in (
                        self.capture_generic_frames()
                    ):
                        yield generic_frame, generic_timestamp
                    return
                continue
            else:
                # Reset fail count on successful read
                fail_count = 0
                backoff_seconds = self.reopen_delay

                # Mark as successfully captured
                self.successfully_captured = True

                if (
                    self._should_reconnect_after_stalled_source_timestamp()
                    or self._should_reconnect_after_frozen_frame(frame)
                ):
                    print(
                        'Capture watchdog detected a stalled source; '
                        'reconnecting stream. '
                        f'source={_redact_stream_url(self.stream_url)}',
                        flush=True,
                    )
                    self._begin_reconnect()
                    await self.release_resources()
                    await asyncio.sleep(self.reopen_delay)
                    await self.initialise_stream(self.stream_url)
                    continue
                self._mark_connected()

            # Process the frame if the capture interval has elapsed
            current_time = datetime.datetime.now()
            elapsed_time = (current_time - last_process_time).total_seconds()

            # If the capture interval has elapsed, yield the frame
            if elapsed_time >= self.capture_interval:
                last_process_time = current_time
                timestamp = current_time.timestamp()
                yield frame, timestamp

            await asyncio.sleep(0.01)  # Adjust the sleep time as needed

        await self.release_resources()

    def check_internet_speed(self) -> tuple[float, float]:
        """
        Checks internet speed using the Speedtest library.

        Returns:
            Tuple[float, float]: Download and upload speeds (Mbps).
        """
        st = speedtest.Speedtest()
        st.get_best_server()
        download_speed = st.download() / 1_000_000  # Turn into Mbps
        upload_speed = st.upload() / 1_000_000
        return download_speed, upload_speed

    def select_quality_based_on_speed(self) -> str | None:
        """
        Selects stream quality based on internet speed.

        Returns:
            str: The URL of the selected stream quality.

        Raises:
            Exception: If compatible stream quality is not available.
        """
        download_speed, _ = self.check_internet_speed()
        try:
            streams = streamlink.streams(self.stream_url)
            available_qualities = list(streams.keys())
            print(f"Available qualities: {available_qualities}")

            if download_speed > 10:
                preferred_qualities = [
                    'best',
                    '1080p',
                    '720p',
                    '480p',
                    '360p',
                    '240p',
                    'worst',
                ]
            elif 5 < download_speed <= 10:
                preferred_qualities = ['720p', '480p', '360p', '240p', 'worst']
            else:
                preferred_qualities = ['480p', '360p', '240p', 'worst']

            for quality in preferred_qualities:
                if quality in available_qualities:
                    selected_stream = streams[quality]
                    print(f"Selected quality based on speed: {quality}")
                    return selected_stream.url

            raise Exception('No compatible stream quality is available.')
        except Exception as e:
            print(f"Error selecting quality based on speed: {e}")
            return None

    async def capture_generic_frames(
        self,
    ) -> AsyncGenerator[tuple[np.ndarray, float]]:
        """
        Captures frames from a generic stream.

        Yields:
            Tuple[np.ndarray, float]: The captured frame and the timestamp.
        """
        # Select the stream quality based on internet speed
        stream_url = self.select_quality_based_on_speed()
        if not stream_url:
            print('Failed to get suitable stream quality.')
            return

        # Initialise the stream with the selected URL
        await self.initialise_stream(stream_url)

        last_process_time = datetime.datetime.now()
        fail_count = 0  # Counter for consecutive failures
        backoff_seconds = self.reopen_delay

        while True:
            # Read the frame from the stream
            ret, frame = (
                self.cap.read() if self.cap is not None else (False, None)
            )

            # Handle failed frame reads
            if not ret or not self._is_usable_frame(frame):
                fail_count += 1
                self._begin_reconnect()
                print(
                    'Failed to read frame from generic stream. '
                    f"Fail count: {fail_count}, "
                    'source='
                    f'{_redact_stream_url(stream_url or self.stream_url)}',
                    flush=True,
                )
                await asyncio.sleep(backoff_seconds)
                backoff_seconds = min(
                    self.max_reopen_delay,
                    max(self.reopen_delay, backoff_seconds * 1.5),
                )

                # Reinitialise the stream after 5 consecutive failures
                if fail_count >= 5 and not self.successfully_captured:
                    print('Reinitialising the generic stream.')
                    await self.release_resources()
                    await asyncio.sleep(5)
                    stream_url = self.select_quality_based_on_speed()

                    # Exit if no suitable stream quality is available
                    if not stream_url:
                        print('Failed to get suitable stream quality.')
                        continue

                    # Reinitialise the stream with the new URL
                    await self.initialise_stream(stream_url)
                    fail_count = 0
                continue
            else:
                # Reset fail count on successful read
                fail_count = 0
                backoff_seconds = self.reopen_delay

                # Mark as successfully captured
                self.successfully_captured = True
                self._mark_connected()

            current_time = datetime.datetime.now()
            elapsed_time = (current_time - last_process_time).total_seconds()

            if elapsed_time >= self.capture_interval:
                last_process_time = current_time
                timestamp = current_time.timestamp()
                yield frame, timestamp

            await asyncio.sleep(0.01)  # Adjust the sleep time as needed

    def update_capture_interval(self, new_interval: float) -> None:
        """
        Updates the capture interval.

        Args:
            new_interval (int): Frame capture interval in seconds.
        """
        self.capture_interval = new_interval


async def main() -> None:
    """Run the stream capture command-line utility."""
    parser = argparse.ArgumentParser(
        description='Capture video stream frames asynchronously.',
    )
    parser.add_argument(
        '--url',
        type=str,
        help='Live stream URL',
        required=True,
    )
    args = parser.parse_args()

    stream_capture = StreamCapture(args.url)
    async for frame, timestamp in stream_capture.execute_capture():
        # Process the frame here
        print(f"Frame at {timestamp} displayed")
        # Release the frame resources
        del frame
        gc.collect()


if __name__ == '__main__':
    asyncio.run(main())
