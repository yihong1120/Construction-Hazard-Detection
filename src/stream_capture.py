from __future__ import annotations

import argparse
import asyncio
import datetime
import gc
import logging
import os
import time
from collections.abc import AsyncGenerator
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict
from typing import TypeGuard
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import cv2
import numpy as np
import speedtest  # type: ignore[import-untyped]
import streamlink

logger = logging.getLogger(__name__)


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
        host = f"{host}:{parts.port}"
    return urlunsplit(
        (
            parts.scheme,
            f"<credentials>@{host}",
            parts.path,
            parts.query,
            parts.fragment,
        ),
    )


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
        # OpenCV's FFmpeg calls can block for several seconds despite the
        # configured timeout.  One executor per capture preserves operation
        # ordering while keeping this process's publisher and lease tasks on
        # the event loop.
        self._capture_executor: ThreadPoolExecutor | None = None

    async def _run_capture_operation[T](
        self,
        operation: Callable[..., T],
        *args: object,
    ) -> T:
        """Run one blocking capture operation off this stream's event loop."""
        if self._capture_executor is None:
            self._capture_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix='stream-capture',
            )
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._capture_executor,
            operation,
            *args,
        )

    def _close_capture_executor(self) -> None:
        """Release the dedicated worker after its capture is closed."""
        executor = self._capture_executor
        self._capture_executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    async def initialise_stream(self, stream_url: str) -> None:
        """Initialises the video stream.

        Args:
            stream_url (str): The URL of the stream to initialise.
        """
        self._reset_frozen_frame_watchdog()
        capture = await self._run_capture_operation(
            self._create_capture,
            stream_url,
        )
        self.cap = capture

        if not capture.isOpened():
            await asyncio.sleep(self.reopen_delay)
            await self._run_capture_operation(capture.release)
            self.cap = await self._run_capture_operation(
                self._create_capture,
                stream_url,
            )

    @staticmethod
    def _create_capture(stream_url: str) -> cv2.VideoCapture:
        """Create a configured OpenCV capture object."""
        # OpenCV's FFmpeg open/read timeouts are open-only properties. Passing
        # them after VideoCapture() has connected leaves the backend's
        # 30-second defaults active during a source outage.
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
        transport are healthy.  Source timestamps distinguish that case from a
        decoder that repeatedly returns a stale frame.  Some OpenCV/RTSP
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
        """Releases resources like the capture object."""
        cap = self.cap
        self.cap = None
        if cap is not None:
            await self._run_capture_operation(cap.release)
        self._close_capture_executor()

    async def execute_capture(
        self,
    ) -> AsyncGenerator[tuple[np.ndarray, float]]:
        """Captures frames from the stream and yields them with timestamps.

        Yields:
            Tuple[np.ndarray, float]: The captured frame and the timestamp.
        """
        await self.initialise_stream(self.stream_url)
        last_process_time = datetime.datetime.now() - datetime.timedelta(
            seconds=self.capture_interval,
        )
        fail_count = 0
        backoff_seconds = self.reopen_delay

        try:
            while True:
                if self.cap is None:
                    await self.initialise_stream(self.stream_url)

                cap = self.cap
                ret, frame = (
                    await self._run_capture_operation(cap.read)
                    if cap is not None
                    else (False, None)
                )
                if not ret or not self._is_usable_frame(frame):
                    fail_count += 1
                    self._begin_reconnect()
                    logger.info(
                        'Failed to read frame, trying to reinitialise stream. '
                        f"Fail count: {fail_count}, "
                        f"source={_redact_stream_url(self.stream_url)}",
                    )
                    await self.release_resources()
                    await asyncio.sleep(backoff_seconds)
                    backoff_seconds = min(
                        self.max_reopen_delay,
                        max(self.reopen_delay, backoff_seconds * 1.5),
                    )
                    # Switching before reopening avoids briefly creating a
                    # second FFmpeg capture that the generic source replaces.
                    if (
                        fail_count >= 5
                        and not self.successfully_captured
                        and not _is_rtsp_url(self.stream_url)
                    ):
                        logger.info(
                            'Switching to generic frame capture method.',
                        )
                        async for (
                            generic_frame,
                            generic_timestamp,
                        ) in self.capture_generic_frames():
                            yield generic_frame, generic_timestamp
                        return
                    await self.initialise_stream(self.stream_url)
                    continue

                fail_count = 0
                backoff_seconds = self.reopen_delay
                self.successfully_captured = True
                if (
                    self._should_reconnect_after_stalled_source_timestamp()
                    or self._should_reconnect_after_frozen_frame(frame)
                ):
                    logger.info(
                        'Capture watchdog detected a stalled source; '
                        'reconnecting stream. '
                        f"source={_redact_stream_url(self.stream_url)}",
                    )
                    self._begin_reconnect()
                    await self.release_resources()
                    await asyncio.sleep(self.reopen_delay)
                    await self.initialise_stream(self.stream_url)
                    continue
                self._mark_connected()

                current_time = datetime.datetime.now()
                elapsed_time = (
                    current_time - last_process_time
                ).total_seconds()
                if elapsed_time >= self.capture_interval:
                    last_process_time = current_time
                    yield frame, current_time.timestamp()
                await asyncio.sleep(0.01)
        finally:
            await self.release_resources()

    def check_internet_speed(self) -> tuple[float, float]:
        """Checks internet speed using the Speedtest library.

        Returns:
            Tuple[float, float]: Download and upload speeds (Mbps).
        """
        st = speedtest.Speedtest()
        st.get_best_server()
        download_speed = st.download() / 1_000_000  # Turn into Mbps
        upload_speed = st.upload() / 1_000_000
        return download_speed, upload_speed

    def select_quality_based_on_speed(self) -> str | None:
        """Selects stream quality based on internet speed.

        Returns:
            str: The URL of the selected stream quality.

        Raises:
            Exception: If compatible stream quality is not available.
        """
        download_speed, _ = self.check_internet_speed()
        try:
            streams = streamlink.streams(self.stream_url)
            available_qualities = list(streams.keys())
            logger.info(f"Available qualities: {available_qualities}")

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
                    logger.info(f"Selected quality based on speed: {quality}")
                    return selected_stream.url

            raise Exception('No compatible stream quality is available.')
        except Exception as e:
            logger.info(f"Error selecting quality based on speed: {e}")
            return None

    async def capture_generic_frames(
        self,
    ) -> AsyncGenerator[tuple[np.ndarray, float]]:
        """Captures frames from a generic stream.

        Yields:
            Tuple[np.ndarray, float]: The captured frame and the timestamp.
        """
        stream_url = await self._run_capture_operation(
            self.select_quality_based_on_speed,
        )
        if not stream_url:
            logger.info('Failed to get suitable stream quality.')
            self._close_capture_executor()
            return

        await self.initialise_stream(stream_url)
        last_process_time = datetime.datetime.now()
        fail_count = 0
        backoff_seconds = self.reopen_delay

        try:
            while True:
                cap = self.cap
                ret, frame = (
                    await self._run_capture_operation(cap.read)
                    if cap is not None
                    else (False, None)
                )

                if not ret or not self._is_usable_frame(frame):
                    fail_count += 1
                    self._begin_reconnect()
                    logger.info(
                        'Failed to read frame from generic stream. '
                        f"Fail count: {fail_count}, "
                        'source='
                        f"{_redact_stream_url(stream_url or self.stream_url)}",
                    )
                    await asyncio.sleep(backoff_seconds)
                    backoff_seconds = min(
                        self.max_reopen_delay,
                        max(self.reopen_delay, backoff_seconds * 1.5),
                    )

                    if fail_count >= 5 and not self.successfully_captured:
                        logger.info('Reinitialising the generic stream.')
                        await self.release_resources()
                        await asyncio.sleep(5)
                        stream_url = await self._run_capture_operation(
                            self.select_quality_based_on_speed,
                        )

                        if not stream_url:
                            logger.info(
                                'Failed to get suitable stream quality.',
                            )
                            continue

                        await self.initialise_stream(stream_url)
                        fail_count = 0
                    continue

                fail_count = 0
                backoff_seconds = self.reopen_delay
                self.successfully_captured = True
                self._mark_connected()
                current_time = datetime.datetime.now()
                elapsed_time = (
                    current_time - last_process_time
                ).total_seconds()
                if elapsed_time >= self.capture_interval:
                    last_process_time = current_time
                    yield frame, current_time.timestamp()
                await asyncio.sleep(0.01)
        finally:
            await self.release_resources()

    def update_capture_interval(self, new_interval: float) -> None:
        """Updates the capture interval.

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
        logger.info(f"Frame at {timestamp} displayed")
        # Release the frame resources
        del frame
        gc.collect()


if __name__ == '__main__':
    asyncio.run(main())
