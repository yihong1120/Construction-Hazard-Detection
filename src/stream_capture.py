from __future__ import annotations

import argparse
import datetime
import gc
import time
from collections.abc import Generator
from typing import TypedDict, Optional, Tuple

import cv2
import speedtest
import streamlink

class FrameData(TypedDict):
    frame: cv2.Mat
    timestamp: float

class SpeedData(TypedDict):
    download_speed: float
    upload_speed: float

class StreamCapture:
    """
    A class to capture frames from a video stream.
    """

    def __init__(self, stream_url: str, capture_interval: int = 15):
        """
        Initialises the StreamCapture with the given stream URL.

        Args:
            stream_url (str): The URL of the video stream.
        """
        self.stream_url = stream_url
        self.cap: Optional[cv2.VideoCapture] = None
        self.capture_interval = capture_interval

    def initialise_stream(self) -> None:
        """
        Initialises the video stream and sets up the H264 codec.
        """
        self.cap = cv2.VideoCapture(self.stream_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
        if not self.cap.isOpened():
            time.sleep(5)
            self.cap.open(self.stream_url)

    def release_resources(self) -> None:
        """
        Releases resources like the capture object.
        """
        if self.cap:
            self.cap.release()
            self.cap = None
        gc.collect()

    def capture_frames(self) -> Generator[FrameData, None, None]:
        """
        Captures frames from the stream and yields them with timestamps.

        Yields:
            FrameData: The captured frame and the timestamp.
        """
        self.initialise_stream()
        last_process_time = datetime.datetime.now() - datetime.timedelta(
            seconds=self.capture_interval,
        )
        while True:
            if self.cap is None:
                self.initialise_stream()

            ret, frame = self.cap.read() if self.cap else (False, None)

            if not ret:
                print('Failed to read frame, trying to reinitialise stream.')
                self.release_resources()
                self.initialise_stream()
                continue

            # Process the frame if the capture interval has elapsed
            current_time = datetime.datetime.now()
            elapsed_time = (current_time - last_process_time).total_seconds()

            # If the capture interval has elapsed, yield the frame
            if elapsed_time >= self.capture_interval:
                last_process_time = current_time
                timestamp = current_time.timestamp()
                yield {'frame': frame, 'timestamp': timestamp}

                # Clear memory
                del frame, timestamp
                gc.collect()

            time.sleep(0.01)  # Adjust the sleep time as needed

        self.release_resources()

    def check_internet_speed(self) -> SpeedData:
        """
        Checks internet speed using the Speedtest library.

        Returns:
            SpeedData: The download and upload speeds in Mbps.
        """
        st = speedtest.Speedtest()
        st.get_best_server()
        download_speed = st.download() / 1_000_000
        upload_speed = st.upload() / 1_000_000
        return {'download_speed': download_speed, 'upload_speed': upload_speed}

    def select_quality_based_on_speed(self) -> Optional[str]:
        """
        Selects stream quality based on internet speed.

        Returns:
            Optional[str]: The URL of the selected stream quality.

        Raises:
            Exception: If no compatible stream quality is available.
        """
        speed_data = self.check_internet_speed()
        download_speed = speed_data['download_speed']
        try:
            streams = streamlink.streams(self.stream_url)
            available_qualities = list(streams.keys())
            print('Available qualities:', available_qualities)

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

    def capture_youtube_frames(
        self,
    ) -> Generator[FrameData, None, None]:
        """
        Captures frames from a YouTube stream.

        Yields:
            FrameData: The captured frame and the timestamp.
        """
        stream_url = self.select_quality_based_on_speed()
        if not stream_url:
            print('Failed to get suitable stream quality.')
            return

        try:
            self.cap = cv2.VideoCapture(stream_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
            last_process_time = datetime.datetime.now()
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print('Failed to read frame from YouTube stream.')
                    continue

                current_time = datetime.datetime.now()
                if (
                    current_time - last_process_time
                ).total_seconds() >= self.capture_interval:
                    last_process_time = current_time
                    timestamp = current_time.timestamp()
                    yield {'frame': frame, 'timestamp': timestamp}

                    # 清理內存
                    del frame, timestamp
                    gc.collect()

                time.sleep(0.01)  # Adjust the sleep time as needed
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.release_resources()

    def execute_capture(self) -> Generator[FrameData, None, None]:
        """
        Returns capture generator for stream type.

        Returns:
            Generator[FrameData]: Yields frames and timestamps.
        """
        if (
            'youtube.com' in self.stream_url.lower()
            or 'youtu.be' in self.stream_url.lower()
        ):
            return self.capture_youtube_frames()
        else:
            return self.capture_frames()

    def update_capture_interval(self, new_interval: int) -> None:
        """
        Updates the capture interval.

        Args:
            new_interval (int): Frame capture interval in seconds.
        """
        self.capture_interval = new_interval


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Capture video stream frames.',
    )
    parser.add_argument(
        '--url',
        type=str,
        help='Live stream URL',
        required=True,
    )
    args = parser.parse_args()

    stream_capture = StreamCapture(args.url)
    for frame_data in stream_capture.execute_capture():
        # Process the frame here
        print(f"Frame at {frame_data['timestamp']} displayed")
        # Release the frame resources
        del frame_data['frame']
        gc.collect()
