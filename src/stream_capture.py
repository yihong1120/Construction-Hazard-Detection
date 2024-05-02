import argparse
from typing import Generator, Tuple
import cv2
import streamlink
import gc
import datetime
import time
import speedtest

class StreamCapture:
    '''
    Class designed to capture frames from a video stream.
    '''
    def __init__(self, stream_url: str):
        '''
        Constructor to initialise the stream URL.

        Args:
            stream_url (str): The URL of the video stream.
            cap (cv2.VideoCapture): The OpenCV VideoCapture object.
        '''
        self.stream_url = stream_url
        self.cap = None

    def initialise_stream(self, stream_url: str) -> None:
        '''
        Initialises the video stream from the provided URL.

        Args:
            stream_url (str): The URL of the video stream.

        Raises:
            Exception: If an error occurs while initialising the stream.
        '''
        self.cap = cv2.VideoCapture(stream_url)
        # Set buffer size to 1 to reduce latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Check if the stream has opened correctly.
        if not self.cap.isOpened():
            time.sleep(5)  # Wait for 5 seconds before retrying.
            self.cap.open(stream_url)

    def release_resources(self) -> None:
        '''
        Release resources such as the capture object and destroy any OpenCV windows.
        
        Args:
            cap (cv2.VideoCapture): The OpenCV VideoCapture object.

        Raises:
            Exception: If an error occurs while releasing resources.
        '''
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        gc.collect()

    def capture_frames(self) -> Generator[Tuple[cv2.Mat, float], None, None]:
        '''
        Captures frames from a video stream, accommodating both generic and YouTube streams.

        Args:
            stream_url (str): The URL of the video stream.

        Yields:
            Tuple: A tuple containing the frame and timestamp.

        Raises:
            Exception: If an error occurs while capturing frames.
        '''
        # If the stream originates from YouTube, select the quality based on internet speed.
        if "youtube.com" in self.stream_url.lower() or "youtu.be" in self.stream_url.lower():
            stream_url = self.select_quality_based_on_speed()
            if not stream_url:
                print("Unable to obtain a suitable stream quality based on internet speed.")
                return
        else:
            stream_url = self.stream_url

        self.initialise_stream(stream_url)
        last_process_time = datetime.datetime.now()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame, attempting to reinitialise stream.")
                self.release_resources()
                self.initialise_stream(stream_url)
                continue

            current_time = datetime.datetime.now()
            if (current_time - last_process_time).total_seconds() >= 5:
                last_process_time = current_time
                timestamp = current_time.timestamp()
                yield frame, timestamp

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.release_resources()

    def check_internet_speed(self) -> tuple:
        '''
        Checks the internet speed using the Speedtest library.

        Returns:
            Tuple: A tuple containing the download and upload speeds in Mbps.

        Raises:
            Exception: If an error occurs while checking the internet speed.
        '''
        st = speedtest.Speedtest()
        st.get_best_server()
        download_speed = st.download() / 1_000_000  # Mbps
        upload_speed = st.upload() / 1_000_000  # Mbps
        return download_speed, upload_speed

    def select_quality_based_on_speed(self) -> str:
        '''
        Selects the stream quality based on the internet speed.

        Returns:
            str: The URL of the selected stream quality.

        Raises:
            Exception: If no compatible stream quality is available.
        '''
        download_speed, _ = self.check_internet_speed()
        try:
            streams = streamlink.streams(self.stream_url)
            available_qualities = list(streams.keys())
            print("Available qualities:", available_qualities)
            
            # Define the preferred qualities based on the download speed.
            if download_speed > 10:  # 10 Mbps or more
                preferred_qualities = ['best', '1080p', '720p', '480p', '360p', '240p', 'worst']
            elif 5 < download_speed <= 10:  # 5-10 Mbps
                preferred_qualities = ['720p', '480p', '360p', '240p', 'worst']
            else:  # 5 Mbps or less
                preferred_qualities = ['480p', '360p', '240p', 'worst']
            
            # Select the first available quality from the preferred qualities list.
            for quality in preferred_qualities:
                if quality in available_qualities:
                    selected_stream = streams[quality]
                    print(f"Selected quality based on speed: {quality}")
                    return selected_stream.url

            raise Exception("No compatible stream quality is available.")
        except Exception as e:
            print(f"Error selecting quality based on speed: {e}")
            return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Capture video stream frames.')
    parser.add_argument('--url', type=str, help='Live stream URL', required=True)
    args = parser.parse_args()

    stream_capture = StreamCapture(args.url)
    for frame, timestamp in stream_capture.capture_frames():
        # Process the frame here
        print(f"Frame at {timestamp} displayed")