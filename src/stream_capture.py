import argparse
from typing import Any, Generator, Tuple
import cv2
import streamlink
import gc
import datetime
import time
import speedtest

class StreamCapture:
    '''
    Class to capture frames from a video stream.
    '''
    def __init__(self, stream_url: str):
        '''
        Constructor to initialize the stream URL.

        Args:
            stream_url (str): The URL of the video stream.
            cap (cv2.VideoCapture): The OpenCV VideoCapture object.
        '''
        self.stream_url = stream_url
        self.cap = None

    def initialise_stream(self) -> None:
        '''
        Initialises the video stream from the provided URL.

        Raises:
            Exception: If the stream is not opened correctly.
        '''
        self.cap = cv2.VideoCapture(self.stream_url)
        # Set buffer size to 1 to reduce latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Check if the stream is opened correctly.
        if not self.cap.isOpened():
            time.sleep(5)  # Wait for 5 seconds before retrying.
            self.cap.open(self.stream_url)

    def release_resources(self) -> None:
        '''
        Release resources like capture object and destroy any OpenCV windows. 
        
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
        Capture frames using a generic or a RTSP link. 
        
        Args:
            stream_url (str): The URL of the video stream.

        Raises:
            Exception: If an error occurs while reading frames.
        '''
        self.initialise_stream()
        last_process_time = datetime.datetime.now() - datetime.timedelta(seconds=15)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame, trying to reinitialise stream.")
                self.release_resources()
                self.initialise_stream()
                continue

            current_time = datetime.datetime.now()
            if (current_time - last_process_time).total_seconds() >= 15:
                last_process_time = current_time
                timestamp = current_time.timestamp()
                
                yield frame, timestamp

            # # Skip frames if necessary
            # for _ in range(5):  # Skip 5 frames
            #     self.cap.grab()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.release_resources()

    def check_internet_speed(self) -> tuple:
        '''
        Check the internet speed using the Speedtest library.

        Returns:
            Tuple: A tuple containing the download and upload speeds in Mbps.
        '''
        st = speedtest.Speedtest()
        st.get_best_server()
        download_speed = st.download() / 1_000_000  # Mbps
        upload_speed = st.upload() / 1_000_000  # Mbps
        return download_speed, upload_speed

    def select_quality_based_on_speed(self) -> str:
        '''
        Select the stream quality based on the internet speed.

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

    def capture_youtube_frames(self) -> Generator[Tuple[cv2.Mat, float], None, None]:
        '''
        Capture frames from a YouTube stream.

        Returns:
            Generator: A generator yielding frames and timestamps.

        Raises:
            Exception: If an error occurs while reading frames.
        '''
        stream_url = self.select_quality_based_on_speed()
        if not stream_url:
            print("Unable to obtain a suitable stream quality based on internet speed.")
            return

        try:
            self.cap = cv2.VideoCapture(stream_url)
            # Set buffer size to 1 to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            last_process_time = datetime.datetime.now()
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from YouTube stream.")
                    continue

                current_time = datetime.datetime.now()
                if (current_time - last_process_time).total_seconds() >= 5:
                    last_process_time = current_time
                    timestamp = current_time.timestamp()
                    yield frame, timestamp

                # # Skip frames if necessary to manage frame rate
                # for _ in range(5):  # Skip 5 frames
                #     self.cap.grab()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.release_resources()
        
    def execute_capture(self) -> None:
        '''
        Determine the stream type and return the appropriate capture generator. 

        Returns:
            Generator: A generator yielding frames and timestamps.
        '''
        if "youtube.com" in self.stream_url.lower() or "youtu.be" in self.stream_url.lower():
            return self.capture_youtube_frames()
        else:
            return self.capture_frames()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Capture video stream frames.')
    parser.add_argument('--url', type=str, help='Live stream URL', required=True)
    args = parser.parse_args()

    stream_capture = StreamCapture(args.url)
    for frame, timestamp in stream_capture.execute_capture():
        # Process the frame here
        print(f"Frame at {timestamp} displayed")