# filename: live_stream_detector.py
import argparse
import cv2
import datetime
import youtube_dl
from ultralytics import YOLO
from typing import Generator, Tuple

class LiveStreamDetector:
    """
    A class to perform live stream detection and tracking using YOLOv8.
    """

    def __init__(self, youtube_url: str, model_path: str = '../models/yolov8n.pt'):
        """
        Initialises the live stream detector with a YouTube URL and a path to a YOLO model.

        Args:
            youtube_url (str): The full URL to the YouTube video.
            model_path (str): The path to the YOLOv8 model file.
        """
        self.youtube_url = youtube_url
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        self.live_stream_url = self.get_live_video_url(self.youtube_url)
        self.cap = cv2.VideoCapture(self.live_stream_url)

    @staticmethod
    def get_live_video_url(youtube_url: str) -> str:
        """
        Retrieves the live video stream URL from a YouTube video.

        Args:
            youtube_url (str): The full URL to the YouTube video.

        Returns:
            str: The live stream URL of the YouTube video.
        """
        ydl_opts = {
            'format': 'bestaudio/best',  # Choose the best quality
            'quiet': True,  # Suppress output
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=False)
            video_url = info_dict.get("url")
            return video_url

    def generate_detections(self) -> Generator[Tuple, None, None]:
        """
        Yields detection results and the current timestamp from a video capture object frame by frame.

        Yields:
            Generator[Tuple]: Tuple of detection ids, detection data, the frame, and the current timestamp for each video frame.
        """
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break  # Exit if there is an issue with video capture

            # Get the current timestamp
            now = datetime.datetime.now()
            timestamp = now.timestamp()  # Convert the datetime to a Unix timestamp

            # Run YOLOv8 tracking on the frame, maintaining tracks between frames
            results = self.model.track(frame, persist=True)

            # Yield the detection results, the frame, and the current timestamp
            if results[0].boxes is not None:
                ids = results[0].boxes.id.numpy().tolist()  # Convert tensor to list
                datas = results[0].boxes.data.numpy().tolist()  # Convert tensor to list
                yield ids, datas, frame, timestamp
            else:
                yield [], [], frame, timestamp

            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def release_resources(self):
        """
        Releases the video capture object and closes all OpenCV windows.
        """
        self.cap.release()
        cv2.destroyAllWindows()

    def run_detection(self):
        """
        Runs the live stream detection and prints out detection results.
        """
        for ids, datas, frame, timestamp in self.generate_detections():
            print("Timestamp:", datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'))
            print("IDs:", ids)
            print("Data (xyxy format):")
            print(datas)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform live stream detection and tracking using YOLOv8.')
    parser.add_argument('--url', type=str, help='YouTube video URL', required=True)
    parser.add_argument('--model', type=str, default='../models/yolov8n.pt', help='Path to the YOLOv8 model')
    args = parser.parse_args()

    # Initialise the live stream detector with the provided YouTube URL and model path
    detector = LiveStreamDetector(args.url, args.model)

    # Run the detection and print the results
    detector.run_detection()

    # Release resources after detection is complete
    detector.release_resources()

"""
example:
python live_stream_tracking.py --url https://www.youtube.com/watch?v=ZxL5Hm3mIBk

output:
0: 384x640 1 person, 2 cars, 57.3ms
Speed: 1.7ms preprocess, 57.3ms inference, 0.7ms postprocess per image at shape (1, 3, 384, 640)
Timestamp: 2024-02-20 19:25:13
IDs: tensor([1., 2., 3.])
Data (xyxy format):
tensor([[0.0000e+00, 5.8455e+02, 8.8502e+02, 1.0074e+03, 1.0000e+00, 8.8050e-01, 2.0000e+00],
        [9.4921e+02, 5.7641e+02, 1.0652e+03, 8.1061e+02, 2.0000e+00, 8.4425e-01, 0.0000e+00],
        [7.9515e+02, 6.2592e+02, 9.8356e+02, 7.5257e+02, 3.0000e+00, 7.0361e-01, 2.0000e+00]])

0: 384x640 1 person, 2 cars, 55.3ms
Speed: 1.4ms preprocess, 55.3ms inference, 0.5ms postprocess per image at shape (1, 3, 384, 640)
Timestamp: 2024-02-20 19:25:13
IDs: tensor([1., 2., 3.])
Data (xyxy format):
tensor([[0.0000e+00, 5.8485e+02, 8.8459e+02, 1.0119e+03, 1.0000e+00, 8.7216e-01, 2.0000e+00],
        [9.4962e+02, 5.7691e+02, 1.0664e+03, 8.1135e+02, 2.0000e+00, 8.4799e-01, 0.0000e+00],
        [7.9445e+02, 6.2432e+02, 9.8249e+02, 7.5349e+02, 3.0000e+00, 6.3214e-01, 2.0000e+00]])
"""