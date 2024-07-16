from __future__ import annotations

import argparse
import datetime
from collections.abc import Generator
from typing import TypedDict

import cv2
from ultralytics import YOLO


class DetectionResult(TypedDict):
    ids: list[int]
    data: list[list[float]]
    frame: cv2.Mat
    timestamp: float


class LiveStreamDetector:
    """
    A class to perform live stream detection and tracking using YOLOv8.
    """

    def __init__(
        self,
        stream_url: str,
        model_path: str = '../models/pt/best_yolov8n.pt',
    ):
        """
        Initialise live stream detector with video URL, YOLO model path.

        Args:
            stream_url (str): The full URL to the live video stream.
            model_path (str): The path to the YOLOv8 model file.
        """
        self.stream_url = stream_url
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        self.cap = cv2.VideoCapture(self.stream_url)

    def generate_detections(
        self,
    ) -> Generator[tuple[list[int], list[list[float]], cv2.Mat, float]]:
        """
        Yields detection results, timestamp per frame from video capture.

        Yields:
            Generator[Tuple]: Tuple of detection ids, detection data, frame,
            and the current timestamp for each video frame.
        """
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break  # Exit if there is an issue with video capture

            # Get the current timestamp
            now = datetime.datetime.now()
            timestamp = now.timestamp()  # Unix timestamp

            # Run YOLOv8 tracking, maintain tracks across frames
            results = self.model.track(source=frame, persist=True)

            # If detections exist, extract IDs and data
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                # Check emptiness, not converting to list
                ids = results[0].boxes.id
                datas = results[0].boxes.data  # Same as above

                # Convert ids and datas to lists if they are not empty
                ids_list = (
                    ids.cpu().numpy().tolist()
                    if ids is not None and len(ids) > 0
                    else []
                )
                datas_list = (
                    datas.cpu().numpy().tolist()
                    if datas is not None and len(datas) > 0
                    else []
                )

                # Yield the results
                yield ids_list, datas_list, frame, timestamp
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
            print(
                'Timestamp:', datetime.datetime.fromtimestamp(
                    timestamp, tz=datetime.timezone.utc
                ).strftime('%Y-%m-%d %H:%M:%S'),
            )
            print('IDs:', ids)
            print('Data (xyxy format):')
            print(datas)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Perform live stream detection and tracking using YOLOv8.',
    )
    parser.add_argument(
        '--url',
        type=str,
        help='Live stream URL',
        required=True,
    )
    parser.add_argument(
        '--model',
        type=str,
        default='../models/yolov8n.pt',
        help='Path to the YOLOv8 model',
    )
    args = parser.parse_args()

    # Initialize the detector with the stream URL and model path
    detector = LiveStreamDetector(args.url, args.model)

    # Run the detection and print the results
    detector.run_detection()

    # Release resources after detection is complete
    detector.release_resources()

"""example
python live_stream_tracker.py --url https://cctv6.kctmc.nat.gov.tw/ea05668e/
"""
