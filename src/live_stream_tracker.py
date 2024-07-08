from __future__ import annotations

import argparse
import datetime
from collections.abc import Generator
from typing import TypedDict, List, Tuple

import cv2
from ultralytics import YOLO

class BoundingBox(TypedDict):
    x1: float
    y1: float
    x2: float
    y2: float

class DetectionData(TypedDict):
    bbox: BoundingBox
    confidence: float
    class_label: int

class DetectionResult(TypedDict):
    ids: List[int]
    datas: List[DetectionData]
    frame: cv2.Mat
    timestamp: float

class LiveStreamDetector:
    """
    A class to perform live stream detection and tracking using YOLOv8.
    """

    def __init__(
        self,
        stream_url: str,
        model_path: str = '../models/yolov8n.pt',
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

    def generate_detections(self) -> Generator[DetectionResult, None, None]:
        """
        Yields detection results, timestamp per frame from video capture.

        Yields:
            Generator[DetectionResult]: Detection result including detection ids,
            detection data, frame, and the current timestamp for each video frame.
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
                    ids.numpy().tolist()
                    if ids is not None and len(ids) > 0
                    else []
                )
                datas_list = (
                    datas.numpy().tolist()
                    if datas is not None and len(datas) > 0
                    else []
                )

                # Convert datas_list to DetectionData format
                detection_datas = [
                    {
                        'bbox': {
                            'x1': data[0],
                            'y1': data[1],
                            'x2': data[2],
                            'y2': data[3],
                        },
                        'confidence': data[4],
                        'class_label': int(data[5]),
                    }
                    for data in datas_list
                ]

                # Yield the results
                yield {
                    'ids': ids_list,
                    'datas': detection_datas,
                    'frame': frame,
                    'timestamp': timestamp,
                }
            else:
                yield {
                    'ids': [],
                    'datas': [],
                    'frame': frame,
                    'timestamp': timestamp,
                }

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
        for result in self.generate_detections():
            print(
                'Timestamp:', datetime.datetime.fromtimestamp(
                    result['timestamp'],
                ).strftime('%Y-%m-%d %H:%M:%S'),
            )
            print('IDs:', result['ids'])
            print('Data (xyxy format):')
            for data in result['datas']:
                print(data)

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

# Example usage
# python live_stream_tracker.py --url https://cctv6.kctmc.nat.gov.tw/ea05668e/
