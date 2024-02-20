import argparse
import cv2
import datetime
from ultralytics import YOLO
from typing import Generator, Tuple

class LiveStreamDetector:
    """
    A class to perform live stream detection and tracking using YOLOv8.
    """

    def __init__(self, stream_url: str, model_path: str = '../models/yolov8n.pt'):
        """
        Initialises the live stream detector with a video stream URL and a path to a YOLO model.

        Args:
            stream_url (str): The full URL to the live video stream.
            model_path (str): The path to the YOLOv8 model file.
        """
        self.stream_url = stream_url
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        self.cap = cv2.VideoCapture(self.stream_url)

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

            # Check if there are any detections and if so, extract their IDs and data
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                ids = results[0].boxes.id  # No longer converting to list here, check for emptiness instead
                datas = results[0].boxes.data  # Same as above

                # Convert ids and datas to lists if they are not empty
                ids_list = ids.numpy().tolist() if ids is not None and len(ids) > 0 else []
                datas_list = datas.numpy().tolist() if datas is not None and len(datas) > 0 else []

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
            print("Timestamp:", datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'))
            print("IDs:", ids)
            print("Data (xyxy format):")
            print(datas)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform live stream detection and tracking using YOLOv8.')
    parser.add_argument('--url', type=str, help='Live stream URL', required=True)
    parser.add_argument('--model', type=str, default='../models/yolov8n.pt', help='Path to the YOLOv8 model')
    args = parser.parse_args()

    # Initialise the live stream detector with the provided stream URL and model path
    detector = LiveStreamDetector(args.url, args.model)

    # Run the detection and print the results
    detector.run_detection()

    # Release resources after detection is complete
    detector.release_resources()

"""example
python live_stream_tracker.py --url https://cctv6.kctmc.nat.gov.tw/ea05668e/
"""