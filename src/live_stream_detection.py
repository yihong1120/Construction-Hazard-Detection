import argparse
import cv2
import datetime
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from typing import Generator, Tuple, List
import time

class LiveStreamDetector:
    """
    A class to perform live stream detection and tracking using YOLOv8 with SAHI.
    
    Attributes:
        stream_url (str): The URL of the live video stream.
        model_path (str): The file path to the YOLOv8 model.
        cap (cv2.VideoCapture): Video capture object for the stream.
        sahi_model (AutoDetectionModel): The SAHI model for object detection.
    """

    def __init__(self, stream_url: str, model_path: str = '../models/yolov8n.pt'):
        """
        Initialises the live stream detector with a video stream URL and a path to a YOLO model.

        Args:
            stream_url (str): The full URL to the live video stream.
            model_path (str): The path to the YOLOv8 model file.
        """
        self.stream_url: str = stream_url
        self.model_path: str = model_path
        self.initialise_stream()
        self.sahi_model: AutoDetectionModel = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=model_path,
            confidence_threshold=0.3,
            # Set device to "cpu" or "cuda:0" based on your setup.
        )

    def initialise_stream(self) -> None:
        """
        Initialises the video stream from the provided URL.
        """
        self.cap = cv2.VideoCapture(self.stream_url)
        # Check if the stream is opened correctly.
        if not self.cap.isOpened():
            time.sleep(5)  # Wait for 5 seconds before retrying.
            self.cap.open(self.stream_url)

    def generate_detections(self) -> Generator[Tuple[List, cv2.Mat, float], None, None]:
        """
        Generates detections from the video stream, capturing frames every five seconds.

        Yields:
            A tuple containing detection data, the current frame, and the timestamp for each frame.
        """
        last_process_time = datetime.datetime.now() - datetime.timedelta(seconds=5)  # Ensure the first frame is processed.

        while True:
            if not self.cap.isOpened():
                self.initialise_stream()
            success, frame = self.cap.read()
            if not success:
                self.release_resources()
                print("Failed to read frame, trying to reinitialise stream.")
                self.initialise_stream()
                continue

            # Convert frame to RGB as SAHI expects RGB images
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frames every five seconds
            current_time = datetime.datetime.now()
            if (current_time - last_process_time).total_seconds() >= 5:
                last_process_time = current_time  # 更新最后处理时间
                timestamp = current_time.timestamp()

                # Generate predictions for the current frame
                result = get_sliced_prediction(
                    frame_rgb,
                    self.sahi_model,
                    slice_height=384,
                    slice_width=384,
                    overlap_height_ratio=0.3,
                    overlap_width_ratio=0.3
                )

                # Export visuals for debugging and interpretation
                result.export_visuals(export_dir="demo_data/")

                # Compile detection data in YOLOv8 format
                datas = []
                for object_prediction in result.object_prediction_list:
                    label = object_prediction.category.id
                    x1, y1, x2, y2 = object_prediction.bbox.to_voc_bbox()
                    confidence = object_prediction.score.value
                    datas.append([x1, y1, x2, y2, confidence, label])

                yield datas, frame, timestamp

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def release_resources(self) -> None:
        """
        Releases resources after detection is complete.
        """
        self.cap.release()
        cv2.destroyAllWindows()

    def run_detection(self) -> None:
        """
        Runs the detection process and prints the results.
        """
        for datas, frame, timestamp in self.generate_detections():
            print("Timestamp:", datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'))
            print("Datas:", datas)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform live stream detection and tracking using YOLOv8.')
    parser.add_argument('--url', type=str, help='Live stream URL', required=True)
    parser.add_argument('--sahi_model', type=str, default='../models/yolov8n.pt', help='Path to the YOLOv8 model')
    args = parser.parse_args()

    # Initialise the live stream detector with the provided stream URL and model path.
    detector = LiveStreamDetector(args.url, args.sahi_model)

    # Run the detection and print the results.
    detector.run_detection()

    # Release resources after detection is complete.
    detector.release_resources()

"""example
python live_stream_detection_SAHI_YOLOv8.py --url https://cctv6.kctmc.nat.gov.tw/ea05668e/
"""