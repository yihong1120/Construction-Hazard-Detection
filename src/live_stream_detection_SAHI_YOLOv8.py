import argparse
import cv2
import datetime
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from typing import Generator, Tuple

class LiveStreamDetector:
    """
    A class to perform live stream detection and tracking using YOLOv8 with SAHI.
    """

    def __init__(self, stream_url: str, model_path: str = '../models/yolov8n.pt'):
        """
        Initialises the live stream detector with a video stream URL and a path to a YOLO model.
        
        Args:
            stream_url (str): The full URL to the live video stream.
            model_path (str): The path to the YOLOv8 model file.
        """
        self.stream_url = stream_url
        self.sahi_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=model_path,
            confidence_threshold=0.3,
            # device="cpu", or 'cuda:0'
        )
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

            now = datetime.datetime.now()
            timestamp = now.timestamp()

            # Perform detection using SAHI
            result = get_sliced_prediction(
                frame,
                self.sahi_model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2
            )

            # Process object predictions in YOLOv8 format
            datas = []
            for object_prediction in result.object_prediction_list:
                label = object_prediction.category.id  # Use category ID for YOLOv8 format
                x1, y1, x2, y2 = object_prediction.bbox.to_voc_bbox()  # Convert to xyxy format
                confidence = object_prediction.score.value
                datas.append([x1, y1, x2, y2, confidence, label])  # YOLOv8 format

            yield datas, frame, timestamp

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def release_resources(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def run_detection(self):
        for datas, frame, timestamp in self.generate_detections():
            print("Timestamp:", datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'))
            print("datas:", datas)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform live stream detection and tracking using YOLOv8.')
    parser.add_argument('--url', type=str, help='Live stream URL', required=True)
    parser.add_argument('--sahi_model', type=str, default='../models/yolov8n.pt', help='Path to the YOLOv8 model')
    args = parser.parse_args()

    # Initialise the live stream detector with the provided stream URL and model path
    detector = LiveStreamDetector(args.url, args.sahi_model)

    # Run the detection and print the results
    detector.run_detection()

    # Release resources after detection is complete
    detector.release_resources()

"""example
python live_stream_detection(SAHI+YOLOv8).py --url https://cctv6.kctmc.nat.gov.tw/ea05668e/
"""