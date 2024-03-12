import argparse
import cv2
import datetime
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from typing import Generator, Tuple, List
import time
from pathlib import Path

class LiveStreamDetector:
    """
    A class to perform live stream detection and tracking using YOLOv8 with SAHI.
    
    Attributes:
        stream_url (str): The URL of the live video stream.
        model_path (str): The file path to the YOLOv8 model.
        cap (cv2.VideoCapture): Video capture object for the stream.
        sahi_model (AutoDetectionModel): The SAHI model for object detection.
    """

    def __init__(self, stream_url: str, model_path: str = '../models/best_yolov8n.pt', output_filename: str = 'detected_frame.jpg'):
        """
        Initialises the live stream detector with a video stream URL, a path to a YOLO model, and an output filename.

        Args:
            stream_url (str): The full URL to the live video stream.
            model_path (str): The path to the YOLOv8 model file.
            output_filename (str): The name of the output image file.
        """
        self.stream_url = stream_url
        self.model_path = model_path
        self.output_filename = output_filename
        self.initialise_stream()
        self.sahi_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=model_path,
            confidence_threshold=0.3,
            # Set device to "cpu" or "cuda:0" based on your setup.
        )

        # Add a category ID to name mapping based on your model's training labels
        self.category_id_to_name = {
            0: 'Hardhat',
            1: 'Mask',
            2: 'NO-Hardhat',
            3: 'NO-Mask',
            4: 'NO-Safety Vest',
            5: 'Person',
            6: 'Safety Cone',
            7: 'Safety Vest',
            8: 'Machinery',
            9: 'Vehicle'
        }

    def initialise_stream(self) -> None:
        """
        Initialises the video stream from the provided URL.
        """
        self.cap = cv2.VideoCapture(self.stream_url)
        # Check if the stream is opened correctly.
        if not self.cap.isOpened():
            time.sleep(5)  # Wait for 5 seconds before retrying.
            self.cap.open(self.stream_url)

    def draw_detections_on_frame(self, frame: cv2.Mat, datas: List) -> None:
        """
        Draws detection boxes on the frame with specific colors based on the label,
        and saves the frame to 'demo_data/' directory with the provided filename.

        Args:
            frame (cv2.Mat): The original video frame.
            datas (List): The list of detection data.
            output_filename (str): The name of the output image file.
        """
        # Define colors for different labels
        colors = {
            'Hardhat': (0, 255, 0),  # Green
            'Safety Vest': (0, 255, 0),  # Green
            'machinery': (0, 255, 255),  # Yellow
            'vehicle': (0, 255, 255),  # Yellow
            'NO-Hardhat': (0, 0, 255),  # Red
            'NO-Safety Vest': (0, 0, 255),  # Red
            'Person': (255, 165, 0),  # Orange
            # Add more colors if necessary
        }

        # List of labels to exclude from drawing
        exclude_labels = ['Safety Cone', 'Mask', 'NO-Mask']

        # Draw detections on the frame
        for data in datas:
            x1, y1, x2, y2, confidence, label_id = data

            # Check if label_id exists in category_id_to_name
            if label_id in self.category_id_to_name:
                label = self.category_id_to_name[label_id]  # Convert ID to label name
            else:
                # If the label_id does not exist in the mapping, skip this detection
                continue

            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            if label not in exclude_labels:
                color = colors.get(label, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Text settings
                font_scale = 0.5  # Smaller font size
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = f'{label}: {confidence:.2f}'
                text_size = cv2.getTextSize(text, font, font_scale, 1)[0]

                # Background rectangle for text
                rect_start = (x1, y1 - 5 - text_size[1])
                rect_end = (x1 + text_size[0], y1)
                cv2.rectangle(frame, rect_start, rect_end, color, -1)  # Solid fill

                # White text
                cv2.putText(frame, text, (x1, y1 - 5), font, font_scale, (255, 255, 255), 1)

        # Save the frame to the output directory
        output_dir = Path('demo_data')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / self.output_filename
        cv2.imwrite(str(output_path), frame)

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
        Runs the detection process and draws the results on each frame continuously.
        """
        try:
            for datas, frame, timestamp in self.generate_detections():
                print("Timestamp:", datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'))
                print("Datas:", datas)
                self.draw_detections_on_frame(frame, datas)
                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.release_resources()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform live stream detection and tracking using YOLOv8.')
    parser.add_argument('--url', type=str, help='Live stream URL', required=True)
    parser.add_argument('--sahi_model', type=str, default='../models/best_yolov8x.pt', help='Path to the YOLOv8 model')
    parser.add_argument('--output', type=str, default='detected_frame.jpg', help='Output image file name')
    args = parser.parse_args()

    # Initialise the live stream detector with the provided stream URL, model path, and output filename.
    detector = LiveStreamDetector(args.url, args.sahi_model, args.output)

    # Run the detection process and continuously output images
    detector.run_detection()

"""example
python live_stream_detection.py --url https://cctv6.kctmc.nat.gov.tw/ea05668e/
"""