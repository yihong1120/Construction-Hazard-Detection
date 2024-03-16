import argparse
import cv2
import datetime
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from typing import Generator, Tuple, List
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import platform

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
            0: '安全帽', # Hardhat
            1: '口罩', # Mask
            2: '無安全帽', # No-Hardhat
            3: '無口罩', # No-Mask
            4: '無安全背心', # No-Safety Vest
            5: '人員', # Person
            6: '安全錐', # Safety Cone
            7: '安全背心', # Safety Vest
            8: '機具', # machinery
            9: '車輛' # vehicle
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
        # Determine system type for font path
        system_name = platform.system()
        if system_name == "Windows": # Windows
            font_path = "C:\\Windows\\Fonts\\msyh.ttc" 
        elif system_name == "Darwin":  # macOS
            font_path = "/System/Library/Fonts/STHeiti Medium.ttc"
        else:  # Linux
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

        # Convert cv2 image to PIL image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.truetype(font_path, 20)  # Define font size and font

        # Define colors for different labels
        colors = {
            '安全帽': (0, 255, 0),  # Hardhat: Green
            '安全背心': (0, 255, 0),  # Safety Vest: Green
            '機具': (255, 225, 0),  # machinery: Yellow
            '車輛': (255, 255, 0),  # vehicle: Yellow
            '無安全帽': (255, 0, 0),  # No-Hardhat: Red
            '無安全背心': (255, 0, 0),  # No-Safety Vest: Red
            '人員': (255, 165, 0),  # Person: Orange
        }

        # List of labels to exclude from drawing
        exclude_labels = ['安全錐', '口罩', '無口罩'] # Safety Cone, Mask, No-Mask

        # Draw detections on the frame
        for data in datas:
            x1, y1, x2, y2, _, label_id = data

            # Check if label_id exists in category_id_to_name
            if label_id in self.category_id_to_name:
                label = self.category_id_to_name[label_id]  # Convert ID to label name
            else:
                # If the label_id does not exist in the mapping, skip this detection
                continue

            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            if label not in exclude_labels:
                # Get color for the label
                color = colors.get(label, (255, 255, 255))  # Default to white
                # Draw rectangle around the object
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

                # Draw label text
                text = f'{label}'
                text_bbox = draw.textbbox((x1, y1), text, font=font)
                text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                text_background = (x1, y1 - text_height - 5, x1 + text_width, y1)

                # Draw background rectangle for text
                draw.rectangle(text_background, fill=color)

                # Draw text
                draw.text((x1, y1 - text_height - 5), text, fill=(255, 255, 255), font=font)

        # Convert PIL image back to cv2
        frame_with_detections = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Save the frame to the output directory
        output_dir = Path('demo_data')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / self.output_filename
        cv2.imwrite(str(output_path), frame_with_detections)

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