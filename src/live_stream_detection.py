import argparse
import cv2
import datetime
from pathlib import Path
from typing import Generator, Tuple, List
import time
import gc
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import requests
import datetime

class LiveStreamDetector:
    """
    A class to perform live stream detection and tracking using YOLOv8 with SAHI.
    
    Attributes:
        stream_url (str): The URL of the live video stream.
        model_path (str): The file path to the YOLOv8 model.
        cap (cv2.VideoCapture): Video capture object for the stream.
    """

    def __init__(self, stream_url: str, api_url: str = 'http://localhost:5000/detect', model_key: str = 'yolov8n', output_filename: str = 'detected_frame.jpg'):
        '''
        Initialise the LiveStreamDetector object with the provided stream URL, model path, and output filename.

        Args:
            stream_url (str): The URL of the live video stream.
            api_url (str): The URL of the SAHI API for object detection.
            model_key (str): The key of the model to use for detection.
            output_filename (str): The filename to save the detected frames.
        '''
        self.stream_url = stream_url
        self.api_url = api_url
        self.model_key = model_key
        self.output_filename = output_filename
        self.initialise_stream()
        self.cap = cv2.VideoCapture(self.stream_url)
        
        # Load the font for text drawing
        self.font = ImageFont.truetype("assets/fonts/NotoSansTC-VariableFont_wght.ttf", 20)

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

        # 字体路径和字体对象
        self.font_path, self.font = self.load_font()
        
        # Define colours for different labels
        self.colors = {
            '安全帽': (0, 255, 0),
            '安全背心': (0, 255, 0),
            '機具': (255, 225, 0),
            '車輛': (255, 255, 0),
            '無安全帽': (255, 0, 0),
            '無安全背心': (255, 0, 0),
            '人員': (255, 165, 0),
        }
        
        # List of labels to exclude from drawing
        self.exclude_labels = ['安全錐', '口罩', '無口罩'] # Safety Cone, Mask, No-Mask

    def load_font(self):
        '''
        Load font for text drawing
        '''
        font_path = "assets/fonts/NotoSansTC-VariableFont_wght.ttf"
        font = ImageFont.truetype(font_path, 20)  # Define font size and font
        return font_path, font

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
        Draws detection boxes on the frame with specific colours based on the label,
        and saves the frame to '../detected_frames/' directory with the provided filename.

        Args:
            frame (cv2.Mat): The original video frame.
            datas (List): The list of detection data.
        """
        # Convert cv2 image to PIL image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Create a draw object
        draw = ImageDraw.Draw(pil_image)

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

            if label not in self.exclude_labels:
                # Get colour for the label
                color = self.colors.get(label, (255, 255, 255))  # Default to white
                # Draw rectangle around the object
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

                # Draw label text
                text = f'{label}'
                text_bbox = draw.textbbox((x1, y1), text, font=self.font)
                text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                
                # Calculate the background rectangle for the text
                text_background = (x1, y1 - text_height - 5, x1 + text_width, y1)
                # Draw the background rectangle
                draw.rectangle(text_background, fill=color)

                # Adjust text position
                # By default, the text is drawn at the top left corner of the bounding box
                text_y = y1 - text_height - 5 / 2 - text_height / 2  # Center the text vertically

                # Draw shadow for the text
                for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:  # Draw shadow
                    draw.text((x1 + dx, text_y + dy), text, fill=(0, 0, 0), font=self.font)

                # Draw text on the frame
                draw.text((x1, text_y), text, fill=(255, 255, 255), font=self.font)

        # Convert PIL image back to cv2
        frame_with_detections = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Save the frame to the output directory
        self.save_frame(frame_with_detections)

        # Clear memory by deleting variables
        del frame_rgb, pil_image, draw, frame_with_detections
        gc.collect()

    def save_frame(self, frame: cv2.Mat) -> None:
        """
        Saves the frame to the 'detected_frames/' directory with the provided filename.

        Args:
            frame (cv2.Mat): The frame to be saved.
        """
        output_dir = Path('detected_frames')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / self.output_filename
        cv2.imwrite(str(output_path), frame)

        # Clear memory by deleting variables
        del output_dir, output_path
        gc.collect()

    def generate_detections(self) -> Generator[Tuple[List, cv2.Mat, float], None, None]:
        """
        Generates detections from the video stream, capturing frames every five seconds.

        Yields:
            A tuple containing detection data, the current frame, and the timestamp for each frame.
        """
        last_process_time = datetime.datetime.now() - datetime.timedelta(seconds=300)  # Ensure the first frame is processed.

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

            # Process frames every 300 seconds
            current_time = datetime.datetime.now()
            if (current_time - last_process_time).total_seconds() >= 300:
                last_process_time = current_time  # 更新最后处理时间
                timestamp = current_time.timestamp()

                # Compile detection data in YOLOv8 format
                frame_encoded = cv2.imencode('.jpg', frame_rgb)[1].tobytes()

                # Compile detection data in YOLOv8 format
                _, frame_encoded = cv2.imencode('.jpg', frame_rgb)
                frame_encoded = frame_encoded.tobytes()

                filename = f"frame_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"
                response = requests.post(
                    self.api_url,
                    files={'image': (filename, frame_encoded, 'image/jpeg')},
                    params={'model': self.model_key}
                )

                detections = response.json()

                # Remove overlapping labels for Hardhat and Safety Vest categories
                detections = self.remove_overlapping_labels(detections)

                yield detections, frame, timestamp

                # Clear memory by running garbage collection
                del frame_encoded, filename, response, detections
                gc.collect()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            gc.collect()

    def remove_overlapping_labels(self, datas: List[List[float]]) -> List[List[float]]:
        """
        Removes overlapping labels for Hardhat and Safety Vest categories.

        Args:
            datas (List[List[float]]): List of detection data.

        Returns:
            List[List[float]]: Updated list of detection data with overlapping labels removed.
        """
        hardhat_indices = [i for i, d in enumerate(datas) if d[5] == 0.0]  # Indices of Hardhat detections
        no_hardhat_indices = [i for i, d in enumerate(datas) if d[5] == 2.0]  # Indices of NO-Hardhat detections
        safety_vest_indices = [i for i, d in enumerate(datas) if d[5] == 7.0]  # Indices of Safety Vest detections
        no_safety_vest_indices = [i for i, d in enumerate(datas) if d[5] == 4.0]  # Indices of NO-Safety Vest detections

        # Find all no_hardhat and no_safety_vest indices to remove
        to_remove = set()
        for hardhat_index in hardhat_indices:
            for no_hardhat_index in no_hardhat_indices:
                if self.overlap_percentage(datas[hardhat_index][:4], datas[no_hardhat_index][:4]) > 0.8:
                    to_remove.add(no_hardhat_index)

        for safety_vest_index in safety_vest_indices:
            for no_safety_vest_index in no_safety_vest_indices:
                if self.overlap_percentage(datas[safety_vest_index][:4], datas[no_safety_vest_index][:4]) > 0.8:
                    to_remove.add(no_safety_vest_index)

        # Remove the detected overlapping labels
        # Sort the indices in descending order to avoid index shifting during removal
        for index in sorted(to_remove, reverse=True):
            datas.pop(index)

        # Clear memory by running garbage collection
        del hardhat_indices, no_hardhat_indices, safety_vest_indices, no_safety_vest_indices, to_remove
        gc.collect()

        return datas

    def overlap_percentage(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculates the percentage of overlap between two bounding boxes.

        Args:
            bbox1 (List[float]): The coordinates of the first bounding box in the format [x1, y1, x2, y2].
            bbox2 (List[float]): The coordinates of the second bounding box in the format [x1, y1, x2, y2].

        Returns:
            float: The percentage of overlap between the two bounding boxes.
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
        bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

        overlap_percentage = intersection_area / float(bbox1_area + bbox2_area - intersection_area)

        # Clear memory by running garbage collection
        del x1, y1, x2, y2, intersection_area, bbox1_area, bbox2_area
        gc.collect()

        return overlap_percentage

    def release_resources(self) -> None:
        """
        Releases resources after detection is complete.
        """
        self.cap.release()
        cv2.destroyAllWindows()
        gc.collect()

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
    parser.add_argument('--api_url', type=str, default='http://localhost:5000/detect', help='API URL for detection')
    parser.add_argument('--model_key', type=str, default='yolov8n', help='Model key for detection')
    parser.add_argument('--output', type=str, default='detected_frame.jpg', help='Output image file name')
    args = parser.parse_args()

    # Initialise the live stream detector with the provided stream URL, API URL, model key, and output filename.
    detector = LiveStreamDetector(args.url, args.api_url, args.model_key, args.output)

    # Run the detection process and continuously output images
    detector.run_detection()

"""example
python live_stream_detection.py --url https://cctv6.kctmc.nat.gov.tw/ea05668e/
"""