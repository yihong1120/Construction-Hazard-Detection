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
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import datetime
import os
from dotenv import load_dotenv
load_dotenv()

class LiveStreamDetector:
    """
    A class to perform live stream detection and tracking using YOLOv8 with SAHI.
    
    Attributes:
        stream_url (str): The URL of the live video stream.
        model_path (str): The file path to the YOLOv8 model.
        cap (cv2.VideoCapture): Video capture object for the stream.
    """

    def __init__(self, stream_url: str, api_url: str = 'http://localhost:5000', model_key: str = 'yolov8l', output_filename: str = 'detected_frame.jpg'):
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
        self.session = self.requests_retry_session()
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

        self.access_token = None
        self.authenticate()  # 在初始化时进行认证

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

    def requests_retry_session(
        self,
        retries=5,
        backoff_factor=0.5,
        status_forcelist=(500, 502, 504, 104),
        session=None,
        allowed_methods=frozenset(['HEAD', 'GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'TRACE'])
    ):
        session = session or requests.Session()
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            method_whitelist=method_whitelist
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def authenticate(self):
        """Authenticate with the API server to obtain JWT."""
        try:
            response = self.session.post(
                f"{self.api_url}/token",
                json={"username": os.getenv('API_USERNAME'), "password": os.getenv('API_PASSWORD')}
            )
            response.raise_for_status()  # 将抛出异常，如果响应码不是 200
            self.access_token = response.json()['access_token']
        except (ConnectionError, requests.RequestException) as e:
            print(f"网络连接异常或请求错误: {e}")
            raise

    def generate_detections(self) -> Generator[Tuple[List, cv2.Mat, float], None, None]:
        """
        Generates detections from the video stream, capturing frames every five seconds.

        Yields:
            A tuple containing detection data, the current frame, and the timestamp for each frame.
        """
        last_process_time = datetime.datetime.now() - datetime.timedelta(seconds=5)  # 确保第一帧被处理
        while True:
            if not self.cap.isOpened():
                self.initialise_stream()
            success, frame = self.cap.read()
            if not success:
                self.release_resources()
                print("Failed to read frame, trying to reinitialise stream.")
                self.initialise_stream()
                continue

            current_time = datetime.datetime.now()
            if (current_time - last_process_time).total_seconds() >= 5:
                last_process_time = current_time
                timestamp = current_time.timestamp()

                _, frame_encoded = cv2.imencode('.png', frame)
                frame_encoded = frame_encoded.tobytes()
                filename = f"frame_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.png"

                try:
                    headers = {'Authorization': f'Bearer {self.access_token}'}
                    response = requests.post(
                        f'{self.api_url}/detect',
                        files={'image': (filename, frame_encoded, 'image/png')},
                        params={'model': self.model_key},
                        headers=headers
                    )
                    response.raise_for_status()
                    detections = response.json()
                    yield detections, frame, timestamp
                except requests.exceptions.RequestException as e:
                    print(f"Request failed: {e}, re-authenticating and retrying...")
                    self.authenticate()
                    try:
                        response = requests.post(
                            f'{self.api_url}/detect',
                            files={'image': (filename, frame_encoded, 'image/png')},
                            params={'model': self.model_key},
                            headers=headers
                        )
                        response.raise_for_status()
                        detections = response.json()
                        yield detections, frame, timestamp
                    except requests.exceptions.RequestException as e:
                        print(f"Failed after retry: {e}")
                        continue

                del frame_encoded, filename, response, detections
                gc.collect()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            gc.collect()

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
        except requests.RequestException as e:
            print(f"处理请求时发生错误: {e}")
        except Exception as e:
            print(f"发生未预期的错误: {e}")
        finally:
            self.release_resources()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform live stream detection and tracking using YOLOv8.')
    parser.add_argument('--url', type=str, help='Live stream URL', required=True)
    parser.add_argument('--api_url', type=str, default='http://localhost:5000', help='API URL for detection')
    parser.add_argument('--model_key', type=str, default='yolov8n', help='Model key for detection')
    parser.add_argument('--output', type=str, default='detected_frame.jpg', help='Output image file name')
    args = parser.parse_args()

    # Initialise the live stream detector with the provided stream URL, API URL, model key, and output filename.
    detector = LiveStreamDetector(args.url, args.api_url, args.model_key, args.output)

    # Run the detection process and continuously output images
    detector.authenticate()
    detector.run_detection()

"""example
python live_stream_detection.py --url https://cctv6.kctmc.nat.gov.tw/ea05668e/
"""