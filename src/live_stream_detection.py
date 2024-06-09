import argparse
import cv2
import datetime
from pathlib import Path
from typing import Tuple, List, Optional
import gc
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time
import os
from dotenv import load_dotenv
load_dotenv()

class LiveStreamDetector:
    """
    A class to perform live stream detection and tracking using YOLOv8 with SAHI.
    """

    def __init__(self, api_url: str = 'http://localhost:5000', model_key: str = 'yolov8l',
                 output_folder: Optional[str] = None):
        """
        Initialises the LiveStreamDetector.

        Args:
            api_url (str): The URL of the API for detection.
            model_key (str): The model key for detection.
            output_folder (Optional[str]): The output folder for detected frames.
            output_filename (str): The output image file name.
        """
        self.api_url = api_url
        self.model_key = model_key
        self.output_folder = output_folder
        self.session = self.requests_retry_session()
        
        # Load the font used for drawing labels on the image
        self.font = ImageFont.truetype("assets/fonts/NotoSansTC-VariableFont_wght.ttf", 20)

        # Mapping of category IDs to their corresponding names
        self.category_id_to_name = {
            0: '安全帽',
            1: '口罩',
            2: '無安全帽',
            3: '無口罩',
            4: '無安全背心',
            5: '人員',
            6: '安全錐',
            7: '安全背心',
            8: '機具',
            9: '車輛'
        }

        # Define colours for each category
        self.colors = {
            '安全帽': (0, 255, 0),
            '安全背心': (0, 255, 0),
            '機具': (255, 225, 0),
            '車輛': (255, 255, 0),
            '無安全帽': (255, 0, 0),
            '無安全背心': (255, 0, 0),
            '人員': (255, 165, 0),
        }

        # Generate exclude_labels automatically
        self.exclude_labels = [
            label for label in self.category_id_to_name.values() if label not in self.colors
        ]
        
        # Authenticate and get access token
        self.access_token = None
        self.authenticate()

    def draw_detections_on_frame(self, frame: cv2.Mat, datas: List[List[float]]) -> None:
        """
        Draws detections on the given frame.

        Args:
            frame (cv2.Mat): The frame on which to draw detections.
            datas (List[List[float]]): The detection data.

        Returns:
            frame_with_detections(cv2.Mat): The frame with detections drawn on it.
        """
        # Convert the frame to RGB and create a PIL image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)

        for data in datas:
            x1, y1, x2, y2, _, label_id = data
            if label_id in self.category_id_to_name:
                label = self.category_id_to_name[label_id]
            else:
                continue

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            if label not in self.exclude_labels:
                color = self.colors.get(label, (255, 255, 255))
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                text = f'{label}'
                text_bbox = draw.textbbox((x1, y1), text, font=self.font)
                text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                text_background = (x1, y1 - text_height - 5, x1 + text_width, y1)
                draw.rectangle(text_background, fill=color)

                text_y = y1 - text_height - 5 / 2 - text_height / 2
                for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    draw.text((x1 + dx, text_y + dy), text, fill=(0, 0, 0), font=self.font)
                draw.text((x1, text_y), text, fill=(255, 255, 255), font=self.font)

        # Convert the PIL image back to OpenCV format
        frame_with_detections = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return frame_with_detections

    def save_frame(self, frame_bytes: bytes, output_filename: str) -> None:
        """
        Saves the frame with detections to the specified output folder and filename.

        Args:
            frame_bytes (bytes): The byte stream of the frame.
            output_filename (str): The output filename.
        """
        # Create the output directory if it does not exist
        base_output_dir = Path('detected_frames')
        output_dir = base_output_dir / self.output_folder if self.output_folder else base_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define the output path
        output_path = output_dir / f"{output_filename}.png"
        
        # Save the byte stream to the output path
        with open(output_path, 'wb') as f:
            f.write(frame_bytes.tobytes())

        # Clean up
        del output_dir, output_path, frame_bytes
        gc.collect()

    def requests_retry_session(self, retries: int = 7, backoff_factor: int = 1, 
                               status_forcelist: Tuple[int] = (500, 502, 504, 401, 104), 
                               session: Optional[requests.Session] = None, 
                               allowed_methods: frozenset = frozenset(['HEAD', 'GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'TRACE'])) -> requests.Session:
        """
        Configures a requests session with retry logic.

        Args:
            retries (int): The number of retry attempts.
            backoff_factor (int): The backoff factor for retries.
            status_forcelist (Tuple[int]): The list of HTTP status codes to trigger a retry.
            session (Optional[requests.Session]): An optional requests session.
            allowed_methods (frozenset): The set of allowed HTTP methods.

        Returns:
            requests.Session: The configured requests session.
        """
        session = session or requests.Session()
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            allowed_methods=allowed_methods,
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(requests.RequestException))
    def authenticate(self) -> None:
        """
        Authenticates with the API and retrieves the access token.
        """
        response = self.session.post(
            f"{self.api_url}/token",
            json={"username": os.getenv('API_USERNAME'), "password": os.getenv('API_PASSWORD')}
        )
        response.raise_for_status()
        token_data = response.json()
        if 'msg' in token_data:
            raise Exception(token_data['msg'])
        elif 'access_token' in token_data:
            self.access_token = token_data['access_token']
        else:
            raise Exception("Token data does not contain 'msg' or 'access_token'")
        self.token_expiry = time.time() + 850

    def token_expired(self) -> bool:
        """
        Checks if the access token has expired.

        Returns:
            bool: True if the token has expired, False otherwise.
        """
        return time.time() >= self.token_expiry

    def ensure_authenticated(self) -> None:
        """
        Ensures that the access token is valid and not expired.
        """
        if self.access_token is None or self.token_expired():
            self.authenticate()

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(3), retry=retry_if_exception_type(requests.RequestException))
    def generate_detections(self, frame: cv2.Mat) -> Tuple[List[List[float]], cv2.Mat]:
        """
        Sends the frame to the API for detection and retrieves the detections.

        Args:
            frame (cv2.Mat): The frame to send for detection.

        Returns:
            Tuple[List[List[float]], cv2.Mat]: The detection data and the original frame.
        """
        self.ensure_authenticated()

        _, frame_encoded = cv2.imencode('.png', frame)
        frame_encoded = frame_encoded.tobytes()
        filename = f"frame_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.png"

        headers = {'Authorization': f'Bearer {self.access_token}'}
        response = self.session.post(
            f'{self.api_url}/detect',
            files={'image': (filename, frame_encoded, 'image/png')},
            params={'model': self.model_key},
            headers=headers
        )
        response.raise_for_status()
        detections = response.json()
        return detections, frame

    def run_detection(self, stream_url: str) -> None:
        """
        Runs detection on the live stream.

        Args:
            stream_url (str): The URL of the live stream.
        """
        cap = cv2.VideoCapture(stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from the stream. Retrying...")
                time.sleep(2)
                continue

            try:
                datas, _ = self.generate_detections(frame)
                self.draw_detections_on_frame(frame, datas)
            except Exception as e:
                print(f"Detection error: {e}")

            del frame, ret, datas
            gc.collect()

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform live stream detection and tracking using YOLOv8.')
    parser.add_argument('--url', type=str, help='Live stream URL', required=True)
    parser.add_argument('--api_url', type=str, default='http://localhost:5000', help='API URL for detection')
    parser.add_argument('--model_key', type=str, default='yolov8n', help='Model key for detection')
    parser.add_argument('--output_folder', type=str, help='Output folder for detected frames')
    args = parser.parse_args()

    detector = LiveStreamDetector(api_url=args.api_url, model_key=args.model_key, output_folder=args.output_folder)
    detector.run_detection(args.url)
