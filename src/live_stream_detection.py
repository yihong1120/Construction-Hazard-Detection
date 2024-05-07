import argparse
import cv2
import datetime
from pathlib import Path
from typing import Generator, Tuple, List
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
    def __init__(self, api_url: str = 'http://localhost:5000', model_key: str = 'yolov8l', output_folder: str = None, output_filename: str = 'detected_frame.jpg'):
        '''
        Initialise the LiveStreamDetector object with the provided stream URL, model path, and output filename.

        Args:
            api_url (str): The URL of the SAHI API for object detection.
            model_key (str): The key of the model to use for detection.
            output_folder (str): The folder to save the detected frames.
            output_filename (str): The filename to save the detected frames.
        '''
        self.api_url = api_url
        self.model_key = model_key
        self.output_folder = output_folder
        self.output_filename = output_filename
        self.session = self.requests_retry_session()
        
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

        # Load the font for text drawing
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
        self.authenticate()  # Authenticate with the API server to obtain JWT

    def load_font(self):
        '''
        Load font for text drawing

        Returns:
            font_path (str): The path to the font file.
        '''
        font_path = "assets/fonts/NotoSansTC-VariableFont_wght.ttf"
        font = ImageFont.truetype(font_path, 20)  # Define font size and font
        return font_path, font

    def draw_detections_on_frame(self, frame: cv2.Mat, datas: List) -> None:
        """
        Draws detection boxes on the frame with specific colours based on the label,
        and saves the frame to '../detected_frames/' directory with the provided filename.

        Args:
            frame (cv2.Mat): The original video frame.
            datas (List): The list of detection data.

        Returns:
            None

        Raises:
            Exception: If the file cannot be saved.
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
        Saves the frame to the specified directory and filename.

        Args:
            frame (cv2.Mat): The frame to be saved.

        Returns:
            None

        Raises:
            Exception: If the file cannot be saved.
        """
        # Define the base directory for detected frames
        base_output_dir = Path('detected_frames')
        
        # Include additional subdirectory if specified
        if self.output_folder:
            output_dir = base_output_dir / self.output_folder
        else:
            output_dir = base_output_dir
        
        # Ensure the directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define the complete output path including the filename
        output_path = output_dir / self.output_filename
        
        # # Check if the file exists and remove it if it does
        # if output_path.exists():
        #     output_path.unlink() 

        # Save the image to the specified path
        cv2.imwrite(str(output_path), frame)

        # Optional: Print the path where the frame was saved (can be removed in production)
        # print(f"Frame saved to: {output_path}")

        # Clear memory by deleting variables
        del output_dir, output_path
        gc.collect()

    def requests_retry_session(
        self,
        retries=7,  # Number of retries
        backoff_factor=1,  # Exponential backoff factor
        status_forcelist=(500, 502, 504, 401, 104),
        session=None,
        allowed_methods=frozenset(['HEAD', 'GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'TRACE'])
    ):
        """
        Retry session for handling network errors and retries.

        Args:
            retries (int): The number of retries for the session.
            backoff_factor (int): The backoff factor for retries.
            status_forcelist (tuple): The status codes to force a retry.
            session (requests.Session): The session to use for retries.
            allowed_methods (frozenset): The allowed methods for the session.

        Returns:
            requests.Session: The session with retries enabled.

        Raises:
            requests.exceptions.RequestException: If an error occurs during the request.
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
    def authenticate(self):
        """
        Authenticate with the API server to obtain JWT.

        Returns:
            str: The access token for authentication.

        Raises:
            Exception: If the authentication fails.
        """
        response = self.session.post(
            f"{self.api_url}/token",
            json={"username": os.getenv('API_USERNAME'), "password": os.getenv('API_PASSWORD')}
        )
        response.raise_for_status()  # Raise an exception for 4xx/5xx status codes
        token_data = response.json()
        if 'msg' in token_data:
            raise Exception(token_data['msg'])
        elif 'access_token' in token_data:
            self.access_token = token_data['access_token']
        else:
            raise Exception("Token data does not contain 'msg' or 'access_token'")

        # Assume the token expires in 15 minutes (900 seconds)
        self.token_expiry = time.time() + 890

    def token_expired(self):
        """
        Check if the current access token has expired.
        
        Args:
            token_expiry (float): The timestamp when the token expires.

        Returns:
            bool: True if the token has expired, False otherwise.

        Raises:
            Exception: If the token expiry time is not set.
        """
        return time.time() >= self.token_expiry

    def ensure_authenticated(self):
        """
        Ensure that the detector is authenticated and the token is valid.
        
        Args:
            api_url (str): The URL of the SAHI API for object detection.
            model_key (str): The key of the model to use for detection.

        Returns:
            str: The access token for authentication.

        Raises:
            Exception: If the authentication fails.
        """
        if self.access_token is None or self.token_expired():
            self.authenticate()

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(3), retry=retry_if_exception_type(requests.RequestException))
    def generate_detections(self, frame) -> Generator[Tuple[List, cv2.Mat, float], None, None]:
        """
        Generates detections from the video stream, capturing frames every five seconds.

        Args:
            frame (cv2.Mat): The frame to generate detections from.

        Returns:
            Generator: A generator yielding a tuple of detections, frame, and timestamp.

        Raises:
            Exception: If an error occurs during the request.
        """
        self.ensure_authenticated()

        _, frame_encoded = cv2.imencode('.png', frame)
        frame_encoded = frame_encoded.tobytes()
        filename = f"frame_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.png"

        headers = {'Authorization': f'Bearer {self.access_token}'}
        response = requests.post(
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
        Runs the detection process and draws the results on each frame continuously.
        """
        cap = cv2.VideoCapture(stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while True:
            ret, frame = cap.read()
            if ret:
                datas, _ = self.generate_detections(frame)
                self.draw_detections_on_frame(frame, datas)
            else:
                break

            del frame, ret, datas
            gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform live stream detection and tracking using YOLOv8.')
    parser.add_argument('--url', type=str, help='Live stream URL', required=True)
    parser.add_argument('--api_url', type=str, default='http://localhost:5000', help='API URL for detection')
    parser.add_argument('--model_key', type=str, default='yolov8n', help='Model key for detection')
    parser.add_argument('--output', type=str, default='detected_frame.jpg', help='Output image file name')
    args = parser.parse_args()

    # Initialise the live stream detector with the provided stream URL, API URL, model key, and output filename.
    detector = LiveStreamDetector(args.api_url, args.model_key, args.output)

    # Run the detection process and continuously output images
    detector.authenticate()
    detector.run_detection(args.url)

"""example
python live_stream_detection.py --url https://cctv6.kctmc.nat.gov.tw/ea05668e/
"""