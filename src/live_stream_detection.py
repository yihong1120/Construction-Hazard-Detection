from __future__ import annotations

import argparse
import datetime
import gc
import os
import time
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from requests.adapters import HTTPAdapter
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_fixed
from urllib3.util import Retry

load_dotenv()


class InputData(TypedDict):
    frame: cv2.Mat
    model_key: str
    run_local: bool


class DetectionData(TypedDict):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: int


class LiveStreamDetector:
    """
    A class to perform live stream detection and tracking
    using YOLOv8 with SAHI.
    """

    def __init__(
        self,
        api_url: str = 'http://localhost:5000',
        model_key: str = 'yolov8l',
        output_folder: str | None = None,
        run_local: bool = True,
    ):
        """
        Initialises the LiveStreamDetector.

        Args:
            api_url (str): The URL of the API for detection.
            model_key (str): The model key for detection.
            output_folder (Optional[str]): Folder for detected frames.
        """
        self.api_url = api_url
        self.model_key = model_key
        self.output_folder = output_folder
        self.session = self.requests_retry_session()
        self.run_local = run_local
        self.model = None
        self.access_token = None
        self.token_expiry = 0.0

        # Load the font used for drawing labels on the image
        self.font = ImageFont.truetype(
            'assets/fonts/NotoSansTC-VariableFont_wght.ttf',
            20,
        )

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
            9: '車輛',
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
            label
            for label in self.category_id_to_name.values()
            if label not in self.colors
        ]

    def draw_detections_on_frame(
        self,
        frame: cv2.Mat,
        datas: list[list[float]],
    ) -> cv2.Mat:
        """
        Draws detections on the given frame.

        Args:
            frame (cv2.Mat): The frame on which to draw detections.
            datas (List[List[float]]): The detection data.

        Returns:
            frame_with_detections(cv2.Mat): The frame with detections drawn.
        """
        # Convert the frame to RGB and create a PIL image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)

        for data in datas:
            x1, y1, x2, y2, _, label_id = data
            label_id = int(label_id)  # Ensure label_id is an integer
            if label_id in self.category_id_to_name:
                label = self.category_id_to_name[label_id]
            else:
                continue

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            if label not in self.exclude_labels:
                color = self.colors.get(label, (255, 255, 255))
                draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
                text = f"{label}"
                text_bbox = draw.textbbox((x1, y1), text, font=self.font)
                text_width, text_height = (
                    text_bbox[2] - text_bbox[0],
                    text_bbox[3] - text_bbox[1],
                )
                text_background = (
                    x1,
                    y1 - text_height - 5,
                    x1 + text_width,
                    y1,
                )
                draw.rectangle(text_background, fill=color)

                text_y = y1 - text_height - 5 / 2 - text_height / 2
                for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    draw.text(
                        (x1 + dx, text_y + dy),
                        text,
                        fill=(0, 0, 0),
                        font=self.font,
                    )
                draw.text(
                    (x1, text_y),
                    text,
                    fill=(
                        255,
                        255,
                        255,
                    ),
                    font=self.font,
                )

        # Convert the PIL image back to OpenCV format
        frame_with_detections = cv2.cvtColor(
            np.array(pil_image),
            cv2.COLOR_RGB2BGR,
        )

        return frame_with_detections

    def save_frame(self, frame_bytes: bytearray, output_filename: str) -> None:
        """
        Saves detected frame to given output folder and filename.

        Args:
            frame_bytes (bytearray): The byte stream of the frame.
            output_filename (str): The output filename.
        """
        # Create the output directory if it does not exist
        base_output_dir = Path('detected_frames')
        output_dir = (
            base_output_dir / self.output_folder
            if self.output_folder
            else base_output_dir
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define the output path
        output_path = output_dir / f"{output_filename}.png"

        # Save the byte stream to the output path
        with open(output_path, 'wb') as f:
            f.write(frame_bytes)

        # Clean up
        del output_dir, output_path, frame_bytes
        gc.collect()

    def requests_retry_session(
        self,
        retries: int = 7,
        backoff_factor: int = 1,
        status_forcelist: tuple[int, ...] = (500, 502, 504, 401, 104),
        session: requests.Session | None = None,
        allowed_methods: frozenset = frozenset(
            ['HEAD', 'GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'TRACE'],
        ),
    ) -> requests.Session:
        """
        Configures a requests session with retry logic.

        Args:
            retries (int): The number of retry attempts.
            backoff_factor (int): The backoff factor for retries.
            status_forcelist (Tuple[int]): List of HTTP status codes for retry.
            session (Optional[requests.Session]): An optional requests session.
            allowed_methods (frozenset): The set of allowed HTTP methods.

        Returns:
            requests.Session: The configured requests requests session.
        """
        session = session or requests.Session()
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            allowed_methods=allowed_methods,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(requests.RequestException),
    )
    def authenticate(self) -> None:
        """
        Authenticates with the API and retrieves the access token.
        """
        response = self.session.post(
            f"{self.api_url}/token",
            json={
                'username': os.getenv(
                    'API_USERNAME',
                ),
                'password': os.getenv('API_PASSWORD'),
            },
        )
        response.raise_for_status()
        token_data = response.json()
        if 'msg' in token_data:
            raise Exception(token_data['msg'])
        elif 'access_token' in token_data:
            self.access_token = token_data['access_token']
        else:
            raise Exception(
                "Token data does not contain 'msg' or 'access_token'",
            )
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

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_fixed(3),
        retry=retry_if_exception_type(requests.RequestException),
    )
    def generate_detections_cloud(self, frame: cv2.Mat) -> list[list[float]]:
        """
        Sends the frame to the API for detection and retrieves the detections.

        Args:
            frame (cv2.Mat): The frame to send for detection.

        Returns:
            List[List[float]]: The detection data.
        """
        self.ensure_authenticated()

        _, frame_encoded = cv2.imencode('.png', frame)
        frame_encoded = frame_encoded.tobytes()
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        filename = f"frame_{timestamp}.png"

        headers = {'Authorization': f"Bearer {self.access_token}"}
        response = self.session.post(
            f"{self.api_url}/detect",
            files={'image': (filename, frame_encoded, 'image/png')},
            params={'model': self.model_key},
            headers=headers,
        )
        response.raise_for_status()
        detections = response.json()
        return detections

    def generate_detections_local(self, frame: cv2.Mat) -> list[list[float]]:
        """
        Generates detections locally using YOLOv8.

        Args:
            frame (cv2.Mat): The frame to send for detection.

        Returns:
            List[List[float]]: The detection data.
        """
        if self.model is None:
            model_path = Path('models/pt/') / f"best_{self.model_key}.pt"
            self.model = AutoDetectionModel.from_pretrained(
                'yolov8',
                model_path=model_path,
                device='cuda:0',
            )

        result = get_sliced_prediction(
            frame,
            self.model,
            slice_height=376,
            slice_width=376,
            overlap_height_ratio=0.3,
            overlap_width_ratio=0.3,
        )

        # Compile detection data in YOLOv8 format
        datas = []
        for object_prediction in result.object_prediction_list:
            label = int(object_prediction.category.id)
            x1, y1, x2, y2 = (
                int(x)
                for x in object_prediction.bbox.to_voc_bbox()
            )
            confidence = float(object_prediction.score.value)
            datas.append([x1, y1, x2, y2, confidence, label])

        # Remove overlapping labels for Hardhat and Safety Vest categories
        datas = self.remove_overlapping_labels(datas)

        # Remove fully contained Hardhat and Safety Vest labels
        datas = self.remove_completely_contained_labels(datas)

        return datas

    def remove_overlapping_labels(self, datas):
        """
        Removes overlapping labels for Hardhat and Safety Vest categories.

        Args:
            datas (list): A list of detection data in YOLOv8 format.

        Returns:
            list: A list of detection data with overlapping labels removed.
        """
        hardhat_indices = [
            i
            for i, d in enumerate(
                datas,
            )
            if d[5] == 0
        ]  # Indices of Hardhat detections
        # Indices of NO-Hardhat detections
        no_hardhat_indices = [i for i, d in enumerate(datas) if d[5] == 2]
        # Indices of Safety Vest detections
        safety_vest_indices = [i for i, d in enumerate(datas) if d[5] == 7]
        # Indices of NO-Safety Vest detections
        no_safety_vest_indices = [i for i, d in enumerate(datas) if d[5] == 4]

        to_remove = set()
        for hardhat_index in hardhat_indices:
            for no_hardhat_index in no_hardhat_indices:
                if (
                    self.overlap_percentage(
                        datas[hardhat_index][:4],
                        datas[no_hardhat_index][:4],
                    )
                    > 0.8
                ):
                    to_remove.add(no_hardhat_index)

        for safety_vest_index in safety_vest_indices:
            for no_safety_vest_index in no_safety_vest_indices:
                if (
                    self.overlap_percentage(
                        datas[safety_vest_index][:4],
                        datas[no_safety_vest_index][:4],
                    )
                    > 0.8
                ):
                    to_remove.add(no_safety_vest_index)

        for index in sorted(to_remove, reverse=True):
            datas.pop(index)

        gc.collect()
        return datas

    def overlap_percentage(self, bbox1, bbox2):
        """
        Calculates the percentage of overlap between two bounding boxes.

        Args:
            bbox1 (list): The first bounding box [x1, y1, x2, y2].
            bbox2 (list): The second bounding box [x1, y1, x2, y2].

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

        overlap_percentage = intersection_area / float(
            bbox1_area + bbox2_area - intersection_area,
        )
        gc.collect()

        return overlap_percentage

    def is_contained(self, inner_bbox, outer_bbox):
        """
        Determines if one bounding box is completely contained within another.

        Args:
            inner_bbox (list): The inner bounding box [x1, y1, x2, y2].
            outer_bbox (list): The outer bounding box [x1, y1, x2, y2].

        Returns:
            bool: Checks if inner box is fully within outer bounding box.
        """
        return (
            inner_bbox[0] >= outer_bbox[0]
            and inner_bbox[2] <= outer_bbox[2]
            and inner_bbox[1] >= outer_bbox[1]
            and inner_bbox[3] <= outer_bbox[3]
        )

    def remove_completely_contained_labels(self, datas):
        """
        Removes labels fully contained in Hardhat/Safety Vest categories.

        Args:
            datas (list): A list of detection data in YOLOv8 format.

        Returns:
            list: Detection data with fully contained labels removed.
        """
        hardhat_indices = [
            i
            for i, d in enumerate(
                datas,
            )
            if d[5] == 0
        ]  # Indices of Hardhat detections
        # Indices of NO-Hardhat detections
        no_hardhat_indices = [i for i, d in enumerate(datas) if d[5] == 2]
        # Indices of Safety Vest detections
        safety_vest_indices = [i for i, d in enumerate(datas) if d[5] == 7]
        # Indices of NO-Safety Vest detections
        no_safety_vest_indices = [i for i, d in enumerate(datas) if d[5] == 4]

        to_remove = set()
        # Check hardhats
        for hardhat_index in hardhat_indices:
            for no_hardhat_index in no_hardhat_indices:
                if self.is_contained(
                    datas[no_hardhat_index][:4],
                    datas[hardhat_index][:4],
                ):
                    to_remove.add(no_hardhat_index)
                elif self.is_contained(
                    datas[hardhat_index][:4],
                    datas[no_hardhat_index][:4],
                ):
                    to_remove.add(hardhat_index)

        # Check safety vests
        for safety_vest_index in safety_vest_indices:
            for no_safety_vest_index in no_safety_vest_indices:
                if self.is_contained(
                    datas[no_safety_vest_index][:4],
                    datas[safety_vest_index][:4],
                ):
                    to_remove.add(no_safety_vest_index)
                elif self.is_contained(
                    datas[safety_vest_index][:4],
                    datas[no_safety_vest_index][:4],
                ):
                    to_remove.add(safety_vest_index)

        for index in sorted(to_remove, reverse=True):
            datas.pop(index)

        return datas

    def generate_detections(
        self, frame: cv2.Mat,
    ) -> tuple[list[list[float]], cv2.Mat]:
        """
        Generates detections with local model or cloud API as configured.

        Args:
            frame (cv2.Mat): The frame to send for detection.

        Returns:
            Tuple[List[List[float]], cv2.Mat]: Detections and original frame.
        """
        if self.run_local:
            datas = self.generate_detections_local(frame)
        else:
            datas = self.generate_detections_cloud(frame)

        return datas, frame

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
                print('Failed to read frame from the stream. Retrying...')
                time.sleep(2)
                continue

            try:
                datas, _ = self.generate_detections(frame)
                frame_with_detections = self.draw_detections_on_frame(
                    frame, datas,
                )
                cv2.imshow('Live Stream Detection', frame_with_detections)
            except Exception as e:
                print(f"Detection error: {e}")

            # Clear variables to free up memory
            del frame, ret, datas
            gc.collect()

            # Break loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


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
        '--api_url',
        type=str,
        default='http://localhost:5000',
        help='API URL for detection',
    )
    parser.add_argument(
        '--model_key',
        type=str,
        default='yolov8n',
        help='Model key for detection',
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        help='Output folder for detected frames',
    )
    parser.add_argument(
        '--run_local',
        action='store_true',
        help='Run detection using local model',
    )
    args = parser.parse_args()

    detector = LiveStreamDetector(
        api_url=args.api_url,
        model_key=args.model_key,
        output_folder=args.output_folder,
        run_local=args.run_local,
    )
    detector.run_detection(args.url)
