from __future__ import annotations

import argparse
import asyncio
import gc
import logging
import os
from pathlib import Path
from typing import MutableMapping
from typing import TypedDict

import aiohttp
import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


class InputData(TypedDict):
    frame: np.ndarray
    model_key: str
    detect_with_server: bool


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
    using YOLO with SAHI.
    """

    def __init__(
        self,
        api_url: str = 'http://localhost:5000',
        model_key: str = 'yolo11n',
        output_folder: str | None = None,
        detect_with_server: bool = False,
        shared_token: MutableMapping[str, str] | None = None,
        shared_lock=None,
    ):
        """
        Initialises the LiveStreamDetector.

        Args:
            api_url (str): The URL of the API for detection.
            model_key (str): The model key for detection.
            output_folder (Optional[str]): Folder for detected frames.
            detect_with_server (bool): Whether to use server-based detection.
            shared_token (Optional[dict]): A shared dictionary for
                token storage.
            shared_lock: A shared multiprocessing Lock to
                avoid multiple logins simultaneously.
        """
        self.api_url: str = (
            api_url if api_url.startswith('http') else f"http://{api_url}"
        )
        self.model_key: str = model_key
        self.output_folder: str | None = output_folder
        self.detect_with_server: bool = detect_with_server
        self.shared_token: MutableMapping[str, str] = shared_token or dict(
            access_token='',
        )
        self.shared_lock = shared_lock
        self.model: AutoDetectionModel | None = None
        self.logger = logging.getLogger(__name__)

    #######################################################################
    # Authentication functions
    #######################################################################

    def acquire_shared_lock(self):
        """
        Acquires the shared lock for authentication.
        """
        if self.shared_lock:
            self.shared_lock.acquire()

    def release_shared_lock(self):
        """
        Releases the shared lock for authentication.
        """
        if self.shared_lock:
            self.shared_lock.release()

    async def authenticate(self, force: bool = False) -> None:
        """
        Ensures that the user is authenticated, re-authenticates if needed.

        Args:
            force (bool): Whether to force re-authentication.

        Raises:
            aiohttp.ClientResponseError: If the authentication fails.
            ValueError: If credentials are missing.
        """
        username = os.getenv('API_USERNAME')
        password = os.getenv('API_PASSWORD')

        if not username or not password:
            raise ValueError(
                'Missing API_USERNAME '
                'or API_PASSWORD in environment variables',
            )

        # Check if re-authentication is needed
        if not force and self.shared_token.get('access_token'):
            return

        self.acquire_shared_lock()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/api/token",
                    json={
                        'username': username,
                        'password': password,
                    },
                ) as response:
                    response.raise_for_status()
                    token_data = await response.json()
                    self.shared_token['access_token'] = (
                        token_data['access_token']
                    )
                    self.logger.info(
                        'Successfully authenticated and retrieved token.',
                    )
        finally:
            self.release_shared_lock()

    #######################################################################
    # Detection functions
    #######################################################################

    async def generate_detections_cloud(
        self,
        frame: np.ndarray,
    ) -> list[list[float]]:
        """
        Sends the frame to the API for detection and retrieves the detections.

        Args:
            frame (np.ndarray): The frame to send for detection.

        Returns:
            list[list[float]]: The detection data.
        """
        # Encode frame as PNG bytes
        success, frame_encoded = cv2.imencode('.png', frame)
        if not success:
            raise ValueError('Failed to encode frame as PNG bytes.')
        frame_bytes = frame_encoded.tobytes()

        # Ensure authenticated
        await self.authenticate()

        # Send detection request
        try:
            headers = {
                'Authorization': f"Bearer {self.shared_token['access_token']}",
            }
            data = aiohttp.FormData()
            data.add_field(
                'image',
                frame_bytes,
                filename='frame.png',
                content_type='image/png',
            )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/api/detect",
                    data=data,
                    params={'model': self.model_key},
                    headers=headers,
                ) as response:
                    # Token expired or invalid
                    if response.status in (401, 403):
                        self.logger.warning(
                            'Token expired or invalid. Re-authenticating...',
                        )
                        # Re-authenticate and retry detection request
                        await self.authenticate(force=True)

                        # Retry detection request
                        return await self.generate_detections_cloud(frame)
                    response.raise_for_status()
                    return await response.json()
        except aiohttp.ClientResponseError as exc:
            self.logger.error(f"Failed to send detection request: {exc}")
            raise

    async def generate_detections_local(
        self,
        frame: np.ndarray,
    ) -> list[list[float]]:
        """
        Generates detections locally using YOLO.

        Args:
            frame (np.ndarray): The frame to send for detection.

        Returns:
            list[list[float]]: The detection data.
        """
        if self.model is None:
            model_path = Path('models/pt/') / f"best_{self.model_key}.pt"
            self.model = AutoDetectionModel.from_pretrained(
                'yolo11',
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

        # Compile detection data in YOLO format
        datas = []
        for obj in result.object_prediction_list:
            label = int(obj.category.id)
            x1, y1, x2, y2 = (int(x) for x in obj.bbox.to_voc_bbox())
            confidence = float(obj.score.value)
            datas.append([x1, y1, x2, y2, confidence, label])

        # Remove overlapping labels for Hardhat and Safety Vest categories
        datas = self.remove_overlapping_labels(datas)

        # Remove fully contained Hardhat and Safety Vest labels
        datas = self.remove_completely_contained_labels(datas)

        return datas

    async def generate_detections(
        self, frame: np.ndarray,
    ) -> tuple[list[list[float]], np.ndarray]:
        """
        Generates detections with local model or cloud API as configured.

        Args:
            frame (np.ndarray): The frame to send for detection.

        Returns:
            Tuple[List[List[float]], np.ndarray]:
                Detections and original frame.
        """
        if self.detect_with_server:
            datas = await self.generate_detections_cloud(frame)
        else:
            datas = await self.generate_detections_local(frame)
        return datas, frame

    async def run_detection(self, stream_url: str) -> None:
        """
        Runs detection on the live stream.

        Args:
            stream_url (str): The URL of the live stream.
        """
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            raise ValueError('Failed to open stream.')

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print('Failed to read frame from the stream. Retrying...')
                    continue

                # Perform detection
                datas, frame = await self.generate_detections(frame)
                print(datas)  # You can replace this with actual processing

                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    #######################################################################
    # Post-processing functions
    #######################################################################

    def remove_overlapping_labels(self, datas):
        """
        Removes overlapping labels for Hardhat and Safety Vest categories.

        Args:
            datas (list): A list of detection data in YOLO format.

        Returns:
            list: A list of detection data with overlapping labels removed.
        """
        # Indices of Hardhat detections
        hardhat_indices = [
            i for i, d in enumerate(
                datas,
            ) if d[5] == 0
        ]
        # Indices of NO-Hardhat detections
        no_hardhat_indices = [i for i, d in enumerate(datas) if d[5] == 2]
        # Indices of Safety Vest detections
        safety_vest_indices = [i for i, d in enumerate(datas) if d[5] == 7]
        # Indices of NO-Safety Vest detections
        no_safety_vest_indices = [i for i, d in enumerate(datas) if d[5] == 4]

        to_remove = set()
        for hardhat_index in hardhat_indices:
            for no_hardhat_index in no_hardhat_indices:
                overlap = self.overlap_percentage(
                    datas[hardhat_index][:4], datas[no_hardhat_index][:4],
                )
                if overlap > 0.8:
                    to_remove.add(no_hardhat_index)

        for safety_vest_index in safety_vest_indices:
            for no_safety_vest_index in no_safety_vest_indices:
                overlap = self.overlap_percentage(
                    datas[safety_vest_index][:4],
                    datas[no_safety_vest_index][:4],
                )
                if overlap > 0.8:
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
            datas (list): A list of detection data in YOLO format.

        Returns:
            list: Detection data with fully contained labels removed.
        """
        # Indices of Hardhat detections
        hardhat_indices = [
            i
            for i, d in enumerate(
                datas,
            )
            if d[5] == 0
        ]

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


async def main():
    parser = argparse.ArgumentParser(
        description='Perform live stream detection and tracking using YOLO.',
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
        default='yolo11n',
        help='Model key for detection',
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        help='Output folder for detected frames',
    )
    parser.add_argument(
        '--detect_with_server',
        action='store_true',
        help='Run detection using server api',
    )
    args = parser.parse_args()

    # Shared token for authentication
    shared_token = {
        'access_token': '',
    }

    detector = LiveStreamDetector(
        api_url=args.api_url,
        model_key=args.model_key,
        output_folder=args.output_folder,
        detect_with_server=args.detect_with_server,
        # If you want to share token across threads, use Manager()
        shared_token=shared_token,
    )
    await detector.run_detection(args.url)


if __name__ == '__main__':
    asyncio.run(main())
