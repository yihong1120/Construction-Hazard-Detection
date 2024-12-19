from __future__ import annotations

import asyncio
import base64
import logging
import os
from datetime import datetime
from typing import Any

import cv2
import numpy as np
import redis.asyncio as redis
from shapely.geometry import MultiPoint
from shapely.geometry import Point
from shapely.geometry import Polygon
from sklearn.cluster import HDBSCAN
from watchdog.events import FileSystemEventHandler

from src.lang_config import Translator


class Utils:
    """
    A class to provide utility functions.
    """

    @staticmethod
    def is_expired(expire_date_str: str | None) -> bool:
        """
        Check if the given expire date string is expired.

        Args:
            expire_date_str (str | None): The expire date string
                in ISO 8601 format.

        Returns:
            bool: True if expired, False otherwise.
        """
        if expire_date_str:
            try:
                expire_date = datetime.fromisoformat(expire_date_str)
                return datetime.now() > expire_date
            except ValueError:
                # If the string cannot be parsed as a valid ISO 8601 date
                return False
        return False

    @staticmethod
    def encode(value: str) -> str:
        """
        Encode a value into a URL-safe Base64 string.

        Args:
            value (str): The value to encode.

        Returns:
            str: The encoded string.
        """
        return base64.urlsafe_b64encode(
            value.encode('utf-8'),
        ).decode('utf-8')

    @staticmethod
    def encode_frame(frame: Any) -> bytes | None:
        try:
            _, buffer = cv2.imencode('.png', frame)
            return buffer.tobytes()
        except Exception as e:
            logging.error(f"Error encoding frame: {e}")
            return None

    @staticmethod
    def generate_message(
        stream_name: str,
        detection_time: datetime,
        warnings: list[str],
        controlled_zone_warning: list[str],
        language: str,
        is_working_hour: bool,
    ) -> str | None:
        """
        Generate a message to send to the notification service.

        Args:
            stream_name (str): The name of the stream.
            detection_time (datetime): The time of detection.
            warnings (list[str]): The list of warnings.
            controlled_zone_warning (list[str]):
                The list of controlled zone warnings.
            language (str): The language for the warnings.
            is_working_hour (bool): Whether it is working hours.

        Returns:
            str | None: The message to send, or None if no message to send.
        """
        if is_working_hour and warnings:
            translated_warnings = Translator.translate_warning(
                tuple(warnings), language,
            )
            return (
                f"{stream_name}\n[{detection_time}]\n"
                + '\n'.join(translated_warnings)
            )

        if not is_working_hour and controlled_zone_warning:
            translated_controlled_zone_warning = (
                Translator.translate_warning(
                    tuple(controlled_zone_warning), language,
                )
            )
            return (
                f"{stream_name}\n[{detection_time}]\n"
                + '\n'.join(translated_controlled_zone_warning)
            )

        return None

    @staticmethod
    def should_notify(
        timestamp: int,
        last_notification_time: int,
        cooldown_period: int = 300,
    ) -> bool:
        """
        Check if a notification should be sent based on the cooldown period.

        Args:
            timestamp (int): The current timestamp.
            last_notification_time (int):
                The timestamp of the last notification.
            cooldown_period (int): The cooldown period in seconds.

        Returns:
            bool: True if a notification should be sent, False otherwise.
        """
        return (timestamp - last_notification_time) >= cooldown_period

    @staticmethod
    def normalise_bbox(bbox: list[float]) -> list[float]:
        """
        Normalises the bounding box coordinates.

        Args:
            bbox (List[float]): The bounding box coordinates.

        Returns:
            List[float]: Normalised coordinates.
        """
        left_x = min(bbox[0], bbox[2])
        right_x = max(bbox[0], bbox[2])
        top_y = min(bbox[1], bbox[3])
        bottom_y = max(bbox[1], bbox[3])
        if len(bbox) > 4:
            return [left_x, top_y, right_x, bottom_y, bbox[4], bbox[5]]
        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def normalise_data(datas: list[list[float]]) -> list[list[float]]:
        """
        Normalises a list of bounding box data.

        Args:
            datas (List[List[float]]): List of bounding box data.

        Returns:
            List[List[float]]: Normalised data.
        """
        return [Utils.normalise_bbox(data[:4] + data[4:]) for data in datas]

    @staticmethod
    def overlap_percentage(bbox1: list[float], bbox2: list[float]) -> float:
        """
        Calculate the overlap percentage between two bounding boxes.

        Args:
            bbox1 (List[float]): The first bounding box.
            bbox2 (List[float]): The second bounding box.

        Returns:
            float: The overlap percentage.
        """
        # Calculate the coordinates of the intersection rectangle
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        # Calculate the area of the intersection rectangle
        overlap_area = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate the area of both bounding boxes
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        # Calculate the overlap percentage
        return overlap_area / float(area1 + area2 - overlap_area)

    @staticmethod
    def is_driver(person_bbox: list[float], vehicle_bbox: list[float]) -> bool:
        """
        Check if a person is a driver based on position near a vehicle.

        Args:
            person_bbox (List[float]): Bounding box of person.
            vehicle_bbox (List[float]): Bounding box of vehicle.

        Returns:
            bool: True if the person is likely the driver, False otherwise.
        """
        # Extract coordinates and dimensions of person and vehicle boxes
        person_bottom_y = person_bbox[3]
        person_top_y = person_bbox[1]
        person_left_x = person_bbox[0]
        person_right_x = person_bbox[2]
        person_width = person_bbox[2] - person_bbox[0]
        person_height = person_bbox[3] - person_bbox[1]

        vehicle_top_y = vehicle_bbox[1]
        vehicle_bottom_y = vehicle_bbox[3]
        vehicle_left_x = vehicle_bbox[0]
        vehicle_right_x = vehicle_bbox[2]
        vehicle_height = vehicle_bbox[3] - vehicle_bbox[1]

        # 1. Check vertical bottom position: person's bottom should be above
        #    the vehicle's bottom by at least half the person's height
        if not (
            person_bottom_y < vehicle_bottom_y
            and vehicle_bottom_y - person_bottom_y >= person_height / 2
        ):
            return False

        # 2. Check horizontal position: person's edges should not extend
        #    beyond half the width of the person from the vehicle's edges
        if not (
            person_left_x >= vehicle_left_x - person_width / 2
            and person_right_x <= vehicle_right_x + person_width / 2
        ):
            return False

        # 3. The person's top must be below the vehicle's top
        if not (person_top_y > vehicle_top_y):
            return False

        # 4. Person's height is less than or equal to half the vehicle's height
        if not (person_height <= vehicle_height / 2):
            return False

        return True

    @staticmethod
    def is_dangerously_close(
        person_bbox: list[float],
        vehicle_bbox: list[float],
        label: str,
    ) -> bool:
        """
        Determine if a person is dangerously close to machinery or vehicles.

        Args:
            person_bbox (list[float]): Bounding box of person.
            vehicle_bbox (list[float]): Machine/vehicle box.
            label (str): Type of the second object ('machinery' or 'vehicle').

        Returns:
            bool: True if the person is dangerously close, False otherwise.
        """
        # Calculate dimensions of the person bounding box
        person_width = person_bbox[2] - person_bbox[0]
        person_height = person_bbox[3] - person_bbox[1]
        person_area = person_width * person_height

        # Calculate the area of the vehicle bounding box
        vehicle_area = (vehicle_bbox[2] - vehicle_bbox[0]) * \
            (vehicle_bbox[3] - vehicle_bbox[1])
        acceptable_ratio = 0.1 if label == 'vehicle' else 0.05

        # Check if person area ratio is acceptable compared to vehicle area
        if person_area / vehicle_area > acceptable_ratio:
            return False

        # Define danger distances
        danger_distance_horizontal = 5 * person_width
        danger_distance_vertical = 1.5 * person_height

        # Calculate min horizontal/vertical distance between person and vehicle
        horizontal_distance = min(
            abs(person_bbox[2] - vehicle_bbox[0]),
            abs(person_bbox[0] - vehicle_bbox[2]),
        )
        vertical_distance = min(
            abs(person_bbox[3] - vehicle_bbox[1]),
            abs(person_bbox[1] - vehicle_bbox[3]),
        )

        # Determine if the person is dangerously close
        return (
            horizontal_distance <= danger_distance_horizontal
            and vertical_distance <= danger_distance_vertical
        )

    @staticmethod
    def detect_polygon_from_cones(
        datas: list[list[float]],
        clusterer: HDBSCAN,
    ) -> list[Polygon]:
        """
        Detects polygons from the safety cones in the detection data.

        Args:
            datas (list[list[float]]): The detection data.

        Returns:
            list[Polygon]: A list of polygons formed by the safety cones.
        """
        if not datas:
            return []

        # Get positions of safety cones
        cone_positions = np.array([
            (
                (float(data[0]) + float(data[2])) / 2,
                (float(data[1]) + float(data[3])) / 2,
            )
            for data in datas if data[5] == 6
        ])

        # Check if there are at least three safety cones to form a polygon
        if len(cone_positions) < 3:
            return []

        # Cluster the safety cones
        labels = clusterer.fit_predict(cone_positions)

        # Extract clusters
        clusters: dict[int, list[np.ndarray]] = {}
        for point, label in zip(cone_positions, labels):
            if label == -1:
                continue  # Skip noise points
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(point)

        # Create polygons from clusters
        polygons = []
        for cluster_points in clusters.values():
            if len(cluster_points) >= 3:
                polygon = MultiPoint(cluster_points).convex_hull
                polygons.append(polygon)

        return polygons

    @staticmethod
    def calculate_people_in_controlled_area(
        polygons: list[Polygon],
        datas: list[list[float]],
    ) -> int:
        """
        Calculates the number of people within the safety cone area.

        Args:
            polygons (list[Polygon]): Polygons representing controlled areas.
            datas (list[list[float]]): The detection data.

        Returns:
            int: The number of people within the controlled area.
        """
        # Check if there are any detections
        if not datas:
            return 0

        # Check if there are valid polygons
        if not polygons:
            return 0

        # Use a set to track unique people
        unique_people = set()

        # Count the number of people within the controlled area
        for data in datas:
            if data[5] == 5:  # Check if it's a person
                x_center = (data[0] + data[2]) / 2
                y_center = (data[1] + data[3]) / 2
                point = Point(x_center, y_center)
                for polygon in polygons:
                    if polygon.contains(point):
                        # Update the set of unique people
                        unique_people.add((x_center, y_center))
                        break  # No need to check other polygons

        return len(unique_people)


class FileEventHandler(FileSystemEventHandler):
    """
    A class to handle file events.
    """

    def __init__(self, file_path: str, callback, loop):
        """
        Initialises the FileEventHandler instance.

        Args:
            file_path (str): The path of the file to watch.
            callback (Callable): The function to call when file is modified.
            loop (asyncio.AbstractEventLoop): The asyncio event loop.
        """
        self.file_path = os.path.abspath(file_path)
        self.callback = callback
        self.loop = loop

    def on_modified(self, event):
        """
        Called when a file is modified.

        Args:
            event (FileSystemEvent): The event object.
        """
        event_path = os.path.abspath(event.src_path)
        if event_path == self.file_path:
            print(f"[DEBUG] Configuration file modified: {event_path}")
            asyncio.run_coroutine_threadsafe(
                # Ensure the callback is run in the loop
                self.callback(), self.loop,
            )


class RedisManager:
    """
    A class to manage Redis operations.
    """

    def __init__(
        self,
        redis_host: str = '127.0.0.1',
        redis_port: int = 6379,
        redis_password: str = '',
    ) -> None:
        """
        Initialises RedisManager with Redis configuration details.

        Args:
            redis_host (str): The Redis server hostname.
            redis_port (int): The Redis server port.
            redis_password (str): The Redis password for authentication.
        """
        self.redis_host: str = os.getenv('REDIS_HOST') or redis_host
        self.redis_port: int = int(os.getenv('REDIS_PORT') or redis_port)
        self.redis_password: str = os.getenv(
            'REDIS_PASSWORD',
        ) or redis_password

        # Create Redis connection
        self.redis = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            password=self.redis_password,
            decode_responses=False,
        )

    async def set(self, key: str, value: bytes) -> None:
        """
        Set a key-value pair in Redis.

        Args:
            key (str): The key under which to store the value.
            value (bytes): The value to store (in bytes).
        """
        try:
            await self.redis.set(key, value)
        except Exception as e:
            logging.error(f"Error setting Redis key {key}: {str(e)}")

    async def get(self, key: str) -> bytes | None:
        """
        Retrieve a value from Redis based on the key.

        Args:
            key (str): The key whose value needs to be retrieved.

        Returns:
            bytes | None: The value if found, None otherwise.
        """
        try:
            return await self.redis.get(key)
        except Exception as e:
            logging.error(f"Error retrieving Redis key {key}: {str(e)}")
            return None

    async def delete(self, key: str) -> None:
        """
        Delete a key from Redis.

        Args:
            key (str): The key to delete from Redis.
        """
        try:
            await self.redis.delete(key)
        except Exception as e:
            logging.error(f"Error deleting Redis key {key}: {str(e)}")

    async def add_to_stream(
        self,
        stream_name: str,
        data: dict,
        maxlen: int = 10,
    ) -> None:
        """
        Add data to a Redis stream with a maximum length.

        Args:
            stream_name (str): The name of the Redis stream.
            data (dict): The data to add to the stream.
            maxlen (int): The maximum length of the stream.
        """
        try:
            await self.redis.xadd(stream_name, data, maxlen=maxlen)
        except Exception as e:
            logging.error(
                f"Error adding to Redis stream {stream_name}: {str(e)}",
            )

    async def read_from_stream(
        self,
        stream_name: str,
        last_id: str = '0',
    ) -> list:
        """
        Read data from a Redis stream.

        Args:
            stream_name (str): The name of the Redis stream.
            last_id (str): The ID of the last read message.

        Returns:
            list: A list of messages from the stream.
        """
        try:
            return await self.redis.xread({stream_name: last_id})
        except Exception as e:
            logging.error(
                f"Error reading from Redis stream {stream_name}: {str(e)}",
            )
            return []

    async def delete_stream(self, stream_name: str) -> None:
        """
        Delete a Redis stream.

        Args:
            stream_name (str): The name of the Redis stream to delete.
        """
        try:
            await self.redis.delete(stream_name)
            logging.info(f"Deleted Redis stream: {stream_name}")
        except Exception as e:
            logging.error(
                f"Error deleting Redis stream {stream_name}: {str(e)}",
            )

    async def close_connection(self) -> None:
        """
        Close the Redis connection.
        """
        try:
            await self.redis.close()
            logging.info('[INFO] Redis connection successfully closed.')
        except Exception as e:
            logging.error(f"[ERROR] Failed to close Redis connection: {e}")

    async def store_to_redis(
        self,
        site: str,
        stream_name: str,
        frame_bytes: bytes | None,
        warnings: list[str],
        language: str = 'en',
    ) -> None:
        """
        Store frame and warnings to a Redis stream.

        Args:
            site (str): Site name.
            stream_name (str): Stream name.
            frame_bytes (optional[bytes]): Encoded frame bytes.
            warnings (list[str]): List of warnings.
            language (str): Language for the warnings.
        """
        # Check if frame is None
        if not frame_bytes:
            return

        # Generate the Redis key
        key = f"stream_frame:{Utils.encode(site)}|{Utils.encode(stream_name)}"

        # Translate warnings to the specified language
        warnings_to_translate = warnings if warnings else ['No warning']
        translated_warnings = Translator.translate_warning(
            tuple(warnings_to_translate), language,
        )
        warnings_str = '\n'.join(translated_warnings)

        try:
            await self.add_to_stream(
                key,
                {'frame': frame_bytes, 'warnings': warnings_str},
                maxlen=10,
            )
        except Exception as e:
            logging.error(f"Error storing data to Redis: {e}")
