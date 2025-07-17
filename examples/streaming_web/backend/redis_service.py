from __future__ import annotations

import base64
import re
from typing import Any

from examples.streaming_web.backend.utils import Utils

DELIMITER = b'--IMG--'


async def scan_for_labels(rds: Any) -> list[str]:
    """
    Scan Redis keys to find distinct labels.

    Args:
        rds (Any): An asynchronous Redis client.

    Returns:
        List[str]: A sorted list of distinct labels found in Redis.
    """
    cursor: int = 0
    labels: set[str] = set()

    while True:
        # Perform a scan operation in Redis for the next batch of keys.
        cursor, keys = await rds.scan(cursor=cursor)
        if not keys:
            break

        for key in keys:
            decoded_key: str = key.decode('utf-8', errors='ignore')
            match = re.match(
                r'stream_frame:([A-Za-z0-9\-_]+=*)\|([A-Za-z0-9\-_]+=*)',
                decoded_key,
            )
            if not match:
                continue

            encoded_label, _ = match.groups()
            try:
                label_str = base64.urlsafe_b64decode(
                    encoded_label,
                ).decode('utf-8')
            except Exception:
                # If decoding fails, skip this key.
                continue

            # Filter out labels containing 'test'.
            if 'test' not in label_str:
                labels.add(label_str)

        # If the cursor returns to 0, the scan is complete.
        if cursor == 0:
            break

    return sorted(labels)


async def get_keys_for_label(rds: Any, label: str) -> list[str]:
    """
    Retrieve all stream keys in Redis for a given label.\

    Args:
        rds (Any): An asynchronous Redis client.
        label (str): The human-readable label to search for.

    Returns:
        List[str]: A sorted list of matching keys in Redis for the given label.
    """
    encoded_label = base64.urlsafe_b64encode(
        label.encode('utf-8'),
    ).decode('utf-8')
    cursor: int = 0
    matching_keys: list[str] = []

    while True:
        # Scan Redis keys that match the base64-encoded label.
        cursor, keys = await rds.scan(
            cursor=cursor,
            match=f"stream_frame:{encoded_label}|*",
        )

        for key in keys:
            decoded_key: str = key.decode('utf-8', errors='ignore')
            if decoded_key.startswith('stream_frame:'):
                matching_keys.append(decoded_key)

        if cursor == 0:
            break

    return sorted(matching_keys)


async def fetch_latest_frames(
    rds: Any,
    last_ids: dict[str, str],
) -> list[dict[str, Any]]:
    """
    Fetch the latest frames from Redis for the specified keys.

    Args:
        rds (Any): An asynchronous Redis client.
        last_ids (Dict[str, str]):
            A dictionary mapping Redis keys to their last processed IDs.

    Returns:
        List[Dict[str, Any]]:
            A list of dictionaries containing the latest frame data and
            associated metadata for each key.
    """
    updated_data: list[dict[str, Any]] = []

    for key, _ in last_ids.items():
        # Get the most recent entry from the Redis stream.
        messages = await rds.xrevrange(key, count=1)
        if not messages:
            continue

        message_id, data = messages[0]
        last_ids[key] = message_id

        frame_data: bytes | None = data.get(b'frame')
        if frame_data:
            splitted = key.split('|')
            if len(splitted) >= 2:
                encoded_stream_name = splitted[-1]
                try:
                    stream_name = base64.urlsafe_b64decode(
                        encoded_stream_name,
                    ).decode('utf-8')
                except Exception:
                    stream_name = 'Unknown'
            else:
                stream_name = 'Unknown'

            # Decode optional metadata fields.
            warnings = data.get(b'warnings')
            cone_polygons = data.get(b'cone_polygons')
            pole_polygons = data.get(b'pole_polygons')
            detection_items = data.get(b'detection_items')
            width = data.get(b'width')
            height = data.get(b'height')

            updated_data.append({
                'key': stream_name,
                'frame_bytes': frame_data,
                'warnings': warnings.decode('utf-8') if warnings else '',
                'cone_polygons': (
                    cone_polygons.decode('utf-8') if cone_polygons else ''
                ),
                'pole_polygons': (
                    pole_polygons.decode('utf-8') if pole_polygons else ''
                ),
                'detection_items': (
                    detection_items.decode('utf-8') if detection_items else ''
                ),
                'width': width.decode('utf-8') if width else '',
                'height': height.decode('utf-8') if height else '',
            })

    return updated_data


async def fetch_latest_frame_for_key(
    rds: Any,
    redis_key: str,
    last_id: str,
) -> dict[str, Any] | None:
    """
    Fetch the latest frame for a single Redis key.

    Uses XREVRANGE to retrieve the newest entry for the specified key,
    ignoring entries older than the given 'last_id'. If a new entry
    is found and it contains frame data, this function decodes
    metadata and returns a dictionary of frame information.

    Args:
        rds (Any): An asynchronous Redis client.
        redis_key (str): The Redis stream key.
        last_id (str): The last processed message ID for this key.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing frame data
            and associated metadata if a new frame is found. Otherwise,
            returns None.
    """
    messages = await rds.xrevrange(redis_key, min=last_id, count=1)
    if not messages:
        return None

    message_id, data = messages[0]
    frame_data: bytes | None = data.get(b'frame')
    if frame_data:
        warnings = data.get(b'warnings')
        cone_polygons = data.get(b'cone_polygons')
        pole_polygons = data.get(b'pole_polygons')
        detection_items = data.get(b'detection_items')
        width = data.get(b'width')
        height = data.get(b'height')

        return {
            'id': message_id.decode('utf-8'),
            'frame_bytes': frame_data,
            'warnings': warnings.decode('utf-8') if warnings else '',
            'cone_polygons': (
                cone_polygons.decode('utf-8') if cone_polygons else ''
            ),
            'pole_polygons': (
                pole_polygons.decode('utf-8') if pole_polygons else ''
            ),
            'detection_items': (
                detection_items.decode('utf-8') if detection_items else ''
            ),
            'width': width.decode('utf-8') if width else '',
            'height': height.decode('utf-8') if height else '',
        }

    return None


async def store_to_redis(
    rds: Any,
    site: str,
    stream_name: str,
    frame_bytes: bytes | None,
    warnings_json: str,
    cone_polygons_json: str,
    pole_polygons_json: str,
    detection_items_json: str,
    width: int,
    height: int,
) -> None:
    """
    Store a frame and its metadata in Redis.

    This function constructs a Redis stream key from a combination
    of 'site' and 'stream_name', both encoded via Utils.encode().

    Args:
        rds (Any): An asynchronous Redis client.
        site (str): A label representing the site or location.
        stream_name (str): A label representing the specific stream.
        frame_bytes (Optional[bytes]): The binary data of the frame to store.
        warnings_json (str): JSON string representing any warnings.
        cone_polygons_json (str): JSON string for cone polygon data.
        pole_polygons_json (str): JSON string for pole polygon data.
        detection_items_json (str): JSON string for detected items.
        width (int): The width of the frame.
        height (int): The height of the frame.

    Returns:
        None
    """
    if not frame_bytes:
        # If there is no frame, do nothing.
        return

    # Construct the Redis key using base64-encoded site and stream_name.
    key = f"stream_frame:{Utils.encode(site)}|{Utils.encode(stream_name)}"

    # Add the frame data and associated metadata to Redis Streams.
    await rds.xadd(
        key,
        {
            'frame': frame_bytes,
            'warnings': warnings_json,
            'cone_polygons': cone_polygons_json,
            'pole_polygons': pole_polygons_json,
            'detection_items': detection_items_json,
            'width': width,
            'height': height,
        },
        maxlen=10,
    )
