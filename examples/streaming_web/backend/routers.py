from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi_limiter.depends import RateLimiter

from examples.auth.jwt_config import jwt_access
from examples.auth.redis_pool import get_redis_pool
from examples.auth.redis_pool import get_redis_pool_ws
from examples.streaming_web.backend.redis_service import DELIMITER
from examples.streaming_web.backend.redis_service import (
    fetch_latest_frame_for_key,
)
from examples.streaming_web.backend.redis_service import fetch_latest_frames
from examples.streaming_web.backend.redis_service import get_keys_for_label
from examples.streaming_web.backend.redis_service import scan_for_labels
from examples.streaming_web.backend.redis_service import store_to_redis
from examples.streaming_web.backend.schemas import FramePostResponse
from examples.streaming_web.backend.schemas import LabelListResponse
from examples.streaming_web.backend.utils import Utils

# Create a FastAPI router instance to encapsulate endpoints.
router = APIRouter()

# Rate limiters to limit endpoint calls.
rate_limiter_index = RateLimiter(times=60, seconds=60)
rate_limiter_label = RateLimiter(times=600, seconds=60)


@router.get(
    '/labels',
    response_model=LabelListResponse,
    dependencies=[Depends(rate_limiter_index)],
)
async def get_labels_route(rds=Depends(get_redis_pool)) -> LabelListResponse:
    """
    Fetches a list of labels from Redis.

    Args:
        rds (Any): The Redis connection pool dependency.

    Returns:
        LabelListResponse: A Pydantic model containing the list of labels.

    Raises:
        HTTPException: If there is an error when fetching labels from Redis.
    """
    try:
        labels = await scan_for_labels(rds)
        return LabelListResponse(labels=labels)
    except Exception as e:
        print(f"Failed to fetch labels: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch labels: {str(e)}",
        )


@router.post(
    '/frames',
    response_model=FramePostResponse,
    dependencies=[Depends(jwt_access)],
)
async def post_frame(
    label: str = Form(...),
    key: str = Form(...),
    file: UploadFile = File(...),
    warnings_json: str = Form(''),
    cone_polygons_json: str = Form(''),
    pole_polygons_json: str = Form(''),
    detection_items_json: str = Form(''),
    width: int = Form(0),
    height: int = Form(0),
    rds=Depends(get_redis_pool),
) -> FramePostResponse:
    """
    Stores a frame in Redis along with associated metadata.

    Args:
        label (str): A site or camera label to group frames.
        key (str): A key identifying a specific video feed or stream.
        file (UploadFile): The uploaded image file.
        warnings_json (str, optional): JSON string of warning messages.
        cone_polygons_json (str, optional): JSON string of cone polygon data.
        pole_polygons_json (str, optional): JSON string of pole polygon data.
        detection_items_json (str, optional): JSON string of detected items.
        width (int, optional): The width of the frame. Defaults to 0.
        height (int, optional): The height of the frame. Defaults to 0.
        rds (Any): The Redis connection pool dependency.

    Returns:
        FramePostResponse: A Pydantic model indicating the status and message.

    Raises:
        HTTPException: If storing the frame in Redis fails.
    """
    try:
        # Read the file bytes asynchronously.
        frame_bytes = await file.read()

        # Store the frame and its associated data in Redis.
        await store_to_redis(
            rds,
            label,
            key,
            frame_bytes,
            warnings_json,
            cone_polygons_json,
            pole_polygons_json,
            detection_items_json,
            width,
            height,
        )

        return FramePostResponse(
            status='ok',
            message='Frame stored successfully.',
        )
    except Exception as e:
        print(f"Failed to store frame: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store frame: {str(e)}",
        )


@router.websocket('/ws/labels/{label}')
async def websocket_label_stream(
    websocket: WebSocket,
    label: str,
    rds=Depends(get_redis_pool_ws),
) -> None:
    """
    Provides a WebSocket endpoint for streaming frames associated with
    a specific label.

    Args:
        websocket (WebSocket): The active WebSocket connection.
        label (str): The label under which frames are grouped.
        rds (Any): The Redis connection pool dependency for WebSocket usage.
    """
    await websocket.accept()
    try:
        # Fetch all keys associated with the provided label.
        keys = await get_keys_for_label(rds, label)
        if not keys:
            # Send an error response if no keys are found for this label.
            await websocket.send_json({
                'error': f"No keys found for label '{label}'",
            })
            await websocket.close()
            return

        # Track the last processed ID for each stream to avoid duplicates.
        last_ids: dict[str, str] = {k: '0' for k in keys}

        while True:
            # Fetch the latest frames for all keys since their last IDs.
            updated_data = await fetch_latest_frames(rds, last_ids)
            if updated_data:
                for data in updated_data:
                    # Construct a JSON header containing metadata.
                    header = json.dumps({
                        'key': data.get('key', ''),
                        'warnings': data.get('warnings', ''),
                        'cone_polygons': data.get('cone_polygons', ''),
                        'pole_polygons': data.get('pole_polygons', ''),
                        'detection_items': data.get('detection_items', ''),
                        'width': data.get('width', 0),
                        'height': data.get('height', 0),
                    }).encode('utf-8')

                    # Retrieve the image bytes from the fetched data.
                    frame_bytes = data.get('frame_bytes', b'')

                    if frame_bytes:
                        # Update the last ID for this key to avoid re-fetching.
                        message_bytes = header + DELIMITER + frame_bytes

                        # Send the combined bytes to the WebSocket client.
                        await websocket.send_bytes(message_bytes)

            # Sleep briefly to avoid excessive polling of Redis.
            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        print('WebSocket disconnected')
    except Exception as e:
        # On any unexpected error, log it and close the WebSocket.
        print(f"Error in /ws/labels/{label}: {e}")
        await websocket.close()
    finally:
        print('WebSocket connection closed')


@router.websocket('/ws/stream/{label}/{key}')
async def websocket_stream(
    websocket: WebSocket,
    label: str,
    key: str,
    rds=Depends(get_redis_pool_ws),
) -> None:
    """
    Provides a WebSocket endpoint for streaming frames associated with
    a specific label and key.

    Args:
        websocket (WebSocket): The active WebSocket connection.
        label (str): The label grouping the frames (e.g., site or camera).
        key (str): A specific key within that label identifying a stream.
        rds (Any): The Redis connection pool dependency for WebSocket usage.
    """
    # Accept the WebSocket connection.
    await websocket.accept()

    # Construct the Redis key used to store frames (frame + metadata).
    redis_key = f"stream_frame:{Utils.encode(label)}|{Utils.encode(key)}"

    # Keep track of the last retrieved frame ID.
    last_id = '0'

    try:
        while True:
            # Receive a message from the client in text format.
            msg = await websocket.receive_text()
            data: dict[str, Any] = json.loads(msg)

            # Check the "action" field in the received JSON.
            if data.get('action') == 'ping':
                # Respond with a pong for keepalive.
                await websocket.send_text(json.dumps({'action': 'pong'}))

            elif data.get('action') == 'pull':
                # Fetch the latest frame from Redis for this particular key.
                frame_data = await fetch_latest_frame_for_key(
                    rds,
                    redis_key,
                    last_id,
                )
                if frame_data:
                    # Update `last_id` so subsequent pulls only get new frames.
                    last_id = frame_data['id']

                    # Construct a JSON header for the frame metadata.
                    header = json.dumps({
                        'id': frame_data['id'],
                        'warnings': frame_data['warnings'],
                        'cone_polygons': frame_data['cone_polygons'],
                        'pole_polygons': frame_data['pole_polygons'],
                        'detection_items': frame_data['detection_items'],
                        'width': frame_data['width'],
                        'height': frame_data['height'],
                    }).encode('utf-8')

                    # Retrieve the raw frame bytes.
                    raw_bytes = frame_data['frame_bytes'] or b''
                    await websocket.send_bytes(header + DELIMITER + raw_bytes)

            else:
                # If the client sends an unknown action, inform them.
                await websocket.send_text(
                    json.dumps({'error': 'unknown action'}),
                )

    except WebSocketDisconnect:
        print('WebSocket disconnected')
    except Exception as e:
        print(f"Error in /ws/stream/{label}/{key}: {e}")
        await websocket.close()
    finally:
        print('WebSocket connection closed')
