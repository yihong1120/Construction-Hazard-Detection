from __future__ import annotations

import asyncio
import json
from typing import cast

import jwt
import redis
from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi_jwt import JwtAuthorizationCredentials
from fastapi_limiter.depends import RateLimiter
from jwt import InvalidTokenError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from examples.auth.cache import get_user_data
from examples.auth.config import Settings
from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.models import User
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

# Global application settings instance
settings: Settings = Settings()

# Create a FastAPI router instance to encapsulate endpoints
router: APIRouter = APIRouter()

# Rate limiters to control endpoint access frequency
rate_limiter_index: RateLimiter = RateLimiter(times=60, seconds=60)
rate_limiter_label: RateLimiter = RateLimiter(times=600, seconds=60)


@router.get(
    '/labels',
    response_model=LabelListResponse,
    dependencies=[Depends(rate_limiter_index)],
)
async def get_labels_route(
    credentials: JwtAuthorizationCredentials = Depends(jwt_access),
    db: AsyncSession = Depends(get_db),
    rds: redis.Redis = Depends(get_redis_pool),
) -> LabelListResponse:
    """
    Fetch a list of labels accessible to the authenticated user.

    Args:
        credentials: JWT authorisation credentials containing user information.
        db: Asynchronous database session for user queries.
        rds: Redis connection pool for label retrieval.

    Returns:
        LabelListResponse containing filtered list of accessible labels.

    Raises:
        HTTPException:
        When token is invalid, user not found, or Redis operation fails.
    """
    try:
        # Extract username from JWT subject
        username: str | None = credentials.subject.get('username')
        if not username:
            raise HTTPException(
                status_code=401, detail='Invalid token: no subject',
            )

        # Query database for user with preloaded site relationships
        stmt = (
            select(User)
            .where(User.username == username)
            # Eagerly load site relationships
            .options(selectinload(User.sites))
        )
        result = await db.execute(stmt)
        user: User | None = result.scalars().first()
        if not user:
            raise HTTPException(status_code=401, detail='Invalid user')

        # Extract user role for permission checking
        user_role: str = user.role

        # Build list of site names the user has access to
        user_site_names: list[str] = [site.name for site in user.sites]
        print(f"User {username} has access to sites: {user_site_names}")

        # Fetch all available labels from Redis
        all_labels: list[str] = await scan_for_labels(rds)
        print(f"All labels in Redis: {all_labels}")

        # Filter labels based on user permissions
        if user_role == 'admin':
            # Admins can access all labels
            filtered_labels: list[str] = all_labels
        else:
            # Regular users only get labels for their accessible sites
            filtered_labels = [
                lbl for lbl in all_labels if lbl in user_site_names
            ]

        return LabelListResponse(labels=filtered_labels)

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
    rds: redis.Redis = Depends(get_redis_pool),
) -> FramePostResponse:
    """
    Store a video frame with associated metadata in Redis.

    Args:
        label: Site or camera label to group frames under.
        key: Unique identifier for the specific video feed or stream.
        file: Uploaded image file containing the video frame.
        warnings_json: JSON string containing warning messages (optional).
        cone_polygons_json:
            JSON string containing traffic cone polygon data (optional).
        pole_polygons_json:
            JSON string containing utility pole polygon data (optional).
        detection_items_json:
            JSON string containing detected object data (optional).
        width: Frame width in pixels (defaults to 0).
        height: Frame height in pixels (defaults to 0).
        rds: Redis connection pool for frame storage.

    Returns:
        FramePostResponse indicating storage success or failure.

    Raises:
        HTTPException: When frame storage in Redis fails.
    """
    try:
        # Read uploaded file content asynchronously
        frame_bytes: bytes = await file.read()

        # Store frame and associated metadata in Redis
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
    rds: redis.Redis = Depends(get_redis_pool_ws),
) -> None:
    """Provide WebSocket endpoint for streaming video frames by label.

    This endpoint establishes a persistent WebSocket connection for streaming
    video frames associated with a specific label. It continuously polls Redis
    for new frames and broadcasts them to connected clients.

    Args:
        websocket: Active WebSocket connection for real-time communication.
        label: Label identifier under which frames are grouped.
        rds: Redis connection pool optimised for WebSocket usage.

    Returns:
        None: Function manages WebSocket lifecycle internally.

    Note:
        JWT authentication is currently commented out but can be enabled
        by uncommenting the relevant authentication blocks.
    """
    # Accept incoming WebSocket connection
    await websocket.accept()

    # Note: JWT authentication is disabled for this endpoint
    # To enable authentication, uncomment the following block:
    #
    # # Extract and validate JWT token from Authorization header
    # auth: str | None = websocket.headers.get('authorization')
    # if not auth or not auth.lower().startswith('bearer '):
    #     # Close connection if Authorization header is missing or malformed
    #     await websocket.close(code=1008)
    #     return

    # # Extract token from "Bearer <token>" format
    # token: str = auth.split(' ', 1)[1]

    # # Verify JWT token using authentication module configuration
    # try:
    #     payload: dict = jwt.decode(
    #         token, settings.authjwt_secret_key,
    #         algorithms=[settings.ALGORITHM],
    #     )
    # except InvalidTokenError:
    #     # Close connection if JWT token is invalid
    #     await websocket.close(code=1008)
    #     return

    # # Ensure JWT payload contains required data
    # if not payload:
    #     await websocket.close(code=1008)
    #     return

    # # Extract username and JTI (JWT ID) from token payload
    # subject_data = payload.get('subject', {})
    # username: str | None = (
    #     subject_data.get('username') if subject_data else None
    # ) or payload.get('username')
    # jti: str | None = (
    #     subject_data.get('jti') if subject_data else None
    # ) or payload.get('jti')

    # # Validate that both username and JTI are present
    # if not username or not jti:
    #     await websocket.close(code=1008)
    #     return

    # # Verify JTI against cached user data in Redis
    # user_data: dict[str, str | list[str]] | None = await get_user_data(
    #     cast(redis.asyncio.Redis, rds), username
    # )

    # # Ensure user data exists and contains the JTI list
    # # If JTI is not found in user's active token list, close connection
    # if (
    #     not user_data
    #     or 'jti_list' not in user_data
    #     or jti not in user_data['jti_list']
    # ):
    #     # Close connection if JTI is not found in user's active token list
    #     await websocket.close(code=1008)
    #     return

    try:
        # Fetch all stream keys associated with the provided label
        keys: list[str] = await get_keys_for_label(rds, label)
        if not keys:
            # Notify client that no streams exist for this label
            await websocket.send_json({
                'error': f"No keys found for label '{label}'",
            })
            await websocket.close()
            return

        # Track last processed message ID for each stream to prevent duplicates
        last_ids: dict[str, str] = {k: '0' for k in keys}

        # Main streaming loop
        while True:
            # Poll Redis for new frames across all streams
            updated_data: list[dict[str, str]] = (
                await fetch_latest_frames(rds, last_ids)
            )

            # If new frames are available, send them to the client
            if updated_data:
                for data in updated_data:
                    # Construct JSON metadata header for frame
                    header: bytes = json.dumps({
                        'key': data.get('key', ''),
                        'warnings': data.get('warnings', ''),
                        'cone_polygons': data.get('cone_polygons', ''),
                        'pole_polygons': data.get('pole_polygons', ''),
                        'detection_items': data.get('detection_items', ''),
                        'width': data.get('width', 0),
                        'height': data.get('height', 0),
                    }).encode('utf-8')

                    # Extract frame bytes from Redis data
                    frame_bytes_raw = data.get('frame_bytes', b'')
                    # Cast to bytes, ensuring type safety
                    frame_bytes: bytes = cast(
                        bytes, frame_bytes_raw,
                    ) if frame_bytes_raw else b''

                    if frame_bytes:
                        # Combine header and frame data with delimiter
                        message_bytes: bytes = header + DELIMITER + frame_bytes

                        # Send binary frame data to WebSocket client
                        await websocket.send_bytes(message_bytes)

            # Brief pause to prevent excessive Redis polling
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
    rds: redis.Redis = Depends(get_redis_pool_ws),
) -> None:
    """
    WebSocket endpoint for streaming video frames with real-time inference.

    Args:
        websocket: Active WebSocket connection for bidirectional communication.
        label: Label grouping the frames (e.g., site or camera identifier).
        key: Specific key within the label identifying a unique stream.
        rds: Redis connection pool optimised for WebSocket operations.

    Returns:
        None: Function manages WebSocket lifecycle and client interactions.

    Note:
        Supports 'ping' (keepalive) and 'pull' (fetch frame) actions.
        JWT authentication is currently disabled but can be enabled.
    """
    # Accept incoming WebSocket connection
    await websocket.accept()

    # Note: JWT authentication is disabled for this endpoint
    # To enable authentication, uncomment the following block:
    #
    # # Extract and validate JWT token from Authorization header
    # auth: str | None = websocket.headers.get('authorization')
    # if not auth or not auth.lower().startswith('bearer '):
    #     # Close connection if Authorization header is missing or malformed
    #     await websocket.close(code=1008)
    #     return

    # # Extract token from "Bearer <token>" format
    # token: str = auth.split(' ', 1)[1]

    # # Verify JWT token using authentication module configuration
    # try:
    #     payload: dict = jwt.decode(
    #         token, settings.authjwt_secret_key,
    #         algorithms=[settings.ALGORITHM],
    #     )
    # except InvalidTokenError:
    #     # Close connection if JWT token is invalid
    #     await websocket.close(code=1008)
    #     return

    # # Ensure JWT payload contains required data
    # if not payload:
    #     await websocket.close(code=1008)
    #     return

    # # Extract username and JTI (JWT ID) from token payload
    # subject_data = payload.get('subject', {})
    # username: str | None = (
    #     subject_data.get('username') if subject_data else None
    # ) or payload.get('username')
    # jti: str | None = (
    #     subject_data.get('jti') if subject_data else None
    # ) or payload.get('jti')

    # # Validate that both username and JTI are present
    # if not username or not jti:
    #     await websocket.close(code=1008)
    #     return

    # # Verify JTI against cached user data in Redis
    # user_data: dict[str, str | list[str]] | None = await get_user_data(
    #     cast(redis.asyncio.Redis, rds), username
    # )

    # # Ensure user data exists and contains the JTI list
    # # If JTI is not found in user's active token list, close connection
    # if (
    #     not user_data
    #     or 'jti_list' not in user_data
    #     or jti not in user_data['jti_list']
    # ):
    #     # Close connection if JTI is not found in user's active token list
    #     await websocket.close(code=1008)
    #     return

    # Construct Redis key for frame storage using encoded label and key
    redis_key: str = f"stream_frame:{Utils.encode(label)}|{Utils.encode(key)}"

    # Track the last retrieved frame ID to fetch only new frames
    last_id: str = '0'

    try:
        # Main client interaction loop
        while True:
            # Receive client message in JSON text format
            msg: str = await websocket.receive_text()
            data: dict[str, str | int] = json.loads(msg)

            # Handle different client actions
            if data.get('action') == 'ping':
                # Respond to keepalive ping with pong
                await websocket.send_text(json.dumps({'action': 'pong'}))

            elif data.get('action') == 'pull':
                # Fetch latest frame data from Redis for this stream
                frame_data: dict[str, str] | None = (
                    await fetch_latest_frame_for_key(
                        rds,
                        redis_key,
                        last_id,
                    )
                )
                if frame_data:
                    # Update last_id to avoid re-fetching same frame
                    last_id = frame_data['id']

                    # Build JSON header containing frame metadata
                    header: bytes = json.dumps({
                        'id': frame_data['id'],
                        'warnings': frame_data['warnings'],
                        'cone_polygons': frame_data['cone_polygons'],
                        'pole_polygons': frame_data['pole_polygons'],
                        'detection_items': frame_data['detection_items'],
                        'width': frame_data['width'],
                        'height': frame_data['height'],
                    }).encode('utf-8')

                    # Extract raw frame bytes from Redis data
                    raw_bytes_data = frame_data.get('frame_bytes') or b''
                    raw_bytes: bytes = cast(
                        bytes, raw_bytes_data,
                    ) if raw_bytes_data else b''

                    # Send combined header and frame data to client
                    await websocket.send_bytes(header + DELIMITER + raw_bytes)

            else:
                # Handle unknown actions with error response
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


@router.websocket('/ws/frames')
async def websocket_frames(
    websocket: WebSocket,
    rds: redis.Redis = Depends(get_redis_pool_ws),
) -> None:
    """WebSocket endpoint for uploading video frames with metadata.

    This endpoint provides a persistent WebSocket connection for uploading
    video frames along with their associated metadata. JWT authentication is
    required via the Authorization header. Frames are stored in Redis for
    later retrieval and processing.

    The expected message format is:
        JSON header + DELIMITER + image bytes

    Args:
        websocket: Active WebSocket connection for frame uploads.
        rds: Redis connection pool for storing frame data.

    Returns:
        None: Function manages WebSocket lifecycle and frame processing.

    Raises:
        WebSocketDisconnect: When the client disconnects from the WebSocket.
        InvalidTokenError: When JWT token validation fails.

    Note:
        Each message must contain a JSON header with frame metadata followed
        by a delimiter and the raw image bytes. JWT token must be provided
        in the Authorization header as 'Bearer <token>'.
    """
    # Accept incoming WebSocket connection
    await websocket.accept()

    # Extract and validate JWT token from Authorization header
    auth: str | None = websocket.headers.get('authorization')
    if not auth or not auth.lower().startswith('bearer '):
        # Close connection if Authorization header is missing or malformed
        await websocket.close(code=1008)
        return

    # Extract token from "Bearer <token>" format
    token: str = auth.split(' ', 1)[1]

    # Verify JWT token using authentication module configuration
    try:
        payload: dict = jwt.decode(
            token, settings.authjwt_secret_key,
            algorithms=[settings.ALGORITHM],
        )
    except InvalidTokenError:
        # Close connection if JWT token is invalid
        await websocket.close(code=1008)
        return

    # Ensure JWT payload contains required data
    if not payload:
        await websocket.close(code=1008)
        return

    # Extract username and JTI (JWT ID) from token payload
    subject_data = payload.get('subject', {})
    username: str | None = (
        subject_data.get('username') if subject_data else None
    ) or payload.get('username')
    jti: str | None = (
        subject_data.get('jti') if subject_data else None
    ) or payload.get('jti')

    # Validate that both username and JTI are present
    if not username or not jti:
        await websocket.close(code=1008)
        return

    # Verify JTI against cached user data in Redis
    user_data: dict[str, str | list[str]] | None = await get_user_data(
        cast(redis.asyncio.Redis, rds), username,
    )

    # Ensure user data exists and contains the JTI list
    # If JTI is not found in user's active token list, close connection
    if (
        not user_data
        or 'jti_list' not in user_data
        or jti not in user_data['jti_list']
    ):
        # Close connection if JTI is not found in user's active token list
        await websocket.close(code=1008)
        return

    try:
        # Main frame processing loop
        while True:
            # Receive frame data in binary format:
            # JSON header + DELIMITER + image bytes
            data: bytes = await websocket.receive_bytes()
            try:
                # Split received data into header and frame components
                header_bytes: bytes
                frame_bytes: bytes
                header_bytes, frame_bytes = data.split(DELIMITER, 1)

                # Parse JSON header containing frame metadata
                header: dict[str, str | int] = json.loads(
                    header_bytes.decode('utf-8'),
                )

                # Extract frame metadata from parsed header
                label_value = header.get('label', '')
                label: str = str(label_value) if label_value else ''
                key_value = header.get('key', '')
                key: str = str(key_value) if key_value else ''
                warnings_json_value = header.get('warnings_json', '')
                warnings_json: str = str(
                    warnings_json_value,
                ) if warnings_json_value else ''
                cone_polygons_json_value = header.get('cone_polygons_json', '')
                cone_polygons_json: str = str(
                    cone_polygons_json_value,
                ) if cone_polygons_json_value else ''
                pole_polygons_json_value = header.get('pole_polygons_json', '')
                pole_polygons_json: str = str(
                    pole_polygons_json_value,
                ) if pole_polygons_json_value else ''
                detection_items_json_value = header.get(
                    'detection_items_json', '',
                )
                detection_items_json: str = str(
                    detection_items_json_value,
                ) if detection_items_json_value else ''
                width_value = header.get('width', 0)
                width: int = int(width_value) if width_value else 0
                height_value = header.get('height', 0)
                height: int = int(height_value) if height_value else 0

                # Store frame and metadata in Redis for later retrieval
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

                # Send success confirmation to client
                await websocket.send_json({
                    'status': 'ok',
                    'message': 'Frame stored successfully.',
                })
            except Exception as e:
                # Send error response if frame processing fails
                await websocket.send_json({
                    'status': 'error',
                    'message': f'Failed to store frame: {str(e)}',
                })
    except WebSocketDisconnect:
        # Log WebSocket disconnection event
        print('WebSocket client disconnected from frames endpoint')
    except Exception as e:
        # Handle general exceptions and close WebSocket gracefully
        print(f"Error in /ws/frames endpoint: {e}")
        await websocket.close()
    finally:
        # Ensure proper cleanup when WebSocket connection terminates
        print('WebSocket connection closed for frames endpoint')
