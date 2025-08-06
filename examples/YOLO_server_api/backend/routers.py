from __future__ import annotations

import asyncio
import base64
import datetime
import json
import time
from asyncio.log import logger
from pathlib import Path
from typing import Any
from typing import cast

import aiofiles
import jwt
import redis
from fastapi import APIRouter
from fastapi import Body
from fastapi import Depends
from fastapi import HTTPException
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi.responses import ORJSONResponse
from fastapi_jwt import JwtAuthorizationCredentials
from jwt import InvalidTokenError
from werkzeug.utils import secure_filename

from examples.auth.cache import custom_rate_limiter
from examples.auth.cache import get_user_data
from examples.auth.config import Settings
from examples.auth.jwt_config import jwt_access
from examples.auth.redis_pool import get_redis_pool_ws
from examples.YOLO_server_api.backend.detection import compile_detection_data
from examples.YOLO_server_api.backend.detection import convert_to_image
from examples.YOLO_server_api.backend.detection import get_prediction_result
from examples.YOLO_server_api.backend.detection import process_labels
from examples.YOLO_server_api.backend.model_files import get_new_model_file
from examples.YOLO_server_api.backend.model_files import update_model_file
from examples.YOLO_server_api.backend.models import DetectionModelManager
from examples.YOLO_server_api.backend.schemas import DetectionRequest
from examples.YOLO_server_api.backend.schemas import ModelFileUpdate
from examples.YOLO_server_api.backend.schemas import UpdateModelRequest

# Router instances for API endpoints
detection_router: APIRouter = APIRouter()
model_management_router: APIRouter = APIRouter()

# Global model manager instance for handling YOLO models
model_loader: DetectionModelManager = DetectionModelManager()

# Application settings configuration
settings: Settings = Settings()

# Concurrency control semaphores to prevent GPU OOM errors
# Limit simultaneous inference operations to prevent memory overflow
INFERENCE_SEMAPHORE: asyncio.Semaphore = asyncio.Semaphore(4)
# WebSocket connections can handle more concurrent operations
WS_INFERENCE_SEMAPHORE: asyncio.Semaphore = asyncio.Semaphore(8)


def _is_websocket_connected(websocket: WebSocket) -> bool:
    """Check if a WebSocket connection is still active and valid.

    This function verifies the WebSocket connection state by checking multiple
    attributes to ensure the connection is properly built and operational.

    Args:
        websocket: The WebSocket connection to check.

    Returns:
        True if the WebSocket is connected and operational, False otherwise.

    Note:
        This function performs defensive checks to handle cases where
        attributes might not exist or the connection state is invalid.
    """
    try:
        # Check if the WebSocket has a client_state attribute
        if not hasattr(websocket, 'client_state'):
            return False

        # Verify the connection state (1 = CONNECTED)
        if websocket.client_state.value != 1:
            return False

        # Ensure the client object exists and is valid
        if not hasattr(websocket, 'client') or not websocket.client:
            return False

        return True
    except Exception:
        # Return False for any unexpected errors during state checking
        return False


async def _safe_websocket_send_json(
    websocket: WebSocket,
    data: dict[str, Any],
    client_info: str = '',
) -> bool:
    """Safely send JSON data through a WebSocket connection.

    This function provides a safe wrapper around WebSocket JSON sending,
    with connection state validation and error handling.

    Args:
        websocket: The WebSocket connection to send data through.
        data: The dictionary data to send as JSON.
        client_info: Optional client identification string for logging.

    Returns:
        True if the data was sent successfully, False otherwise.

    Note:
        This function checks connection state before attempting to send
        and handles exceptions gracefully with optional logging.
    """
    # Verify the WebSocket connection is still active
    if not _is_websocket_connected(websocket):
        if client_info:
            print(
                f"[WebSocket] {client_info}: "
                'Connection closed, skipping JSON send',
            )
        return False

    try:
        # Attempt to send the JSON data
        await websocket.send_json(data)
        return True
    except Exception as e:
        # Log the error if client info is provided
        if client_info:
            print(f"[WebSocket] {client_info}: Failed to send JSON: {e}")
        return False


async def _safe_websocket_receive_bytes(
    websocket: WebSocket,
    client_info: str = '',
) -> bytes | None:
    """Safely receive binary data from a WebSocket connection.

    This function provides a safe wrapper around WebSocket byte receiving,
    with connection state validation and error handling.

    Args:
        websocket: The WebSocket connection to receive data from.
        client_info: Optional client identification string for logging.

    Returns:
        The received bytes data if successful, None if failed or disconnected.

    Note:
        This function checks connection state before attempting to receive
        and handles exceptions gracefully with optional logging.
    """
    # Verify the WebSocket connection is still active
    if not _is_websocket_connected(websocket):
        if client_info:
            print(
                f"[WebSocket] {client_info}: "
                'Connection closed, cannot receive bytes',
            )
        return None

    try:
        # Attempt to receive binary data
        return await websocket.receive_bytes()
    except Exception as e:
        # Log the error if client info is provided
        if client_info:
            print(f"[WebSocket] {client_info}: Failed to receive bytes: {e}")
        return None


@detection_router.post('/detect', response_class=ORJSONResponse)
async def detect(
    detection_request: DetectionRequest = Depends(DetectionRequest.as_form),
    credentials: JwtAuthorizationCredentials = Depends(jwt_access),
    remaining_requests: int = Depends(custom_rate_limiter),
) -> list[list[float | int]]:
    """Process object detection on uploaded images using YOLO models.

    This endpoint accepts image uploads and performs object detection using
    the specified YOLO model. It includes authentication, rate limiting,
    and comprehensive timing metrics for performance monitoring.

    Args:
        detection_request: Form data containing image file and model selection.
        credentials: JWT authentication credentials for user validation.
        remaining_requests: Number of remaining API requests for rate limiting.

    Returns:
        A list of detection results where each detection is represented as
        a list of numerical values [x1, y1, x2, y2, confidence, class_id].

    Raises:
        HTTPException: 404 if the specified model is not found.

    Note:
        This endpoint uses semaphore-based concurrency control to prevent
        GPU memory overflow during simultaneous inference operations.
    """
    # Record the start time for performance monitoring
    start_time: float = time.time()

    # Log authentication and rate limiting information
    print(f"Authenticated user: {credentials.subject}")
    print(f"Remaining requests: {remaining_requests}")

    # Read image data from the uploaded file
    img_bytes: bytes = await detection_request.image.read()
    io_time: float = time.time() - start_time

    # Convert raw bytes to an image object for processing
    img = convert_to_image(img_bytes)

    # Retrieve the requested model instance
    model_instance = model_loader.get_model(detection_request.model)
    if model_instance is None:
        raise HTTPException(status_code=404, detail='Model not found')

    # Perform inference with concurrency control to prevent GPU OOM
    inference_start: float = time.time()
    async with INFERENCE_SEMAPHORE:
        result = await get_prediction_result(img, model_instance)
    inference_time: float = time.time() - inference_start

    # Post-process the detection results
    post_start: float = time.time()
    datas: list[list[float | int]] = compile_detection_data(result)
    datas = await process_labels(datas)
    post_time: float = time.time() - post_start

    # Log comprehensive timing information for performance analysis
    total_time: float = time.time() - start_time
    print(
        f"Detection timing - IO: {io_time:.3f}s, "
        f"Inference: {inference_time:.3f}s, "
        f"Post: {post_time:.3f}s, "
        f"Total: {total_time:.3f}s",
    )

    return datas


@detection_router.websocket('/ws/detect')
async def websocket_detect(
    websocket: WebSocket,
    rds: redis.Redis = Depends(get_redis_pool_ws),
) -> None:
    """Handle real-time object detection via WebSocket connections.

    This WebSocket endpoint provides real-time object detection capabilities
    for streaming image data. It supports JWT authentication via headers or
    query parameters and includes comprehensive error handling and logging.

    Args:
        websocket: The WebSocket connection instance.
        rds: Redis connection for user authentication data validation.

    Returns:
        None. This function handles the WebSocket lifecycle and communication.

    Note:
        The WebSocket expects:
        - JWT authentication via Authorization header
            or 'token' query parameter
        - Model specification via 'x-model-key' header, query parameter,
            or first message
        - Binary image data for processing after initial configuration

        The function maintains the connection until disconnection or error,
        processing frames continuously and returning detection results.
    """
    # Determine client IP address for logging purposes
    client_ip: str = websocket.client.host if websocket.client else 'unknown'
    print(f"[YOLO-WebSocket] New connection from {client_ip}")

    # Accept the incoming WebSocket connection
    await websocket.accept()

    # Extract JWT token from Authorization header or query parameter
    token: str | None = None

    # Priority 1: Try to get token from Authorization header
    # (mobile/desktop apps)
    auth: str | None = websocket.headers.get('authorization')
    if auth and auth.lower().startswith('bearer '):
        token = auth.split(' ', 1)[1]
        print(f"[YOLO-WebSocket] {client_ip}: Token from Authorization header")

    # Priority 2: Try query parameter if header not found (web platforms)
    if not token:
        query_params: dict[str, str] = dict(websocket.query_params)
        token = query_params.get('token')
        if token:
            print(f"[YOLO-WebSocket] {client_ip}: Token from query parameter")

    # Reject connection if no authentication token is provided
    if not token:
        print(
            f"[YOLO-WebSocket] {client_ip}: "
            'No token found in header or query parameter',
        )
        await websocket.close(code=1008, reason='Missing authentication token')
        return

    # Verify JWT token using authentication module configuration
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            settings.authjwt_secret_key,
            algorithms=[settings.ALGORITHM],
        )
    except InvalidTokenError as e:
        print(f"[YOLO-WebSocket] {client_ip}: Invalid JWT token: {e}")
        await websocket.close(code=1008, reason='Invalid token')
        return

    # Ensure JWT payload contains required data
    if not payload:
        print(f"[YOLO-WebSocket] {client_ip}: Empty JWT payload")
        await websocket.close(code=1008, reason='Empty token payload')
        return

    # Extract username and JTI (JWT ID) from token payload
    subject_data: dict[str, Any] = payload.get('subject', {})
    username: str | None = (
        subject_data.get('username') if subject_data else None
    ) or payload.get('username')
    jti: str | None = (
        subject_data.get('jti') if subject_data else None
    ) or payload.get('jti')

    # Validate that both username and JTI are present
    if not username or not jti:
        print(
            f"[YOLO-WebSocket] {client_ip}: Missing username or JTI in token",
        )
        await websocket.close(code=1008, reason='Invalid token data')
        return

    # Verify JTI against cached user data in Redis
    user_data: dict[str, str | list[str]] | None = await get_user_data(
        cast(redis.asyncio.Redis, rds), username,
    )

    # Ensure user data exists and contains the JTI list
    if (
        not user_data
        or 'jti_list' not in user_data
        or jti not in user_data['jti_list']
    ):
        print(
            f"[YOLO-WebSocket] {client_ip}: "
            f"JTI not found in user active tokens for {username}",
        )
        await websocket.close(code=1008, reason='Token not active')
        return

    print(f"[YOLO-WebSocket] {client_ip}: Authenticated as {username}")

    # Get model key from multiple sources
    # (priority: header > query > first message)
    model_key: str | None = None

    # Priority 1: Try to get model key from custom header
    model_key = websocket.headers.get('x-model-key')
    if model_key:
        print(
            f"[YOLO-WebSocket] {client_ip} ({username}): "
            'Model key from header',
        )

    # Priority 2: Try query parameter (backwards compatibility)
    if not model_key:
        query_params = dict(websocket.query_params)
        model_key = query_params.get('model')
        if model_key:
            print(
                f"[YOLO-WebSocket] {client_ip} ({username}): "
                'Model key from query parameter (deprecated)',
            )

    # Priority 3: Expect model key in the first message
    if not model_key:
        print(
            f"[YOLO-WebSocket] {client_ip} ({username}): "
            'Waiting for model key in first message',
        )
        try:
            # Wait for the first message containing model_key as JSON
            first_message: str = await websocket.receive_text()
            config_data: dict[str, Any] = json.loads(first_message)
            model_key = config_data.get('model_key')
            if not model_key:
                print(
                    f"[YOLO-WebSocket] {client_ip} ({username}): "
                    'No model_key found in first message',
                )
                await websocket.close(
                    code=1008,
                    reason='Missing model_key in configuration',
                )
                return
            print(
                f"[YOLO-WebSocket] {client_ip} ({username}): "
                f"Model key from first message",
            )
        except Exception as e:
            print(
                f"[YOLO-WebSocket] {client_ip} ({username}): "
                f"Failed to parse first message: {e}",
            )
            await websocket.close(
                code=1008,
                reason='Invalid configuration message',
            )
            return

    # Validate that the requested model exists
    model_instance = model_loader.get_model(model_key)
    if model_instance is None:
        print(
            f"[YOLO-WebSocket] {client_ip} ({username}): "
            f"Model {model_key} not found",
        )
        # 1003 = Unsupported Data
        await websocket.close(code=1003, reason='Model not found')
        return

    print(
        f"[YOLO-WebSocket] {client_ip} ({username}): Using model {model_key}",
    )

    # Send confirmation message to client
    config_response: dict[str, str] = {
        'status': 'ready',
        'model': model_key,
        'message': 'Model loaded successfully, ready to process images',
    }
    success: bool = await _safe_websocket_send_json(
        websocket,
        config_response,
        f"{client_ip} ({username})",
    )
    if not success:
        print(
            f"[YOLO-WebSocket] {client_ip} ({username}): "
            f"Failed to send configuration response",
        )
        return

    # Initialise frame counter for statistics
    frame_count: int = 0

    # Main processing loop for continuous frame processing
    try:
        while True:
            # Safely receive image data from the WebSocket
            img_bytes: bytes | None = await _safe_websocket_receive_bytes(
                websocket,
                f"{client_ip} ({username})",
            )

            # Check if reception failed (connection closed or error)
            if img_bytes is None:
                print(
                    f"[YOLO-WebSocket] {client_ip} ({username}): "
                    f"Failed to receive image data, connection may be closed",
                )
                break

            frame_count += 1

            try:
                # Convert binary data to image for processing
                img = convert_to_image(img_bytes)

                # Use WebSocket-specific semaphore for concurrency control
                async with WS_INFERENCE_SEMAPHORE:
                    result = await get_prediction_result(img, model_instance)
                    datas = compile_detection_data(result)
                    datas = await process_labels(datas)

                # Safely send results back to client
                success = await _safe_websocket_send_json(
                    websocket,
                    datas,
                    f"{client_ip} ({username})",
                )
                if not success:
                    print(
                        f"[YOLO-WebSocket] {client_ip} ({username}): "
                        f"Failed to send results, stopping",
                    )
                    break

                # Log statistics every 100 frames for monitoring
                if frame_count % 100 == 0:
                    print(
                        f"[YOLO-WebSocket] {client_ip} ({username}): "
                        f"Processed {frame_count} frames",
                    )

            except Exception as e:
                print(
                    f"[YOLO-WebSocket] {client_ip} ({username}): "
                    f"Error processing frame {frame_count}: {e}",
                )
                # Attempt to send error information to client
                await _safe_websocket_send_json(
                    websocket,
                    {'error': f'Frame processing error: {str(e)}'},
                    f"{client_ip} ({username})",
                )

    except WebSocketDisconnect:
        print(
            f'[YOLO-WebSocket] {client_ip} ({username}): '
            f'Client disconnected after {frame_count} frames',
        )
    except Exception as e:
        print(
            f"[YOLO-WebSocket] {client_ip} ({username}): "
            f"Unexpected error: {e}",
        )
        try:
            # 1011 = Internal Error
            await websocket.close(code=1011, reason='Internal server error')
        except Exception:
            # Connection might already be closed
            pass
    finally:
        print(
            f'[YOLO-WebSocket] {client_ip} ({username}): '
            f'Connection closed, total frames processed: {frame_count}',
        )


@model_management_router.post('/model_file_update')
async def model_file_update(
    data: ModelFileUpdate = Depends(ModelFileUpdate.as_form),
    credentials: JwtAuthorizationCredentials = Depends(jwt_access),
) -> dict[str, str]:
    """Update a YOLO model file with administrative privileges.

    This endpoint allows administrators and model managers to upload and
    update YOLO model files. It includes role-based access control,
    secure file handling, and comprehensive error management.

    Args:
        data: Form data containing the model file and model identifier.
        credentials: JWT authentication credentials for authorisation.

    Returns:
        A dictionary containing a success message upon completion.

    Raises:
        HTTPException:
            - 403 if user lacks required permissions
            - 400 for validation errors
            - 500 for I/O errors

    Note:
        This endpoint requires 'admin' or 'model_manage' roles and
        automatically cleans up temporary files after processing.
    """
    # Verify user has the required role for model management
    role: str = credentials.subject.get('role', '')
    if role not in ['admin', 'model_manage']:
        raise HTTPException(
            status_code=403,
            detail="Permission denied. Need 'admin' or 'model_manage' role.",
        )

    # Secure the uploaded filename to prevent directory traversal attacks
    filename: str = data.file.filename or 'default_model_name'
    tmp_path: Path = Path('/tmp') / secure_filename(filename)

    try:
        # Write the uploaded file to a temporary location
        async with aiofiles.open(tmp_path, 'wb') as f:
            await f.write(await data.file.read())

        # Process the model file update
        await update_model_file(data.model, tmp_path)
        logger.info(f"Model {data.model} updated successfully.")
        return {'message': f"Model {data.model} updated successfully."}

    except ValueError as e:
        # Handle validation errors (e.g., invalid model format)
        logger.error(f"Model update validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except OSError as e:
        # Handle I/O errors (e.g., disk space, permissions)
        logger.error(f"Model update I/O error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Ensure temporary file is cleaned up regardless of outcome
        if tmp_path.exists():
            tmp_path.unlink()


@model_management_router.post('/get_new_model')
async def get_new_model(
    update_request: UpdateModelRequest = Body(...),
    credentials: JwtAuthorizationCredentials = Depends(jwt_access),
) -> dict[str, str]:
    """Retrieve updated model files based on timestamp comparison.

    This endpoint checks if a newer version of the requested model is
    available on the server compared to the client's last update time.
    It supports incremental updates to minimise bandwidth usage.

    Args:
        update_request:
            Request body containing model name and last update timestamp.
        credentials:
            JWT authentication credentials for authorisation.

    Returns:
        A dictionary containing either:
        - Update message with base64-encoded model file if newer version exists
        - Up-to-date message if no update is needed

    Raises:
        HTTPException:
            - 403 if user has 'guest' role (insufficient permissions)
            - 400 for invalid timestamp format or validation errors
            - 500 for unexpected server errors

    Note:
        This endpoint excludes 'guest' users but allows other authenticated
        roles to check for and download model updates.
    """
    # Verify user has sufficient privileges (exclude guest users)
    role: str = credentials.subject.get('role', '')
    if role == 'guest':
        raise HTTPException(
            status_code=403,
            detail="Permission denied. Need 'admin' or 'model_manage' role.",
        )

    try:
        # Parse the client's last update timestamp
        user_last_update: datetime.datetime = datetime.datetime.fromisoformat(
            update_request.last_update_time,
        )

        # Check for newer model file on the server
        content: bytes | None = await get_new_model_file(
            update_request.model,
            user_last_update,
        )

        if content:
            # Model update is available - encode and return
            logger.info(
                f"Newer model file for {update_request.model} retrieved.",
            )
            return {
                'message': f"Model {update_request.model} is updated.",
                'model_file': base64.b64encode(content).decode(),
            }

        # Model is already up to date
        return {'message': f"Model {update_request.model} is up to date."}

    except ValueError as e:
        # Handle invalid timestamp format or validation errors
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle unexpected server errors
        logger.error(f"Error retrieving model: {e}")
        raise HTTPException(
            status_code=500,
            detail='Failed to retrieve model.',
        )
