from __future__ import annotations

import base64
import datetime
import time
from asyncio.log import logger
from pathlib import Path

import aiofiles
import redis
from fastapi import APIRouter
from fastapi import Body
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Security
from fastapi import WebSocket
from fastapi.responses import JSONResponse
from fastapi_jwt import JwtAuthorizationCredentials
from werkzeug.utils import secure_filename

from examples.auth.cache import custom_rate_limiter
from examples.auth.config import Settings
from examples.auth.jwt_config import jwt_access
from examples.auth.redis_pool import get_redis_pool_ws
from examples.YOLO_server_api.backend.detection import INFERENCE_SEMAPHORE
from examples.YOLO_server_api.backend.detection import run_detection_from_bytes
from examples.YOLO_server_api.backend.model_files import get_new_model_file
from examples.YOLO_server_api.backend.model_files import update_model_file
from examples.YOLO_server_api.backend.models import DetectionModelManager
from examples.YOLO_server_api.backend.schemas import DetectionRequest
from examples.YOLO_server_api.backend.schemas import ModelFileUpdate
from examples.YOLO_server_api.backend.schemas import UpdateModelRequest
from examples.YOLO_server_api.backend.websocket_handlers import (
    handle_websocket_detect,
)

# Router instances for API endpoints
detection_router: APIRouter = APIRouter()
model_management_router: APIRouter = APIRouter()

# Global model manager instance for handling YOLO models
model_loader: DetectionModelManager = DetectionModelManager()

# Application settings configuration
settings: Settings = Settings()


@detection_router.post('/detect', response_class=JSONResponse)
async def detect(
    detection_request: DetectionRequest = Depends(DetectionRequest.as_form),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
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

    # Retrieve the requested model instance
    model_instance = model_loader.get_model(detection_request.model)
    if model_instance is None:
        raise HTTPException(status_code=404, detail='Model not found')

    # Unified pipeline with concurrency control
    datas, timing = await run_detection_from_bytes(
        img_bytes, model_instance, semaphore=INFERENCE_SEMAPHORE,
    )
    inference_time = timing['inference']
    post_time = timing['post']

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
    """

    """
    await handle_websocket_detect(
        websocket=websocket,
        rds=rds,
        settings=settings,
        model_loader=model_loader,
    )


@model_management_router.post('/model_file_update')
async def model_file_update(
    data: ModelFileUpdate = Depends(ModelFileUpdate.as_form),
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
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
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
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
