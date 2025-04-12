from __future__ import annotations

import base64
import datetime
from asyncio.log import logger
from pathlib import Path

from fastapi import APIRouter
from fastapi import Body
from fastapi import Depends
from fastapi import HTTPException
from fastapi_jwt import JwtAuthorizationCredentials
from werkzeug.utils import secure_filename

from examples.auth.cache import custom_rate_limiter
from examples.auth.jwt_config import jwt_access
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

#: APIRouter for object detection endpoints.
detection_router = APIRouter()

#: APIRouter for model management endpoints.
model_management_router = APIRouter()

#: A manager for loading and retrieving detection models.
model_loader = DetectionModelManager()


@detection_router.post('/detect')
async def detect(
    detection_request: DetectionRequest = Depends(DetectionRequest.as_form),
    credentials: JwtAuthorizationCredentials = Depends(jwt_access),
    remaining_requests: int = Depends(custom_rate_limiter),
) -> list[list[float | int]]:
    """
    Perform object detection on an uploaded image using a specified model.

    Args:
        image (UploadFile):
            The uploaded image file to be processed.
        model (str, optional):
            The name of the model to use for detection. Defaults to 'yolo11n'.
        credentials (JwtAuthorizationCredentials):
            JWT credentials to verify the user. Injected by FastAPI.
        remaining_requests (int):
            The remaining rate limit for the user. Injected by FastAPI.

    Returns:
        list[list[float | int]]: A list of detection results, where each
            sub-list may include bounding box coordinates, confidence scores,
            classification IDs, etc.

    Raises:
        HTTPException: If the specified model is not found (404).
    """
    # Log user info and remaining requests
    print(f"Authenticated user: {credentials.subject}")
    print(f"Remaining requests: {remaining_requests}")

    # Read image data and convert to a format compatible with the model
    data: bytes = await detection_request.image.read()
    img = await convert_to_image(data)

    # Retrieve the specified model
    model_instance = model_loader.get_model(detection_request.model)
    if model_instance is None:
        raise HTTPException(status_code=404, detail='Model not found')

    # Perform detection
    result = await get_prediction_result(img, model_instance)

    # Compile and post-process detection data
    datas = compile_detection_data(result)
    datas = await process_labels(datas)
    return datas


@model_management_router.post('/model_file_update')
async def model_file_update(
    data: ModelFileUpdate = Depends(ModelFileUpdate.as_form),
    credentials: JwtAuthorizationCredentials = Depends(jwt_access),
) -> dict[str, str]:
    """
    Upload and update a model file on the server.

    Args:
        data (ModelFileUpdate):
            Contains the model identifier and the uploaded file.
        credentials (JwtAuthorizationCredentials):
            JWT credentials used to verify the user's role.

    Returns:
        dict[str, str]: A confirmation message on successful update.

    Raises:
        HTTPException: 403 if the user lacks the required role.
        HTTPException: 400 if there is a validation error.
        HTTPException: 500 if there is an I/O error.
    """
    role = credentials.subject.get('role', '')
    if role not in ['admin', 'model_manage']:
        raise HTTPException(
            status_code=403,
            detail="Permission denied. Need 'admin' or 'model_manage' role.",
        )

    # Prepare temporary path
    temp_path = Path('/tmp/default_model_name')

    try:
        # Obtain a secure filename
        filename = data.file.filename or 'default_model_name'
        secure_file_name = secure_filename(filename)
        temp_dir = Path('/tmp')
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / secure_file_name

        # Write the uploaded file to a temporary location
        with temp_path.open('wb') as temp_file:
            temp_file.write(await data.file.read())

        # Update the actual model file
        await update_model_file(data.model, temp_path)

        logger.info(f"Model {data.model} updated successfully.")
        return {'message': f'Model {data.model} updated successfully.'}

    except ValueError as e:
        logger.error(f"Model update validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except OSError as e:
        logger.error(f"Model update I/O error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Remove the temporary file
        if temp_path.exists():
            temp_path.unlink()


@model_management_router.post('/get_new_model')
async def get_new_model(
    update_request: UpdateModelRequest = Body(...),
    credentials: JwtAuthorizationCredentials = Depends(jwt_access),
) -> dict[str, str]:
    """
    Check for a newer model file on the server and return it if available.

    Args:
        update_request (UpdateModelRequest):
            Contains the model identifier and the user's last update time.
        credentials (JwtAuthorizationCredentials):
            JWT credentials to verify user roles.

    Returns:
        dict[str, str]: A dictionary containing a status message and,
            if updated, the base64-encoded model file.

    Raises:
        HTTPException: 403 if the user does not have permission.
        HTTPException: 400 if there is a validation error.
        HTTPException: 500 if retrieval fails for other reasons.
    """
    role = credentials.subject.get('role', '')
    if role in ['guest']:
        raise HTTPException(
            status_code=403,
            detail="Permission denied. Need 'admin' or 'model_manage' role.",
        )

    try:
        model = update_request.model
        last_update_time = update_request.last_update_time
        user_last_update = datetime.datetime.fromisoformat(last_update_time)

        # Check if a newer model file is available
        model_file_content = await get_new_model_file(model, user_last_update)
        if model_file_content:
            logger.info(f"Newer model file for {model} retrieved.")
            return {
                'message': f"Model {model} is updated.",
                'model_file': base64.b64encode(model_file_content).decode(),
            }

        logger.info(f"No update required for model {model}.")
        return {'message': f"Model {model} is up to date."}

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving model: {e}")
        raise HTTPException(
            status_code=500,
            detail='Failed to retrieve model.',
        )
