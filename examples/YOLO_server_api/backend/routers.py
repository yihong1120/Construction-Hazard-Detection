from __future__ import annotations

import base64
import datetime
from asyncio.log import logger
from pathlib import Path

from fastapi import APIRouter
from fastapi import Body
from fastapi import Depends
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi_jwt import JwtAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from werkzeug.utils import secure_filename

from .cache import custom_rate_limiter
from .cache import jwt_access
from .detection import compile_detection_data
from .detection import convert_to_image
from .detection import get_prediction_result
from .detection import process_labels
from .model_files import get_new_model_file
from .model_files import update_model_file
from .models import DetectionModelManager
from .models import get_db
from .user_operation import add_user
from .user_operation import delete_user
from .user_operation import set_user_active_status
from .user_operation import update_password
from .user_operation import update_username

# Define routers for different functionalities
detection_router = APIRouter()
user_management_router = APIRouter()
model_management_router = APIRouter()

# Load detection models
model_loader = DetectionModelManager()


# Detection APIs
class DetectionRequest(BaseModel):
    """
    Represents the input format for the object detection endpoint.
    """
    image: UploadFile
    model: str


@detection_router.post('/api/detect')
async def detect(
    image: UploadFile = File(...),
    model: str = 'yolo11n',
    credentials: JwtAuthorizationCredentials = Depends(jwt_access),
    remaining_requests: int = Depends(custom_rate_limiter),
) -> list[list[float | int]]:
    """
    Processes the uploaded image to detect objects based on
    the specified model.

    Args:
        image (UploadFile): The uploaded image file for object detection.
        model (str): The model name to be used for detection
             (default is 'yolo11n').
        credentials (JwtAuthorizationCredentials): The JWT credentials for
            authorisation.
        remaining_requests (int): The remaining number of allowed requests.

    Returns:
        List[List[float | int]]: A list containing detection data including
        bounding boxes, confidence scores, and labels.
    """
    print(f"Authenticated user: {credentials.subject}")
    print(f"Remaining requests: {remaining_requests}")

    # Retrieve image data and convert to OpenCV format
    data: bytes = await image.read()
    img = await convert_to_image(data)

    # Load the specified model for detection
    model_instance = model_loader.get_model(model)

    if model_instance is None:
        raise HTTPException(status_code=404, detail='Model not found')

    # Perform object detection on the uploaded image
    result = await get_prediction_result(img, model_instance)

    # Compile and process the detection results
    datas = compile_detection_data(result)
    datas = await process_labels(datas)
    return datas


# User Management APIs
class UserCreate(BaseModel):
    """
    Represents the data required to create a user.
    """
    username: str
    password: str
    role: str = 'user'


@user_management_router.post('/api/add_user')
async def add_user_route(
    user: UserCreate,
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Depends(jwt_access),
) -> dict:
    """
    Endpoint to add a new user to the system.

    Args:
        user (UserCreate): The data required to create the user.
        db (AsyncSession): The database session dependency.

    Returns:
        dict: A success message if the operation is successful.

    Raises:
        HTTPException: If user creation fails.
    """
    print(f"UserCreate: {user}")

    if credentials.subject['role'] not in ['admin']:
        raise HTTPException(
            status_code=400,
            detail=(
                'Invalid role. Must be one of: '
                'admin, model_manage, user, guest.'
            ),
        )
    result = await add_user(user.username, user.password, user.role, db)
    if result['success']:
        logger.info(result['message'])
        return {'message': 'User added successfully.'}
    logger.error(f"Add User Error: {result['message']}")
    raise HTTPException(
        status_code=400 if result['error'] == 'IntegrityError' else 500,
        detail='Failed to add user.',
    )


class DeleteUser(BaseModel):
    """
    Represents the data required to delete a user.
    """
    username: str


@user_management_router.post('/api/delete_user')
async def delete_user_route(
    user: DeleteUser,
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Depends(jwt_access),
) -> dict:
    """
    Endpoint to delete a user.

    Args:
        user (DeleteUser): The data required to delete the user.
        db (AsyncSession): The database session dependency.
        credentials (JwtAuthorizationCredentials): The JWT credentials
        for authorisation.

    Returns:
        dict: A success message if the operation is successful.

    Raises:
        HTTPException: If user deletion fails.
    """
    if credentials.subject['role'] not in ['admin']:
        raise HTTPException(
            status_code=403,
            detail='Permission denied. Admin role required.',
        )

    result = await delete_user(user.username, db)
    if result['success']:
        logger.info(result['message'])
        return {'message': 'User deleted successfully.'}

    logger.error(f"Delete User Error: {result['message']}")
    raise HTTPException(
        status_code=404 if result['error'] == 'NotFound' else 500,
        detail='Failed to delete user.',
    )


class UpdateUsername(BaseModel):
    """
    Represents the data required to update a user's username.
    """
    old_username: str
    new_username: str


@user_management_router.put('/api/update_username')
async def update_username_route(
    update_data: UpdateUsername,
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Depends(jwt_access),
) -> dict:
    """
    Endpoint to update a user's username.

    Args:
        update_data (UpdateUsername): The data required for the update.
        db (AsyncSession): The database session dependency.

    Returns:
        dict: A success message if the operation is successful.

    Raises:
        HTTPException: If the username update fails.
    """
    if credentials.subject['role'] not in ['admin']:
        raise HTTPException(
            status_code=400,
            detail=(
                'Invalid role. Must be one of: '
                'admin, model_manage, user, guest.'
            ),
        )

    result = await update_username(
        update_data.old_username, update_data.new_username, db,
    )
    if result['success']:
        logger.info(result['message'])
        return {'message': 'Username updated successfully.'}
    logger.error(f"Update Username Error: {result['message']}")
    raise HTTPException(
        status_code=400 if result['error'] == 'IntegrityError' else 404,
        detail='Failed to update username.',

    )


class UpdatePassword(BaseModel):
    """
    Represents the data required to update a user's password.
    """
    username: str
    new_password: str
    role: str = 'user'


@user_management_router.put('/api/update_password')
async def update_password_route(
    update_data: UpdatePassword,
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Depends(jwt_access),
) -> dict:
    """
    Endpoint to update a user's password.

    Args:
        update_data (UpdatePassword): The data required for the update.
        db (AsyncSession): The database session dependency.

    Returns:
        dict: A success message if the operation is successful.

    Raises:
        HTTPException: If the password update fails.
    """
    if credentials.subject['role'] not in ['admin']:
        raise HTTPException(
            status_code=400,
            detail=(
                'Invalid role. Must be one of: '
                'admin, model_manage, user, guest.'
            ),
        )

    result = await update_password(
        update_data.username, update_data.new_password, db,
    )
    if result['success']:
        logger.info(result['message'])
        return {'message': 'Password updated successfully.'}
    logger.error(f"Update Password Error: {result['message']}")
    raise HTTPException(
        status_code=404 if result['error'] == 'NotFound' else 500,
        detail='Failed to update password.',
    )


class SetUserActiveStatus(BaseModel):
    """
    Represents the data required to update a user's active status.

    Attributes:
        username (str): The username of the user.
        is_active (bool): The new active status to set.
    """
    username: str
    is_active: bool


@user_management_router.put('/api/set_user_active_status')
async def set_user_active_status_route(
    user_status: SetUserActiveStatus,  # 使用請求正文接收數據
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Depends(jwt_access),
) -> dict:
    """
    Endpoint to update a user's active status.

    Args:
        user_status (SetUserActiveStatus): The data containing username
            and active status to update.
        db (AsyncSession): The database session dependency.
        credentials (JwtAuthorizationCredentials): The JWT credentials
        for authorisation.

    Returns:
        dict: A success message if the operation is successful.

    Raises:
        HTTPException: If the status update fails or permissions are invalid.
    """
    if credentials.subject['role'] != 'admin':
        raise HTTPException(
            status_code=403,
            detail='Admin privileges are required to update user status.',
        )

    result = await set_user_active_status(
        username=user_status.username,
        is_active=user_status.is_active,
        db=db,
    )
    if result['success']:
        logger.info(result['message'])
        return {'message': 'User active status updated successfully.'}
    logger.error(f"Set Active Status Error: {result['message']}")
    raise HTTPException(
        status_code=404 if result['error'] == 'NotFound' else 500,
        detail='Failed to update active status.',
    )


# Model Management APIs
class ModelFileUpdate(BaseModel):
    """
    Represents the data required to update a model file.
    """
    model: str
    file: UploadFile

    @classmethod
    def as_form(
        cls,
        model: str = Form(...),
        file: UploadFile = File(...),
    ) -> ModelFileUpdate:
        """
        Enables the use of ModelFileUpdate as a FastAPI dependency
        with form inputs.

        Args:
            model (str): The name of the model.
            file (UploadFile): The file to upload.

        Returns:
            ModelFileUpdate: An instance of ModelFileUpdate populated
            with the inputs.
        """
        return cls(model=model, file=file)


@model_management_router.post('/api/model_file_update')
async def model_file_update(
    data: ModelFileUpdate = Depends(ModelFileUpdate.as_form),
    credentials: JwtAuthorizationCredentials = Depends(jwt_access),
) -> dict:
    """
    Endpoint to update a model file.

    Args:
        data (ModelFileUpdate): The data required to update the model file.
        credentials (JwtAuthorizationCredentials): The JWT credentials
        for authorisation.

    Returns:
        dict: A success message if the operation is successful.
    """
    if credentials.subject['role'] not in ['admin', 'model_manage']:
        raise HTTPException(
            status_code=403,
            detail=(
                "Permission denied. Role must be 'admin' "
                "or 'model_manage'."
            ),
        )

    try:
        # Ensure the filename is secure
        filename = data.file.filename or 'default_model_name'
        secure_file_name = secure_filename(filename)
        temp_dir = Path('/tmp')
        temp_path = temp_dir / secure_file_name

        # Check if the path is within the intended directory
        if not temp_path.resolve().parent == temp_dir:
            raise HTTPException(status_code=400, detail='Invalid file path.')

        # Write file to disk
        with temp_path.open('wb') as temp_file:
            temp_file.write(await data.file.read())

        # Update the model file
        await update_model_file(data.model, temp_path)

        # Log and return success
        logger.info(f"Model {data.model} updated successfully.")
        return {'message': f'Model {data.model} updated successfully.'}
    except ValueError as e:
        logger.error(f"Model update validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except OSError as e:
        logger.error(f"Model update I/O error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if temp_path.exists():
            temp_path.unlink()


class UpdateModelRequest(BaseModel):
    """
    Represents the data required to retrieve a new model file.

    Attributes:
        model (str): The name of the model.
        last_update_time (str): The last update time of the model file
        in ISO format.
    """
    model: str
    last_update_time: str


@model_management_router.post('/api/get_new_model')
async def get_new_model(
    update_request: UpdateModelRequest = Body(...),
) -> dict:
    """
    Endpoint to retrieve the new model file for a specific model.

    Args:
        update_request (UpdateModelRequest): The request data containing
        model name and last update time.

    Returns:
        dict: The new model file if available
        or a message indicating no update.
    """
    try:
        # Extract data from the request
        model = update_request.model
        last_update_time = update_request.last_update_time

        # Parse the last update time provided by the user
        user_last_update = datetime.datetime.fromisoformat(last_update_time)

        # Check for a new model file
        model_file_content = await get_new_model_file(model, user_last_update)
        if model_file_content:
            logger.info(f"Newer model file for {model} retrieved.")
            return {
                'message': f"Model {model} is updated.",
                'model_file': base64.b64encode(model_file_content).decode(),
            }

        # No update required
        logger.info(f"No update required for model {model}.")
        return {'message': f"Model {model} is up to date."}
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving model: {e}")
        raise HTTPException(
            status_code=500, detail='Failed to retrieve model.',
        )
