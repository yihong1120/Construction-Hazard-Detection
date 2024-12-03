from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator

from fastapi import Depends
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker

from examples.user_management.user_operation import add_user
from examples.user_management.user_operation import delete_user
from examples.user_management.user_operation import set_user_active_status
from examples.user_management.user_operation import update_password
from examples.user_management.user_operation import update_username

# Define the database URL for connection
DATABASE_URL: str = os.getenv(
    'DATABASE_URL',
) or 'mysql+asyncmy://username:password@mysql/construction_hazard_detection'

# Initialise the FastAPI application
app: FastAPI = FastAPI()

# Set up the SQLAlchemy asynchronous engine and session
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession,
)


@app.on_event('startup')
async def create_default_user():
    """
    Create a default user with username 'user' and password 'password'
    when the application starts.
    """
    async with async_session() as db:
        result = await add_user('user', 'password', 'user', db)
        if result['success']:
            logger.info('Default user created successfully.')
        else:
            logger.warning(
                f"Failed to create default user: {result['message']}",
            )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


async def get_db() -> AsyncGenerator[AsyncSession]:
    """
    Dependency for obtaining a database session.

    Yields:
        AsyncSession: The database session for interacting with the database.
    """
    async with async_session() as session:
        yield session


# Define the Pydantic models for request validation
class UserCreate(BaseModel):
    """
    Represents the data required to create a user.

    Attributes:
        username (str): The username for the new user.
        password (str): The password for the new user.
        role (str): The role assigned to the user (defaults to 'user').
    """
    username: str
    password: str
    role: str = 'user'


class UpdateUsername(BaseModel):
    """
    Represents the data required to update a user's username.

    Attributes:
        old_username (str): The current username of the user.
        new_username (str): The new username to assign to the user.
    """
    old_username: str
    new_username: str


class UpdatePassword(BaseModel):
    """
    Represents the data required to update a user's password.

    Attributes:
        username (str): The username of the user whose password
            will be updated.
        new_password (str): The new password to assign to the user.
    """
    username: str
    new_password: str


@app.post('/add_user')
async def add_user_route(
    user: UserCreate, db: AsyncSession = Depends(get_db),
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
    result = await add_user(user.username, user.password, user.role, db)
    if result['success']:
        logger.info(result['message'])
        return {'message': 'User added successfully.'}
    logger.error(f"Add User Error: {result['message']}")
    raise HTTPException(
        status_code=400 if result['error'] == 'IntegrityError' else 500,
        detail='Failed to add user.',
    )


@app.delete('/delete_user/{username}')
async def delete_user_route(
    username: str, db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Endpoint to delete a user.

    Args:
        username (str): The username of the user to delete.
        db (AsyncSession): The database session dependency.

    Returns:
        dict: A success message if the operation is successful.

    Raises:
        HTTPException: If user deletion fails.
    """
    result = await delete_user(username, db)
    if result['success']:
        logger.info(result['message'])
        return {'message': 'User deleted successfully.'}
    logger.error(f"Delete User Error: {result['message']}")
    raise HTTPException(
        status_code=404 if result['error'] == 'NotFound' else 500,
        detail='Failed to delete user.',
    )


@app.put('/update_username')
async def update_username_route(
    update_data: UpdateUsername, db: AsyncSession = Depends(get_db),
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


@app.put('/update_password')
async def update_password_route(
    update_data: UpdatePassword, db: AsyncSession = Depends(get_db),
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


@app.put('/set_user_active_status/{username}')
async def set_user_active_status_route(
    username: str, is_active: bool, db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Endpoint to update a user's active status.

    Args:
        username (str): The username of the user to update.
        is_active (bool): The new active status of the user.
        db (AsyncSession): The database session dependency.

    Returns:
        dict: A success message if the operation is successful.

    Raises:
        HTTPException: If the status update fails.
    """
    result = await set_user_active_status(username, is_active, db)
    if result['success']:
        logger.info(result['message'])
        return {'message': 'User active status updated successfully.'}
    logger.error(f"Set Active Status Error: {result['message']}")
    raise HTTPException(
        status_code=404 if result['error'] == 'NotFound' else 500,
        detail='Failed to update active status.',
    )


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
