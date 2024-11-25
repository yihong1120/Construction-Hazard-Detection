from __future__ import annotations

from typing import AsyncGenerator

from fastapi import Depends
from fastapi import FastAPI
from fastapi import HTTPException
from markupsafe import escape
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker
import logging

from examples.user_management.user_operation import add_user
from examples.user_management.user_operation import delete_user
from examples.user_management.user_operation import set_user_active_status
from examples.user_management.user_operation import update_password
from examples.user_management.user_operation import update_username

# Define the database URL for connection
DATABASE_URL = 'mysql+asyncmy://username:password@localhost/database_name'

# Initialise the FastAPI application
app = FastAPI()

# Set up the SQLAlchemy asynchronous engine and session
engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def get_db() -> AsyncGenerator[AsyncSession]:
    """
    Provides a dependency for obtaining a database session.

    Yields:
        AsyncSession: The database session for performing operations.
    """
    async with async_session() as session:
        yield session


# Define the Pydantic models for request validation
class UserCreate(BaseModel):
    """
    Represents the payload for creating a new user.

    Attributes:
        username (str): The username for the new user.
        password (str): The password for the new user.
        role (str): The role assigned to the user (default is 'user').
    """
    username: str
    password: str
    role: str = 'user'


class UpdateUsername(BaseModel):
    """
    Represents the payload for updating a user's username.

    Attributes:
        old_username (str): The current username.
        new_username (str): The new username to be assigned.
    """
    old_username: str
    new_username: str


class UpdatePassword(BaseModel):
    """
    Represents the payload for updating a user's password.

    Attributes:
        username (str): The username of the user whose password will be updated.
        new_password (str): The new password to be set.
    """
    username: str
    new_password: str


@app.post('/add_user')
async def add_user_route(user: UserCreate, db: AsyncSession = Depends(get_db)) -> dict:
    """
    Endpoint to add a new user to the database.

    Args:
        user (UserCreate): The data required to create a user.
        db (AsyncSession): The database session dependency.

    Returns:
        dict: A message indicating the result of the operation.

    Raises:
        HTTPException: If the user creation fails.
    """
    try:
        result = await add_user(user.username, user.password, user.role, db)
        if 'successfully' in result:
            return {'message': 'User added successfully.'}
        raise HTTPException(status_code=400, detail="Failed to add user.")
    except Exception as e:
        logger.error(f"Error adding user: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.delete('/delete_user/{username}')
async def delete_user_route(username: str, db: AsyncSession = Depends(get_db)) -> dict:
    """
    Endpoint to delete a user from the database.

    Args:
        username (str): The username of the user to be deleted.
        db (AsyncSession): The database session dependency.

    Returns:
        dict: A message indicating the result of the operation.

    Raises:
        HTTPException: If the user deletion fails.
    """
    try:
        username = escape(username)
        result = await delete_user(username, db)
        if 'successfully' in result:
            return {'message': 'User deleted successfully.'}
        raise HTTPException(status_code=404, detail="User not found.")
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.put('/update_username')
async def update_username_route(
    update_data: UpdateUsername, db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Endpoint to update a user's username.

    Args:
        update_data (UpdateUsername): The data required to update the username.
        db (AsyncSession): The database session dependency.

    Returns:
        dict: A message indicating the result of the operation.

    Raises:
        HTTPException: If the username update fails.
    """
    try:
        old_username = escape(update_data.old_username)
        new_username = escape(update_data.new_username)
        result = await update_username(old_username, new_username, db)
        if 'successfully' in result:
            return {'message': 'Username updated successfully.'}
        raise HTTPException(status_code=400, detail="Failed to update username.")
    except Exception as e:
        logger.error(f"Error updating username: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.put('/update_password')
async def update_password_route(
    update_data: UpdatePassword, db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Endpoint to update a user's password.

    Args:
        update_data (UpdatePassword): The data required to update the password.
        db (AsyncSession): The database session dependency.

    Returns:
        dict: A message indicating the result of the operation.

    Raises:
        HTTPException: If the password update fails.
    """
    try:
        username = escape(update_data.username)
        new_password = escape(update_data.new_password)
        result = await update_password(username, new_password, db)
        if 'successfully' in result:
            return {'message': 'Password updated successfully.'}
        raise HTTPException(status_code=400, detail="Failed to update password.")
    except Exception as e:
        logger.error(f"Error updating password: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


@app.put('/set_user_active_status/{username}')
async def set_user_active_status_route(
    username: str, is_active: bool, db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Endpoint to update a user's active status.

    Args:
        username (str): The username of the user.
        is_active (bool): The new active status.
        db (AsyncSession): The database session dependency.

    Returns:
        dict: A message indicating the result of the operation.

    Raises:
        HTTPException: If the active status update fails.
    """
    try:
        result = await set_user_active_status(username, is_active, db)
        if 'successfully' in result:
            return {'message': 'Active status updated successfully.'}
        raise HTTPException(status_code=400, detail="Failed to update active status.")
    except Exception as e:
        logger.error(f"Error updating active status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)