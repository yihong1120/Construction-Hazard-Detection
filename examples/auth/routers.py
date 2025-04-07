from __future__ import annotations

import datetime
from typing import Any
from uuid import uuid4

import jwt
from fastapi import APIRouter
from fastapi import Depends
from fastapi import Header
from fastapi import HTTPException
from fastapi_jwt import JwtAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from examples.auth.auth import verify_refresh_token
from examples.auth.cache import get_user_data
from examples.auth.cache import set_user_data
from examples.auth.config import Settings
from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import jwt_refresh
from examples.auth.models import User
from examples.auth.redis_pool import get_redis_pool
from examples.auth.schemas import DeleteUser
from examples.auth.schemas import LogoutRequest
from examples.auth.schemas import RefreshRequest
from examples.auth.schemas import SetUserActiveStatus
from examples.auth.schemas import UpdatePassword
from examples.auth.schemas import UpdateUsername
from examples.auth.schemas import UserCreate
from examples.auth.schemas import UserLogin
from examples.auth.user_operation import add_user
from examples.auth.user_operation import delete_user
from examples.auth.user_operation import set_user_active_status
from examples.auth.user_operation import update_password
from examples.auth.user_operation import update_username

auth_router = APIRouter()
user_management_router = APIRouter()

SECRET_KEY: str = Settings().authjwt_secret_key
ALGORITHM: str = 'HS256'


@auth_router.post('/login')
async def login_endpoint(
    user: UserLogin,
    db: AsyncSession = Depends(get_db),
    redis_pool=Depends(get_redis_pool),
) -> dict[str, Any]:
    """
    Logs a user in and generates both access and refresh tokens.

    Args:
        user (UserLogin):
            A Pydantic model containing `username` and `password`.
        db (AsyncSession):
            The asynchronous SQLAlchemy session dependency.
        redis_pool:
            The Redis client connection dependency.

    Returns:
        dict[str, Any]: A dictionary containing the newly issued access token,
            refresh token, user role, username, and user ID.

    Raises:
        HTTPException: If the username/password is incorrect, user
            account is inactive, or the user lacks a valid role.
    """
    # Step 1: Fetch user data from Redis or database
    user_data: dict[str, Any] | None = await (
        get_user_data(redis_pool, user.username)
    )
    if not user_data:
        # Retrieve from database if Redis has no record
        result = await db.execute(
            select(User).where(User.username == user.username),
        )
        db_user: User | None = result.scalar()

        if not db_user:
            raise HTTPException(
                status_code=401,
                detail='Wrong username or password',
            )

        user_data = {
            'db_user': {
                'id': db_user.id,
                'username': db_user.username,
                'role': db_user.role,
                'is_active': db_user.is_active,
            },
            'jti_list': [],
            'refresh_tokens': [],
        }
    else:
        # Ensure our data structure in Redis has the needed fields
        user_data.setdefault('jti_list', [])
        user_data.setdefault('refresh_tokens', [])

    # Step 2: Validate the provided password
    result = await db.execute(
        select(User).where(User.username == user.username),
    )
    real_db_user: User | None = result.scalar()

    if (
        not real_db_user
        or not await real_db_user.check_password(user.password)
    ):
        raise HTTPException(
            status_code=401,
            detail='Wrong username or password',
        )

    # Step 3: Check account status and role
    if not real_db_user.is_active:
        raise HTTPException(
            status_code=403,
            detail='User account is inactive',
        )

    if real_db_user.role not in ['admin', 'model_manager', 'user', 'guest']:
        raise HTTPException(
            status_code=403,
            detail='User does not have the required role',
        )

    # Step 4: Create new JTI and issue tokens
    new_jti: str = str(uuid4())
    user_data['jti_list'].append(new_jti)

    access_token: str = jwt_access.create_access_token(
        subject={
            'username': real_db_user.username,
            'role': real_db_user.role,
            'jti': new_jti,
        },
        expires_delta=datetime.timedelta(minutes=60),
    )
    refresh_token: str = jwt_refresh.create_access_token(
        subject={'username': real_db_user.username},
        expires_delta=datetime.timedelta(days=30),
    )

    user_data['refresh_tokens'].append(refresh_token)

    # Step 5: Update Redis
    await set_user_data(redis_pool, user.username, user_data)

    return {
        'access_token': access_token,
        'refresh_token': refresh_token,
        'role': real_db_user.role,
        'username': real_db_user.username,
        'user_id': real_db_user.id,
    }


@auth_router.post('/logout')
async def logout_endpoint(
    request: LogoutRequest,
    authorization: str = Header(None),
    redis_pool=Depends(get_redis_pool),
) -> dict[str, Any]:
    """
    Logs a user out by revoking both access and refresh tokens.

    Args:
        request (LogoutRequest):
            Pydantic model containing the `refresh_token` for local revocation.
        authorization (str):
            The `Authorization` header (expected "Bearer <token>").
        redis_pool:
            The Redis client connection dependency.

    Returns:
        dict[str, Any]: A message describing the logout outcome.
    """
    # If there's no auth header, proceed with local-only logout
    if not authorization:
        return {
            'message': (
                'No access token provided, '
                'but local logout can proceed.'
            ),
        }

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != 'bearer':
        return {'message': 'Invalid Authorization header format.'}

    access_token: str = parts[1]

    # Attempt to decode; if it fails, we still permit local logout
    try:
        payload = jwt.decode(
            access_token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            options={'verify_exp': False},
        )
    except jwt.PyJWTError:
        return {'message': 'Invalid token, but local logout can proceed.'}

    username: str | None = payload.get('username')
    token_jti: str | None = payload.get('jti')
    if not username:
        return {'message': 'No username in token payload, logout local only.'}

    user_data: dict[str, Any] | None = await (
        get_user_data(redis_pool, username)
    )
    if not user_data:
        return {'message': 'User data not found in Redis, local logout only.'}

    # Remove the refresh token if it exists
    refresh_tokens: list[str] = user_data.get('refresh_tokens', [])
    if request.refresh_token in refresh_tokens:
        refresh_tokens.remove(request.refresh_token)
        user_data['refresh_tokens'] = refresh_tokens

    # Remove the jti from the user's list
    jti_list: list[str] = user_data.get('jti_list', [])
    if token_jti in jti_list:
        jti_list.remove(token_jti)
        user_data['jti_list'] = jti_list

    await set_user_data(redis_pool, username, user_data)
    return {
        'message': (
            'Logged out successfully '
            '(both access & refresh token revoked).'
        ),
    }


@auth_router.post('/refresh')
async def refresh_token_endpoint(
    request: RefreshRequest,
    db: AsyncSession = Depends(get_db),
    redis_pool=Depends(get_redis_pool),
    authorization: str = Header(None),
) -> dict[str, Any]:
    """
    Refreshes an access token using a valid refresh token.

    Args:
        request (RefreshRequest):
            Pydantic model containing the old refresh token.
        db (AsyncSession):
            Asynchronous SQLAlchemy session dependency.
        redis_pool:
            Redis client connection dependency.
        authorization (str):
            Optional `Authorization` header, typically "Bearer <token>"
            if provided. Not strictly required for this flow.

    Returns:
        dict[str, Any]: A dictionary containing the new access token,
            new refresh token, and a success message.

    Raises:
        HTTPException:
            If the refresh token is missing, invalid, or not recognised.
    """
    if not request.refresh_token:
        raise HTTPException(status_code=401, detail='Refresh token is missing')

    # Validate and decode the refresh token
    payload: dict[str, Any] = await (
        verify_refresh_token(request.refresh_token, redis_pool)
    )
    username: str = payload['subject']['username']

    user_data: dict[str, Any] | None = await (
        get_user_data(redis_pool, username)
    )
    if not user_data:
        raise HTTPException(status_code=401, detail='No user data in Redis')

    # Remove the old refresh token from the user's data
    refresh_tokens: list[str] = user_data.get('refresh_tokens', [])
    if request.refresh_token in refresh_tokens:
        refresh_tokens.remove(request.refresh_token)
    else:
        raise HTTPException(
            status_code=401, detail='Refresh token not recognized',
        )
    user_data['refresh_tokens'] = refresh_tokens

    # Issue new JTI and tokens
    new_jti: str = str(uuid4())
    new_access_token: str = jwt_access.create_access_token(
        subject={
            'username': username,
            'role': user_data['db_user']['role'],
            'jti': new_jti,
        },
        expires_delta=datetime.timedelta(minutes=60),
    )
    new_refresh_token: str = jwt_refresh.create_access_token(
        subject={'username': username},
        expires_delta=datetime.timedelta(days=30),
    )

    user_data.setdefault('jti_list', []).append(new_jti)
    user_data.setdefault('refresh_tokens', []).append(new_refresh_token)
    await set_user_data(redis_pool, username, user_data)

    return {
        'access_token': new_access_token,
        'refresh_token': new_refresh_token,
        'message': 'Token refreshed successfully.',
    }


# ---------------------------------------------
#       User Management (Requires Admin)
# ---------------------------------------------

@user_management_router.post('/add_user')
async def add_user_route(
    user: UserCreate,
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Depends(jwt_access),
) -> dict[str, Any]:
    """
    Creates a new user in the system.

    Args:
        user (UserCreate):
            Contains username, password, and role.
        db (AsyncSession):
            Asynchronous SQLAlchemy session dependency.
        credentials (JwtAuthorizationCredentials):
            The JWT credentials used for authorisation.

    Returns:
        dict[str, Any]: A message indicating success if the user was added.

    Raises:
        HTTPException:
            If the caller does not have the 'admin' role,
            or user creation fails (400 or 500).
    """
    if credentials.subject['role'] != 'admin':
        raise HTTPException(status_code=403, detail='Admin role required.')

    result: dict[str, Any] = await (
        add_user(user.username, user.password, user.role, db)
    )
    if result['success']:
        return {'message': 'User added successfully.'}
    raise HTTPException(
        status_code=400 if result['error'] == 'IntegrityError' else 500,
        detail='Failed to add user.',
    )


@user_management_router.post('/delete_user')
async def delete_user_route(
    user: DeleteUser,
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Depends(jwt_access),
) -> dict[str, Any]:
    """
    Deletes an existing user.

    Args:
        user (DeleteUser):
            Pydantic model containing the username to delete.
        db (AsyncSession):
            Asynchronous SQLAlchemy session dependency.
        credentials (JwtAuthorizationCredentials):
            The JWT credentials used for authorisation.

    Returns:
        dict[str, Any]: A success message if the user is deleted.

    Raises:
        HTTPException:
            If the caller is not 'admin',
            or if the user is not found or deletion fails.
    """
    if credentials.subject['role'] != 'admin':
        raise HTTPException(status_code=403, detail='Admin role required.')

    result: dict[str, Any] = await delete_user(user.username, db)
    if result['success']:
        return {'message': 'User deleted successfully.'}
    raise HTTPException(
        status_code=404 if result['error'] == 'NotFound' else 500,
        detail='Failed to delete user.',
    )


@user_management_router.put('/update_username')
async def update_username_route(
    update_data: UpdateUsername,
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Depends(jwt_access),
) -> dict[str, Any]:
    """
    Updates a user's username.

    Args:
        update_data (UpdateUsername):
            Contains `old_username` and `new_username`.
        db (AsyncSession):
            Asynchronous SQLAlchemy session dependency.
        credentials (JwtAuthorizationCredentials):
            JWT credentials for authorisation.

    Returns:
        dict[str, Any]: A success message if the operation succeeds.

    Raises:
        HTTPException:
            If the caller lacks 'admin' role,
            or if update conflicts or the old username is not found.
    """
    if credentials.subject['role'] != 'admin':
        raise HTTPException(status_code=403, detail='Admin role required.')

    result: dict[str, Any] = await update_username(
        update_data.old_username,
        update_data.new_username,
        db,
    )
    if result['success']:
        return {'message': 'Username updated successfully.'}
    raise HTTPException(
        status_code=400 if result['error'] == 'IntegrityError' else 404,
        detail='Failed to update username.',
    )


@user_management_router.put('/update_password')
async def update_password_route(
    update_data: UpdatePassword,
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Depends(jwt_access),
) -> dict[str, Any]:
    """
    Updates a user's password.

    Args:
        update_data (UpdatePassword):
            Contains the target `username` and new password.
        db (AsyncSession):
            Asynchronous SQLAlchemy session dependency.
        credentials (JwtAuthorizationCredentials):
            JWT credentials for authorisation.

    Returns:
        dict[str, Any]: A success message if the password is updated.

    Raises:
        HTTPException:
            If the caller is not 'admin' or the user is not found
            or if another error occurs (500).
    """
    if credentials.subject['role'] != 'admin':
        raise HTTPException(status_code=403, detail='Admin role required.')

    result: dict[str, Any] = await update_password(
        update_data.username,
        update_data.new_password,
        db,
    )
    if result['success']:
        return {'message': 'Password updated successfully.'}
    raise HTTPException(
        status_code=404 if result['error'] == 'NotFound' else 500,
        detail='Failed to update password.',
    )


@user_management_router.put('/set_user_active_status')
async def set_user_active_status_route(
    user_status: SetUserActiveStatus,
    db: AsyncSession = Depends(get_db),
    credentials: JwtAuthorizationCredentials = Depends(jwt_access),
) -> dict[str, Any]:
    """
    Activates or deactivates a user account.

    Args:
        user_status (SetUserActiveStatus):
            Contains the `username` and desired `is_active` state.
        db (AsyncSession):
            Asynchronous SQLAlchemy session dependency.
        credentials (JwtAuthorizationCredentials):
            JWT credentials for authorisation.

    Returns:
        dict[str, Any]: A message indicating the outcome if successful.

    Raises:
        HTTPException:
            If the caller is not 'admin' or the user is not found,
            or if the operation fails for other reasons.
    """
    if credentials.subject['role'] != 'admin':
        raise HTTPException(status_code=403, detail='Admin role required.')

    result: dict[str, Any] = await set_user_active_status(
        user_status.username,
        user_status.is_active,
        db,
    )
    if result['success']:
        return {'message': 'User active status updated successfully.'}
    raise HTTPException(
        status_code=404 if result['error'] == 'NotFound' else 500,
        detail='Failed to update active status.',
    )
