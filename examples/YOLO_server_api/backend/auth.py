# examples/YOLO_server_api/backend/auth.py
from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import HTTPException
from pydantic import BaseModel
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from .cache import get_user_data
from .cache import set_user_data
from .models import User


class UserLogin(BaseModel):
    username: str
    password: str


async def create_token_logic(
    user: UserLogin,
    db: AsyncSession,
    redis_pool: Redis,
    jwt_access,
    max_jti: int = 2,
) -> dict[str, Any]:
    """
    Authenticates a user and generates an access token for them.

    Args:
        user (UserLogin): The user login details.
        db (AsyncSession): The database session.
        redis_pool (Redis): The Redis connection pool.
        jwt_access: The JWT access token generator.
        max_jti (int): The maximum number of JTI values to store.

    Returns:
        dict[str, Any]: {'access_token': str, 'role': str, 'username': str}
    """
    print(f"db_user.__dict__ = {user.__dict__}")

    # Check if the user is in the cache
    user_data = await get_user_data(redis_pool, user.username)

    # If the user is not in the cache, check the database
    if not user_data:
        result = await db.execute(
            select(User).where(User.username == user.username),
        )
        db_user = result.scalar()
        if not db_user:
            raise HTTPException(401, detail='Wrong username or password')

        user_data = {
            'db_user': {
                'id': db_user.id,
                'username': db_user.username,
                'role': db_user.role,
                'is_active': db_user.is_active,
            },
            'jti_list': [],
        }
    else:
        db_user = user_data['db_user']

    # Check if the user is active and has a valid role
    result = await db.execute(
        select(User).where(User.username == user.username),
    )
    real_db_user = result.scalar()
    if (
        not real_db_user
        or not await real_db_user.check_password(user.password)
    ):
        raise HTTPException(401, detail='Wrong username or password')
    if not real_db_user.is_active:
        raise HTTPException(403, detail='User account is inactive')
    if real_db_user.role not in ['admin', 'model_manager', 'user', 'guest']:
        raise HTTPException(403, detail='User does not have the required role')

    # Generate a new JTI value and update the JTI list
    new_jti = str(uuid4())
    jti_list = user_data['jti_list']
    if len(jti_list) >= max_jti:
        jti_list.pop(0)
    jti_list.append(new_jti)

    # Update the user data with the new JTI value
    user_data['db_user']['role'] = real_db_user.role
    user_data['db_user']['is_active'] = real_db_user.is_active

    # Store the updated user data in the cache
    await set_user_data(redis_pool, user.username, user_data)

    # Generate a new access token for the user
    access_token = jwt_access.create_access_token(
        subject={
            'username': real_db_user.username,
            'role': real_db_user.role,
            'jti': new_jti,
        },
    )

    return {
        'access_token': access_token,
        'role': real_db_user.role,
        'username': real_db_user.username,
    }
