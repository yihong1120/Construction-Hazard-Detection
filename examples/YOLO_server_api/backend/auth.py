from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi_jwt import JwtAccessBearer
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from .cache import user_cache
from .config import Settings
from .models import get_db
from .models import User

auth_router = APIRouter()
jwt_access = JwtAccessBearer(secret_key=Settings().authjwt_secret_key)


class UserLogin(BaseModel):
    username: str
    password: str


@auth_router.post('/api/token')
async def create_token(user: UserLogin, db: AsyncSession = Depends(get_db)):
    """
    Authenticates a user and generates an access token for them.

    Args:
        user (UserLogin): The user login details.
        db (AsyncSession): The database session.

    Returns:
        dict: { 'access_token': '...', 'role': '...', 'username': '...' }
    """
    print(f"db_user.__dict__ = {user.__dict__}")

    # Check if the user is in the cache
    db_user = user_cache.get(user.username)
    if not db_user:
        result = await db.execute(
            select(User).where(User.username == user.username),
        )
        db_user = result.scalar()
        if db_user:
            # Add the user to the cache
            user_cache[user.username] = db_user

    # Check if user and password correct
    if not db_user or not await db_user.check_password(user.password):
        raise HTTPException(
            status_code=401, detail='Wrong username or password',
        )

    # Check active
    if not db_user.is_active:
        raise HTTPException(status_code=403, detail='User account is inactive')

    # Check role
    if db_user.role not in ['admin', 'model_manager', 'user', 'guest']:
        raise HTTPException(
            status_code=403, detail='User does not have the required role',
        )

    # Generate access token
    access_token = jwt_access.create_access_token(
        subject={'username': user.username, 'role': db_user.role},
    )

    return {
        'access_token': access_token,
        'role': db_user.role,
        'username': db_user.username,
    }
