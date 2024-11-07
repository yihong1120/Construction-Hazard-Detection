from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi_jwt import JwtAccessBearer
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .cache import user_cache
from .config import Settings
from .models import get_db
from .models import User

auth_router = APIRouter()
jwt_access = JwtAccessBearer(secret_key=Settings().authjwt_secret_key)


class UserLogin(BaseModel):
    username: str
    password: str


@auth_router.post('/token')
def create_token(user: UserLogin, db: Session = Depends(get_db)):
    print(f"db_user.__dict__ = {user.__dict__}")
    db_user = user_cache.get(user.username)
    print(db_user)
    if not db_user:
        db_user = db.query(User).filter(User.username == user.username).first()
        if db_user:
            user_cache[user.username] = db_user

    if not db_user or not db_user.check_password(user.password):
        raise HTTPException(
            status_code=401, detail='Wrong username or password',
        )

    if not db_user.is_active:
        raise HTTPException(
            status_code=403, detail='User account is inactive',
        )

    if db_user.role not in ['admin', 'model_manager', 'user', 'guest']:
        raise HTTPException(
            status_code=403, detail='User does not have the required role',
        )

    # access_token = jwt_access.create_access_token(subject=user.username)
    access_token = jwt_access.create_access_token(
        subject={'username': user.username, 'role': db_user.role},
    )

    return {'access_token': access_token}
