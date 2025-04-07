from __future__ import annotations

from fastapi_jwt import JwtAccessBearer

from examples.auth.config import Settings

# Instantiate the Settings object to access the JWT secret key
settings: Settings = Settings()

# Create a JwtAccessBearer instance for access tokens
jwt_access: JwtAccessBearer = JwtAccessBearer(
    secret_key=settings.authjwt_secret_key,
)

# Create a JwtAccessBearer instance for refresh tokens
jwt_refresh: JwtAccessBearer = JwtAccessBearer(
    secret_key=settings.authjwt_secret_key,
)
