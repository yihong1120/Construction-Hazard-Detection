from __future__ import annotations

import secrets

from fastapi import FastAPI


def update_secret_key(app: FastAPI) -> None:
    """
    Updates the JWT secret key for the FastAPI application.

    This function generates a new, secure JWT secret key and assigns it to
    the application's state. The new key is a URL-safe token of 16 bytes.

    Args:
        app (FastAPI): The FastAPI application instance to update.
    """
    # Generate a new URL-safe token of 16 bytes
    app.state.jwt_secret_key = secrets.token_urlsafe(16)
