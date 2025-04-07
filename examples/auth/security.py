from __future__ import annotations

import secrets

from fastapi import FastAPI


def update_secret_key(app: FastAPI) -> None:
    """
    Update the FastAPI application's JWT secret key.

    Args:
        app (FastAPI): The FastAPI application instance where
            the secret key will be stored.
    """
    # Generate a new secret key and store it in the FastAPI app state
    app.state.jwt_secret_key = secrets.token_urlsafe(16)
