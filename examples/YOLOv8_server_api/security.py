from __future__ import annotations

import secrets

from flask import Flask


def update_secret_key(app: Flask) -> None:
    """
    Updates the JWT secret key in the application configuration.

    This function generates a new, secure JWT secret key and updates the
    Flask application's config with this token. It helps protect against
    security breaches by regularly rotating the secret key.

    Args:
        app (Flask): The Flask app instance to update the JWT secret key.
    """
    # Securely generate a new, random JWT secret key using the secrets library
    app.config['JWT_SECRET_KEY'] = secrets.token_urlsafe(16)
