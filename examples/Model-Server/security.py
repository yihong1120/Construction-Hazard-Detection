from flask import Flask
import secrets

def update_secret_key(app: Flask) -> None:
    """
    Updates the JWT secret key in the application configuration.

    This function generates a new, secure token to be used as the JWT secret key and updates the
    Flask application's configuration with this new token. It's a crucial security measure to
    help protect against potential security breaches by regularly rotating the secret key.

    Args:
        app (Flask): The Flask application instance whose JWT secret key is to be updated.
    """
    # Securely generate a new, random JWT secret key using the secrets library
    app.config['JWT_SECRET_KEY'] = secrets.token_urlsafe(16)