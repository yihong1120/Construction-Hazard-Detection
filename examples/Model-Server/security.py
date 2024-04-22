import secrets

def update_secret_key(app):
    app.config['JWT_SECRET_KEY'] = secrets.token_urlsafe(16)