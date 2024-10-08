from __future__ import annotations

from flask import Blueprint
from flask import jsonify
from flask import request
from flask import Response
from flask_jwt_extended import create_access_token

from .cache import user_cache
from .models import User

auth_blueprint = Blueprint('auth', __name__)


@auth_blueprint.route('/token', methods=['POST'])
def create_token() -> Response:
    """
    Authenticates a user and generates a JWT token.

    Returns:
        Response: Flask response with JWT token or error message.

    Raises:
        HTTP 401: If the username or password is incorrect.
    """
    # Ensure the request contains JSON data
    if not request.json:
        response = jsonify({'msg': 'Missing JSON in request.'})
        response.status_code = 400
        return response

    # Extract username and password from JSON body of the request
    username = request.json.get('username', None)
    password = request.json.get('password', None)

    # Attempt to fetch the user details from cache
    user = user_cache.get(username)

    # If not in cache, query the database
    if not user:
        user = User.query.filter_by(username=username).first()
        # If user is found, store in cache
        if user:
            user_cache[username] = user

    # Check user and verify password
    if not user or not user.check_password(password):
        # Return error if authentication fails
        response = jsonify({'msg': 'Wrong user name or passcode.'})
        response.status_code = 401
        return response

    # Generate access token
    access_token = create_access_token(identity=username)

    # Return the access token in JSON format
    return jsonify(access_token=access_token)
