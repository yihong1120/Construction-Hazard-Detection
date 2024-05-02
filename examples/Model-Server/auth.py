from flask import Blueprint, request, jsonify
from .models import User, db
from flask_jwt_extended import create_access_token
from .cache import user_cache

auth_blueprint = Blueprint('auth', __name__)

@auth_blueprint.route('/token', methods=['POST'])
def create_token() -> jsonify:
    """
    Authenticates a user and generates a JWT token.

    Returns:
        jsonify: A Flask response object that contains the JWT token or an error message.

    Raises:
        HTTP 401: If the username or password is incorrect.
    """
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
        return jsonify({"msg": "Wrong user name or passcode."}), 401
    
    # Generate access token
    access_token = create_access_token(identity=username)
    
    # Return the access token in JSON format
    return jsonify(access_token=access_token)
