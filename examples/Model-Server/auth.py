from flask import Blueprint, request, jsonify
from .models import User, db
from flask_jwt_extended import create_access_token
from .cache import user_cache

auth_blueprint = Blueprint('auth', __name__)

@auth_blueprint.route('/token', methods=['POST'])
def create_token():
    username = request.json.get('username', None)
    password = request.json.get('password', None)
    user = user_cache.get(username)
    if not user:
        user = User.query.filter_by(username=username).first()
        if user:
            user_cache[username] = user
    if not user or not user.check_password(password):
        return jsonify({"msg": "用户名或密码错误"}), 401
    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token)
