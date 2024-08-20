from __future__ import annotations

import os

import redis
from dotenv import load_dotenv
from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_socketio import SocketIO

from .routes import register_routes
from .sockets import register_sockets

load_dotenv()

# Redis configuration
redis_host: str = os.getenv('redis_host', 'localhost')
redis_port: int = int(os.getenv('redis_port', '6379'))
redis_password: str | None = os.getenv('redis_password', None)

# Connect to Redis
r = redis.StrictRedis(
    host=redis_host,
    port=redis_port,
    password=redis_password,
    decode_responses=False,
)

app = Flask(__name__)
# Allow all origins for WebSocket connections
socketio = SocketIO(app, cors_allowed_origins='*')
limiter = Limiter(key_func=get_remote_address)

# Custom CORS headers


def custom_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = (
        'GET, POST, PUT, DELETE, OPTIONS'
    )
    response.headers['Access-Control-Allow-Headers'] = (
        'Content-Type, Authorization'
    )
    # Allow credentials on WebSocket connections
    return response

# Apply custom CORS headers to all responses


@app.after_request
def after_request(response):
    return custom_cors(response)


register_routes(app, limiter, r)
register_sockets(socketio, r)

if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=8000, debug=False)
