import threading
from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import redis
from flask_socketio import SocketIO
from flask_cors import CORS

r = redis.Redis(host='localhost', port=6379, db=0)

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from any domain
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow all origins for WebSocket connections
limiter = Limiter(key_func=get_remote_address)

from .routes import register_routes
from .sockets import register_sockets

register_routes(app, limiter, r)
register_sockets(socketio, r)

if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=8000, debug=False)
