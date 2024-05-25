from flask_socketio import SocketIO, emit
from typing import Any
from .utils import get_labels, get_image_data

def register_sockets(socketio: SocketIO, r: Any) -> None:
    @socketio.on('connect')
    def handle_connect() -> None:
        """
        Handle client connection to the WebSocket.
        """
        emit('message', {'data': 'Connected'})
        socketio.start_background_task(update_images, socketio, r)

    @socketio.on('disconnect')
    def handle_disconnect() -> None:
        """
        Handle client disconnection from the WebSocket.
        """
        print('Client disconnected')

    @socketio.on('error')
    def handle_error(e: Exception) -> None:
        """
        Handle errors in WebSocket communication.

        Args:
            e (Exception): The exception that occurred.
        """
        print(f'Error: {str(e)}')

def update_images(socketio: SocketIO, r: Any) -> None:
    """
    Update images every 10 seconds by scanning Redis and emitting updates to clients.

    Args:
        socketio (SocketIO): The SocketIO instance.
        r (Any): The Redis connection.
    """
    while True:
        socketio.sleep(10)
        labels = get_labels(r)
        for label in labels:
            image_data = get_image_data(r, label)
            socketio.emit('update', {'label': label, 'images': [img for img, _ in image_data], 'image_names': [name for _, name in image_data]})
