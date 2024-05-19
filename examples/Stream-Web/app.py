import base64
from flask import Flask, abort, render_template, make_response, Response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import redis
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from typing import List, Tuple

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from any domain
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow all origins for WebSocket connections
limiter = Limiter(key_func=get_remote_address)

CACHE_DURATION = 60  # Cache duration in seconds

def get_labels() -> List[str]:
    """
    Retrieve and decode all unique labels from Redis keys.

    Returns:
        list: Sorted list of unique labels.
    """
    cursor, keys = r.scan()
    decoded_keys = [key.decode('utf-8') for key in keys]
    labels = {key.split('_')[0] for key in decoded_keys if key.count('_') == 1 and not key.startswith('_') and not key.endswith('_')}
    return sorted(labels)

def get_image_data(label: str) -> List[Tuple[str, str]]:
    """
    Retrieve and process image data for a specific label.

    Args:
        label (str): The label/category of the images.

    Returns:
        list: List of tuples containing base64 encoded images and their names.
    """
    cursor, keys = r.scan(match=f'{label}_*')
    image_data = [(base64.b64encode(r.get(key)).decode('utf-8'), key.decode('utf-8').split('_')[1]) for key in keys]
    return sorted(image_data, key=lambda x: x[1])

@app.route('/')
@limiter.limit("60 per minute")
def index() -> str:
    """
    Serve the index page with a dynamically generated list of labels from Redis.

    Returns:
        str: The rendered 'index.html' page.
    """
    labels = get_labels()
    return render_template('index.html', labels=labels)

@app.route('/label/<label>', methods=['GET'])
@limiter.limit("60 per minute")
def label_page(label: str) -> str:
    """
    Serve a page for each label, displaying images under that label.

    Args:
        label (str): The label/category of the images.

    Returns:
        str: The rendered 'label.html' page for the specific label.
    """
    image_data = get_image_data(label)
    return render_template('label.html', label=label, image_data=image_data)

@socketio.on('connect')
def handle_connect() -> None:
    """
    Handle client connection to the WebSocket.
    """
    emit('message', {'data': 'Connected'})
    socketio.start_background_task(update_images)

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

def update_images() -> None:
    """
    Update images every 10 seconds by scanning Redis and emitting updates to clients.
    """
    while True:
        socketio.sleep(10)  # Execute every 20 seconds
        labels = get_labels()
        for label in labels:
            image_data = get_image_data(label)
            socketio.emit('update', {'label': label, 'images': [img for img, _ in image_data], 'image_names': [name for _, name in image_data]})

@app.route('/image/<label>/<filename>.png')
@limiter.limit("60 per minute")
def image(label: str, filename: str) -> Response:
    """
    Serve an image file from Redis.

    Args:
        label (str): The label/category of the image.
        filename (str): The filename of the image.

    Returns:
        Response: The image file as a response.
    """
    redis_key = f"{label}_{filename}"
    img_encoded = r.get(redis_key)

    if img_encoded is None:
        abort(404, description="Resource not found")

    response = make_response(img_encoded)
    response.headers.set('Content-Type', 'image/png')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'

    return response

@app.route('/camera/<label>/<camera_id>')
@limiter.limit("60 per minute")
def camera_page(label: str, camera_id: str) -> str:
    """
    Serve a page for viewing the camera stream.

    Args:
        label (str): The label/category of the camera.
        camera_id (str): The specific ID of the camera to view.

    Returns:
        str: The rendered 'camera.html' page for the specific camera.
    """
    return render_template('camera.html', label=label, camera_id=camera_id)

if __name__ == '__main__':
    socketio.start_background_task(target=update_images)
    socketio.run(app, host='127.0.0.1', port=8000, debug=False)
