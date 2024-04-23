from flask import Flask, render_template, send_from_directory, make_response, Response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from typing import List
import os

app = Flask(__name__)
limiter = Limiter(
    app,
    key_func=get_remote_address,  # Use the client's IP address as the key for rate limiting.
    default_limits=["200 per day", "50 per hour"]  # Default rate limits: 50 requests per hour, 200 requests per day.
)
DETECTED_FRAMES_DIR: str = os.path.abspath('detected_frames')  # Absolute path for better security and file handling.

@app.route('/')
@limiter.limit("60 per minute")  # Further limit requests to 60 per minute for this endpoint.
def index() -> str:
    """
    Serves the index page with a dynamically generated list of camera IDs.

    Returns:
        str: The rendered 'index.html' page with a list of camera IDs.
    """
    # Generate list of camera IDs from PNG files in the directory.
    camera_ids: List[str] = [f.split('.')[0] for f in os.listdir(DETECTED_FRAMES_DIR) if f.endswith('.png')]
    camera_ids.sort()
    return render_template('index.html', camera_ids=camera_ids)

@app.route('/image/<camera_id>')
@limiter.limit("60 per minute")  # Apply a rate limit to image serving.
def image(camera_id: str) -> Response:
    """
    Serves an image file based on a given camera ID, with caching disabled for real-time updates.

    Args:
        camera_id: The ID of the camera, corresponding to the filename.

    Returns:
        Response: The Flask Response object with the image file, no caching applied.
    """
    # Build the path to the specified image file.
    image_path = os.path.join(DETECTED_FRAMES_DIR, f'{camera_id}.png')
    # Send the image file with caching disabled.
    response: Response = make_response(send_from_directory(os.path.dirname(image_path), os.path.basename(image_path)))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/camera/<camera_id>')
@limiter.limit("60 per minute")  # Limit requests to enhance server performance.
def camera(camera_id: str) -> str:
    """
    Serves a dedicated page for a single camera view.

    Args:
        camera_id: The ID of the camera to view.

    Returns:
        str: The rendered 'camera.html' page for the specified camera ID.
    """
    return render_template('camera.html', camera_id=camera_id)

@app.errorhandler(429)
def ratelimit_handler(e) -> Response:
    """
    Custom error handler for rate limit exceeding (HTTP 429).

    Args:
        e: The error object provided by Flask.

    Returns:
        Response: The Flask response object with a custom error message.
    """
    return make_response('您的請求過於頻繁，請稍後再試。', 429)

if __name__ == '__main__':
    # Run the Flask application on the localhost at port 8000.
    app.run(thread=True, host='127.0.0.1', port=8000)

""" example usage
gunicorn -w 4 -b 127.0.0.1:8000 "examples.Stream-Web.app:app"
"""