from flask import Flask, render_template, send_from_directory, make_response, Response
from typing import List
import os

app = Flask(__name__)
DETECTED_FRAMES_DIR = os.path.abspath('detected_frames')  # Use an absolute path for increased security.

@app.route('/')
def index() -> str:
    """
    Serve the index page with dynamically generated list of camera IDs.

    Returns:
        Rendered 'index.html' with a list of camera IDs.
    """
    # Dynamically generate list of camera IDs from PNG files in the directory.
    camera_ids: List[str] = [f.split('.')[0] for f in os.listdir(DETECTED_FRAMES_DIR) if f.endswith('.png')]
    camera_ids.sort()
    return render_template('index.html', camera_ids=camera_ids)

@app.route('/image/<camera_id>')
def image(camera_id: str) -> Response:
    """
    Serve an image file with the given camera ID, ensuring no caching to allow real-time updates.

    Args:
        camera_id: The ID of the camera, which corresponds to the filename.

    Returns:
        Flask Response object with the image file.
    """
    # Construct the path to the image file.
    image_path = os.path.join(DETECTED_FRAMES_DIR, f'{camera_id}.png')
    # Send the image file from the directory, ensuring no caching occurs.
    response: Response = make_response(send_from_directory(os.path.dirname(image_path), os.path.basename(image_path)))
    # Set HTTP headers to disable caching.
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/camera/<camera_id>')
def camera(camera_id: str) -> str:
    """
    Serve a page dedicated to a single camera view.

    Args:
        camera_id: The ID of the camera to view.

    Returns:
        Rendered 'camera.html' for the specified camera ID.
    """
    return render_template('camera.html', camera_id=camera_id)

if __name__ == '__main__':
    # Run the Flask application on localhost at port 8000.
    app.run(thread=True,host='127.0.0.1', port=8000)

""" example usage
gunicorn -w 4 -b 127.0.0.1:8000 "examples.Stream-Web.app:app"
"""