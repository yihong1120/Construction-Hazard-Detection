from flask import Flask, redirect, render_template, send_from_directory, make_response, Response, request, jsonify, url_for
import subprocess
import threading
import os
import signal
from threading import Timer
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pathlib import Path
import json
import uuid
import time

app = Flask(__name__)
limiter = Limiter(key_func=get_remote_address)

DETECTED_FRAMES_DIR = Path.cwd() / 'detected_frames'
CACHE_DURATION = 60  # Cache duration in seconds
cache = {}

@app.route('/')
@limiter.limit("60 per minute")  # Apply rate limiting to this endpoint
def index() -> str:
    """
    Serves the index page with a dynamically generated list of labels from subdirectories.

    Returns:
        str: The rendered 'index.html' page.
    """
    labels = [d.name for d in DETECTED_FRAMES_DIR.iterdir() if d.is_dir()]
    labels.sort()
    return render_template('index.html', labels=labels)

@app.route('/label/<label>')
@limiter.limit("60 per minute")  # Apply rate limiting to this endpoint
def label_page(label: str) -> str:
    """
    Serves a page for each label, displaying images under that label.

    Args:
        label: The label/category of the images.

    Returns:
        str: The rendered 'label.html' page for the specific label.
    """
    current_time = time.time()
    # Check if the label is in the cache and the cache duration has not expired
    if label in cache and current_time - cache[label]['timestamp'] < CACHE_DURATION:
        image_files = cache[label]['files']
    else:
        image_files = [f.name for f in (DETECTED_FRAMES_DIR / label).iterdir() if f.suffix == '.png']
        image_files.sort()
        cache[label] = {'files': image_files, 'timestamp': current_time}

    return render_template('label.html', label=label, images=image_files, path=f'/image/{label}/')

@app.route('/image/<label>/<filename>.png')
@limiter.limit("60 per minute")  # Apply rate limiting to this endpoint
def image(label: str, filename: str) -> Response:
    """
    Serves an image file from a label subdirectory.

    Args:
        label: The label/category of the image.
        filename: The filename of the image.

    Returns:
        Response: The image file as a response.
    """
    image_path = DETECTED_FRAMES_DIR / label / (filename + '.png')
    response = make_response(send_from_directory(image_path.parent, image_path.name))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/camera/<label>/<camera_id>')
@limiter.limit("60 per minute")  # Apply rate limiting to this endpoint
def camera_page(label: str, camera_id: str) -> str:
    """
    Serves a page for viewing the camera stream.

    Args:
        label: The label/category of the camera.
        camera_id: The specific ID of the camera to view.

    Returns:
        str: The rendered 'camera.html' page for the specific camera.
    """
    return render_template('camera.html', label=label, camera_id=camera_id)

if __name__ == '__main__':
    # Run the Flask application on localhost at port 8000.
    app.run(thread=True,host='127.0.0.1', port=8000, debug=False)

""" example usage
gunicorn -w 4 -b 127.0.0.1:8000 "examples.Stream-Web.app:app"
"""