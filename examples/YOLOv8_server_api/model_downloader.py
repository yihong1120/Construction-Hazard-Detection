from __future__ import annotations

import datetime
from pathlib import Path

import requests
from flask import Blueprint
from flask import current_app as app
from flask import jsonify
from flask import send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

models_blueprint = Blueprint('models', __name__)
limiter = Limiter(key_func=get_remote_address)

# Define the directory where the model files are stored
MODELS_DIRECTORY = Path('models/pt/')
# Define the allowed models
ALLOWED_MODELS = {'best_yolov8l.pt', 'best_yolov8x.pt'}


@models_blueprint.route('/models/<model_name>', methods=['GET'])
@limiter.limit('10 per minute')
def download_model(model_name: str):
    """
    Endpoint to download model files.

    Args:
        model_name (str): The name of the model file to download.

    Returns:
        Response: A Flask response object that downloads the model file.
    """
    # Ensure the model name is valid
    if model_name not in ALLOWED_MODELS:
        return jsonify({'error': 'Model not found.'}), 404

    try:
        # Define the external URL for model files
        MODEL_URL = (
            f"http://changdar-server.mooo.com:28000/models/{model_name}"
        )

        # Check last modified time via a HEAD request to the external server
        response = requests.head(MODEL_URL)
        if response.status_code == 200 and 'Last-Modified' in response.headers:
            server_last_modified = datetime.datetime.strptime(
                response.headers['Last-Modified'],
                '%a, %d %b %Y %H:%M:%S GMT',
            )

            # Use Path to safely construct the file path
            local_file_path = MODELS_DIRECTORY / model_name

            # Ensure the constructed path is within the expected directory
            try:
                local_file_path = (
                    local_file_path.resolve().relative_to(
                        MODELS_DIRECTORY.resolve(),
                    )
                )
            except ValueError:
                return jsonify({'error': 'Invalid model name.'}), 400

            # Check local file's last modified time
            if local_file_path.exists():
                local_last_modified = datetime.datetime.fromtimestamp(
                    local_file_path.stat().st_mtime,
                )
                if local_last_modified >= server_last_modified:
                    return jsonify(
                        {
                            'message': 'Local model is up-to-date.',
                        },
                    ), 304

        # If not up-to-date, fetch the file and return it
        return send_from_directory(
            MODELS_DIRECTORY,
            model_name,
            as_attachment=True,
        )

    except FileNotFoundError:
        return jsonify(
            {
                'error': 'Model not found.',
            },
        ), 404

    except requests.RequestException:
        # Log the exception details for debugging purposes
        app.logger.error('Failed to fetch model information')
        return jsonify({'error': 'Failed to fetch model information.'}), 500
