from pathlib import Path
from flask import Blueprint, jsonify, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import requests
import datetime

models_blueprint = Blueprint('models', __name__)
limiter = Limiter(key_func=get_remote_address)

@models_blueprint.route('/models/<model_name>', methods=['GET'])
@limiter.limit("10 per minute")
def download_model(model_name):
    """
    Endpoint to download model files.

    Args:
        model_name (str): The name of the model file to download.

    Returns:
        Response: A Flask response object that downloads the model file.
    """
    # Define the directory where the model files are stored
    MODELS_DIRECTORY = 'models/pt/'
    # Define the external URL for model files
    MODEL_URL = f"http://changdar-server.mooo.com:28000/models/{model_name}"

    # Ensure the model name is valid
    allowed_models = ['best_yolov8l.pt', 'best_yolov8x.pt']
    if model_name not in allowed_models:
        return jsonify({"error": "Model not found."}), 404

    try:
        # Check last modified time via a HEAD request to the external server
        response = requests.head(MODEL_URL)
        if response.status_code == 200 and 'Last-Modified' in response.headers:
            server_last_modified = datetime.datetime.strptime(response.headers['Last-Modified'], '%a, %d %b %Y %H:%M:%S GMT')
            local_file_path = f"{MODELS_DIRECTORY}{model_name}"
            # Check local file's last modified time
            local_last_modified = datetime.datetime.fromtimestamp(Path(local_file_path).stat().st_mtime)
            if local_last_modified >= server_last_modified:
                return jsonify({"message": "Local model is up-to-date."}), 304

        # If not up-to-date, fetch the file and return it
        return send_from_directory(MODELS_DIRECTORY, model_name, as_attachment=True)
    except FileNotFoundError:
        return jsonify({"error": "Model not found."}), 404
    except requests.RequestException as e:
        return jsonify({"error": "Failed to fetch model information.", "details": str(e)}), 500
