from __future__ import annotations

from pathlib import Path
from typing import TypedDict, Dict
import requests

class ModelInfo(TypedDict):
    model_name: str
    url: str

def download_model(model_info: ModelInfo) -> None:
    """
    Download a model file if it doesn't already exist.

    Args:
        model_info (ModelInfo): The model information including name and URL.
    """
    model_name = model_info['model_name']
    url = model_info['url']

    # Define the local directory to store the model files
    LOCAL_MODEL_DIRECTORY = Path('models/pt/')

    # Ensure the local directory exists
    LOCAL_MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    # Build the full local file path
    local_file_path = LOCAL_MODEL_DIRECTORY / model_name

    # Check if the model already exists
    if local_file_path.exists():
        print(f"'{model_name}' exists. Skipping download.")
        return

    # Send an HTTP GET request to fetch the model file
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        # Begin the download and write to the file
        with open(local_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"'{model_name}' saved to '{local_file_path}'.")
    else:
        print(f"Error downloading '{model_name}': {response.status_code}")

def main() -> None:
    # Define the URLs for the model files
    MODEL_URLS: Dict[str, str] = {
        'best_yolov8l.pt': 'http://changdar-server.mooo.com:28000/models/best_yolov8l.pt',
        'best_yolov8x.pt': 'http://changdar-server.mooo.com:28000/models/best_yolov8x.pt',
    }

    # Iterate over all models and download them if they don't already exist
    for model_name, url in MODEL_URLS.items():
        model_info: ModelInfo = {
            'model_name': model_name,
            'url': url,
        }
        download_model(model_info)

if __name__ == '__main__':
    main()
