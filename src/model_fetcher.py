from pathlib import Path
import requests

def download_model(model_name, url):
    """
    Download a model file and save it to the local directory if it does not already exist.

    Args:
        model_name (str): The name of the model file.
        url (str): The URL of the model file.
    """
    # Define the local directory to store the model files
    LOCAL_MODEL_DIRECTORY = Path('models/pt/')

    # Ensure the local directory exists
    LOCAL_MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    # Build the full local file path
    local_file_path = LOCAL_MODEL_DIRECTORY / model_name

    # Check if the model already exists
    if local_file_path.exists():
        print(f"Model '{model_name}' already exists at '{local_file_path}'. Skipping download.")
        return

    # Send an HTTP GET request to fetch the model file
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        # Begin the download and write to the file
        with open(local_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Model '{model_name}' has been downloaded and saved to '{local_file_path}'.")
    else:
        print(f"Error: Unable to download model '{model_name}'. HTTP status code: {response.status_code}")

def main():
    # Define the URLs for the model files
    MODEL_URLS = {
        'best_yolov8l.pt': 'http://changdar-server.mooo.com:28000/models/best_yolov8l.pt',
        'best_yolov8x.pt': 'http://changdar-server.mooo.com:28000/models/best_yolov8x.pt'
    }

    # Iterate over all models and download them if they don't already exist
    for model_name, url in MODEL_URLS.items():
        download_model(model_name, url)

if __name__ == '__main__':
    main()
