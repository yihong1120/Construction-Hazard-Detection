from __future__ import annotations

import datetime
import logging
import time
from pathlib import Path

import requests
import schedule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelFetcher:
    """
    A class to fetch and update model files from a server.
    """

    def __init__(
        self,
        api_url: str = 'http://your-server-address/get_new_model',
        models: list[str] | None = None,
        local_dir: str = 'models/pt',
    ) -> None:
        """Initialise the model fetcher.

        Args:
            api_url: API URL for fetching updated models.
            models: Model names to update. Common YOLO model keys are used
                when omitted.
            local_dir: Directory where model files are stored.
        """
        self.api_url = api_url
        self.models = models or [
            'yolo26n', 'yolo26s', 'yolo26m', 'yolo26l', 'yolo26x',
        ]
        self.local_dir = Path(local_dir)

    def get_last_update_time(self, model: str) -> str:
        """
        Get the last update time of a local model file.
        If the file does not exist, return Unix epoch timestamp.

        Args:
            model (str): The name of the model.

        Returns:
            str: The last modification time in ISO format.
        """
        local_file_path = self.local_dir / f'best_{model}.pt'
        if local_file_path.exists():
            last_mod_time = datetime.datetime.fromtimestamp(
                local_file_path.stat().st_mtime,
            )
        else:
            last_mod_time = datetime.datetime(1970, 1, 1)
        return last_mod_time.isoformat()

    def download_and_save_model(
        self,
        model: str,
        model_file_content: bytes,
    ) -> None:
        """
        Download and save the model file to the local directory.

        Args:
            model (str): The name of the model.
            model_file_content (bytes): The content of the model file.

        Returns:
            None
        """
        local_file_path = self.local_dir / f'best_{model}.pt'
        self.local_dir.mkdir(parents=True, exist_ok=True)
        with open(local_file_path, 'wb') as f:
            f.write(model_file_content)
        logger.info(f"Model {model} successfully updated at {local_file_path}")

    def request_new_model(self, model: str, last_update_time: str) -> None:
        """
        Request a new model file from the server.

        Args:
            model (str): The name of the model.
            last_update_time (str): The last modification time of local model.

        Returns:
            None
        """
        try:
            response = requests.get(
                self.api_url,
                params={'model': model, 'last_update_time': last_update_time},
                timeout=10,  # Set timeout for the request
            )

            if response.status_code == 200:
                data = response.json()
                if 'model_file' in data:
                    model_file_content = bytes.fromhex(data['model_file'])
                    self.download_and_save_model(model, model_file_content)
                else:
                    logger.info(f"Model {model} is already up to date.")
            else:
                logger.error(
                    f"Failed to fetch model {model}. "
                    f"Server returned status code: {response.status_code}",
                )
        except requests.exceptions.RequestException as e:
            logger.error(f"Error requesting model {model}: {e}")

    def update_all_models(self) -> None:
        """Attempt to update all configured model files."""
        for model in self.models:
            try:
                logger.info(f"Checking for updates for model {model}...")
                last_update_time = self.get_last_update_time(model)
                self.request_new_model(model, last_update_time)
            except Exception as e:
                logger.error(f"Failed to update model {model}: {e}")


# Schedule the task to run every hour
def schedule_task() -> None:
    """Run one scheduled model update task."""
    updater = ModelFetcher()
    updater.update_all_models()


def run_scheduler_loop() -> None:
    """Run pending scheduled tasks until the process is interrupted."""
    logger.info('Starting scheduled tasks. Press Ctrl+C to exit.')
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == '__main__':
    # Execute the scheduled task every hour
    schedule.every(1).hour.do(schedule_task)
    run_scheduler_loop()
