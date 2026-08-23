from __future__ import annotations

import datetime
import hashlib
import logging
import os
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


class ModelFetcher:
    """
    A class to fetch and update model files from a server.
    """

    def __init__(
        self,
        api_url: str | None = None,
        models: list[str] | None = None,
        local_dir: str = 'models/pt',
        bearer_token: str | None = None,
    ) -> None:
        """Initialise the model fetcher.

        Args:
            api_url: API URL for fetching updated models.
            models: Model names to update. Common YOLO model keys are used
                when omitted.
            local_dir: Directory where model files are stored.
        """
        self.api_url = (
            api_url
            or os.getenv('MODEL_FETCH_API_URL')
            or 'http://your-server-address/get_new_model'
        )
        self.models = models or [
            'yolo26n', 'yolo26s', 'yolo26m', 'yolo26l', 'yolo26x',
        ]
        self.local_dir = Path(local_dir)
        self.bearer_token = bearer_token or os.getenv(
            'MODEL_FETCH_BEARER_TOKEN',
        ) or ''
        self.max_download_bytes = max(
            1,
            int(os.getenv('MODEL_FETCH_MAX_BYTES', str(6 * 1024**3))),
        )

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

    def request_new_model(
        self,
        model: str,
        last_update_time: str,
        *,
        force_download: bool = False,
    ) -> bool:
        """Download a newer model as a bounded authenticated byte stream.

        Args:
            model (str): The name of the model.
            last_update_time (str): The last modification time of local model.

        Returns:
            ``True`` when a new model was atomically installed.
        """
        if not self.bearer_token:
            raise ValueError('MODEL_FETCH_BEARER_TOKEN is required')
        requested_timestamp = (
            '1970-01-01T00:00:00'
            if force_download else last_update_time
        )
        try:
            response = requests.post(
                self.api_url,
                json={
                    'model': model,
                    'last_update_time': requested_timestamp,
                },
                headers={'Authorization': f'Bearer {self.bearer_token}'},
                timeout=(5, 120),
                stream=True,
            )
            if response.status_code == 204:
                logger.info('Model %s is already up to date.', model)
                response.close()
                return False
            if response.status_code != 200:
                logger.error(
                    'Failed to fetch model %s. Server returned status code: %s',
                    model,
                    response.status_code,
                )
                response.close()
                return False
            content_length = int(response.headers.get('content-length', '0'))
            if content_length > self.max_download_bytes:
                response.close()
                raise ValueError('Model download exceeds configured size limit')
            try:
                self._stream_model_to_disk(model, response)
                return True
            finally:
                response.close()
        except requests.exceptions.RequestException as e:
            logger.error('Error requesting model %s: %s', model, e)
            return False

    def _stream_model_to_disk(
        self,
        model: str,
        response: requests.Response,
    ) -> None:
        """Write a response to a temporary file and atomically install it."""
        destination = self.local_dir / f'best_{model}.pt'
        temporary = destination.with_suffix(f'{destination.suffix}.part')
        self.local_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        digest = hashlib.sha256()
        try:
            with temporary.open('wb') as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > self.max_download_bytes:
                        raise ValueError(
                            'Model download exceeds configured size limit',
                        )
                    output.write(chunk)
                    digest.update(chunk)
            if written == 0:
                raise ValueError('Model download was empty')
            expected_checksum = response.headers.get('x-model-sha256')
            if expected_checksum and digest.hexdigest() != expected_checksum:
                raise ValueError('Model download checksum verification failed')
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        logger.info('Model %s atomically updated path=%s', model, destination)
