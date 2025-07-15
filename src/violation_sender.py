from __future__ import annotations

import logging
from datetime import datetime

import httpx

from src.utils import TokenManager


class ViolationSender:
    """
    Responsible for sending violation images and metadata to the backend API.
    Handles authentication, token refresh, and retry logic for robust delivery.
    """

    def __init__(
        self,
        api_url: str = 'http://127.0.0.1:8002',
        max_retries: int = 3,
        timeout: int = 10,
    ) -> None:
        """
        Initialise the ViolationSender.

        Args:
            api_url (str): The base URL for the violation API endpoint.
            max_retries (int): Maximum number of retry attempts for requests.
            timeout (int): Timeout for HTTP requests in seconds.
        """
        self.base_url: str = api_url.rstrip('/')
        self.shared_token: dict[str, str | bool] = {
            'access_token': '',
            'refresh_token': '',
            'is_refreshing': False,
        }
        self.max_retries: int = max_retries
        self.timeout: int = timeout

        logging.getLogger('httpx').setLevel(logging.WARNING)

        self.token_manager: TokenManager = TokenManager(
            shared_token=self.shared_token,
        )

    async def send_violation(
        self,
        site: str,
        stream_name: str,
        image_bytes: bytes,
        detection_time: datetime | None = None,
        warnings_json: str | None = None,
        detections_json: str | None = None,
        cone_polygon_json: str | None = None,
        pole_polygon_json: str | None = None,
    ) -> str | None:
        """
        Send a violation image and associated metadata to the backend API.

        Args:
            site (str): The site label.
            stream_name (str): The stream identifier.
            image_bytes (bytes): The image data in bytes.
            detection_time (Optional[datetime]): The time of detection.
            warnings_json (Optional[str]): JSON string of warnings.
            detections_json (Optional[str]): JSON string of detection items.
            cone_polygon_json (Optional[str]): JSON string of cone polygons.
            pole_polygon_json (Optional[str]): JSON string of pole polygons.

        Returns:
            Optional[str]:
                The violation ID (string) if successful,
                or None if all attempts fail.

        Raises:
            RuntimeError:
                If all retry attempts are exhausted or a critical error occurs.
        """
        # Ensure authentication before sending
        if not self.shared_token or not self.shared_token.get('access_token'):
            await self.token_manager.authenticate(force=True)

        access_token: str = str(self.shared_token.get('access_token', ''))

        headers: dict[str, str] = {}
        if access_token:
            headers['Authorization'] = f"Bearer {access_token}"

        files: dict[str, tuple[str, bytes, str]] = {
            'image': ('violation.png', image_bytes, 'image/png'),
        }

        data: dict[str, str] = {
            'site': site,
            'stream_name': stream_name,
        }
        if detection_time:
            data['detection_time'] = detection_time.isoformat()
        if warnings_json:
            data['warnings_json'] = warnings_json
        if detections_json:
            data['detections_json'] = detections_json
        if cone_polygon_json:
            data['cone_polygon_json'] = cone_polygon_json
        if pole_polygon_json:
            data['pole_polygon_json'] = pole_polygon_json

        upload_url: str = self.base_url + '/upload'

        # Attempt to send the violation data with retries
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        upload_url,
                        data=data,
                        files=files,
                        headers=headers,
                    )
                    resp.raise_for_status()
                    # Return the violation ID from the response
                    return resp.json().get('violation_id')

            except httpx.ConnectTimeout:
                print(
                    f"[send_violation] Attempt {attempt+1}: "
                    'Connection timeout, retry...',
                )
                # Raise RuntimeError if final attempt fails
                if attempt == self.max_retries - 1:
                    raise RuntimeError(
                        '[send_violation] All retry attempts exhausted, '
                        'no success.',
                    )

            except httpx.HTTPStatusError as exc:
                # If 401, refresh token and retry
                if exc.response.status_code == 401:
                    print(
                        '[send_violation] Unauthorized. '
                        'Attempting token refresh...',
                    )
                    await self.token_manager.refresh_token()
                    return await self.send_violation(
                        site,
                        stream_name,
                        image_bytes,
                        detection_time,
                        warnings_json,
                        detections_json,
                        cone_polygon_json,
                        pole_polygon_json,
                    )
                raise

            except Exception as e:
                print(f"[send_violation] Unexpected error: {e}")
                raise

        # If all attempts fail, return None
        return None
