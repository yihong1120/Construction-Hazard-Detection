from __future__ import annotations

import asyncio
import logging
import os

import httpx
from dotenv import load_dotenv

from src.utils import TokenManager

# Load environment variables
load_dotenv()


class FCMSender:
    """
    Class for sending FCM push notifications via backend API.
    Each instance maintains its own token state.
    """

    api_url: str
    shared_token: dict[str, str | bool]
    max_retries: int
    timeout: int
    logger: logging.Logger
    token_manager: TokenManager

    def __init__(
        self,
        api_url: str | None = None,
        max_retries: int = 3,
        timeout: int = 10,
    ) -> None:
        """
        Initialise FCMSender and set API URL and configuration.

        Args:
            api_url (str | None): The unified API URL for FCM. If None, uses
                environment variable.
            max_retries (int): Maximum number of retry attempts.
            timeout (int): Request timeout in seconds.
        """
        # Read FCM API URL from environment variables if not provided
        if api_url is None:
            api_url = os.getenv('FCM_API_URL', 'http://127.0.0.1:8003')

        self.api_url = api_url
        self.shared_token: dict[str, str | bool] = {
            'access_token': '',
            'refresh_token': '',
            'is_refreshing': False,
        }
        self.max_retries = max_retries
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

        # Create an HTTP client connection pool
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

        # Create a TokenManager instance using the shared token state
        self.token_manager = TokenManager(
            shared_token=self.shared_token,
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """
        Obtain or create the HTTP client connection pool.

        Returns:
            httpx.AsyncClient: Asynchronous HTTP client.
        """
        async with self._client_lock:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(self.timeout),
                    limits=httpx.Limits(
                        max_keepalive_connections=5,
                        max_connections=10,
                        keepalive_expiry=30,
                    ),
                    http2=True,  # 啟用 HTTP/2 以提升效能
                )
            return self._client

    async def close(self) -> None:
        """
        Close the HTTP client connection pool.
        """
        async with self._client_lock:
            if self._client and not self._client.is_closed:
                await self._client.aclose()
                self._client = None

    async def send_fcm_message_to_site(
        self,
        site: str,
        stream_name: str,
        message: dict[str, dict[str, int]],
        image_path: str | None = None,
        violation_id: int | None = None,
    ) -> bool:
        """
        Send FCM push notification to a specific site and stream
        with optimized retry logic.

        Args:
            site (str): Name of the construction site.
            stream_name (str): Name of the live stream.
            message (dict[str, dict[str, int]]): Warning data.
            image_path (Optional[str]):
                Image URL to display in the notification.
            violation_id (Optional[int]): Violation record ID.

        Returns:
            bool:
                True if the API call and push notification succeed,
                False otherwise.
        """
        # Get valid token using TokenManager
        access_token = await self.token_manager.get_valid_token()
        if not access_token:
            self.logger.error('Failed to obtain valid access token')
            return False

        headers: dict[str, str] = {'Authorization': f'Bearer {access_token}'}
        payload: dict[str, object] = {
            'site': site,
            'stream_name': stream_name,
            'body': message,
            'image_path': image_path,
            'violation_id': violation_id,
        }
        endpoint: str = f"{self.api_url}/send_fcm_notification"

        # Use shared client connection pool
        client = await self._get_client()

        # Exponential backoff retry strategy
        backoff_delay = 1

        for attempt in range(self.max_retries + 1):
            try:
                response: httpx.Response = await client.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                )

                if response.status_code == 401:
                    self.logger.warning(
                        'FCM API got 401. Attempting to refresh token... '
                        '(attempt %d)',
                        attempt + 1,
                    )
                    # Refresh token and update headers
                    await self.token_manager.refresh_token()
                    new_token = await self.token_manager.get_valid_token()
                    headers['Authorization'] = f'Bearer {new_token}'

                    if attempt < self.max_retries:
                        await asyncio.sleep(backoff_delay)
                        backoff_delay *= 2
                        continue
                    return False

                response.raise_for_status()
                result: dict[str, object] = response.json()
                return bool(result.get('success', False))

            except httpx.RequestError as exc:
                self.logger.error(
                    f"API request failed (attempt {attempt + 1}): {exc}",
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(backoff_delay)
                    backoff_delay *= 2
                else:
                    return False

            except httpx.HTTPStatusError as exc:
                self.logger.error(
                    'API responded with error status (attempt %d): %d',
                    attempt + 1,
                    exc.response.status_code,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(backoff_delay)
                    backoff_delay *= 2
                else:
                    return False

        return False
