from __future__ import annotations

import asyncio
import logging
import os

import httpx
from dotenv import load_dotenv

from src.async_http_client import AsyncHttpClientOwner
from src.auth_tokens import TokenManager
from src.warning_types import Warnings

# Load environment variables
load_dotenv()


class FCMSender(AsyncHttpClientOwner):
    """Class for sending FCM push notifications via backend API.

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
        """Initialise FCMSender and set API URL and configuration.

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
        self.logger = logging.getLogger(__name__)
        super().__init__(timeout)

        # Create a TokenManager instance using the shared token state
        self.token_manager = TokenManager(
            shared_token=self.shared_token,
        )

    async def send_fcm_message_to_site(
        self,
        site: str,
        stream_name: str,
        message: Warnings,
        image_path: str | None = None,
        violation_id: int | None = None,
    ) -> bool:
        """Send FCM push notification to a specific site and stream with
        optimized retry logic.

        Args:
            site (str): Name of the construction site.
            stream_name (str): Name of the live stream.
            message: Warning data.
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

        headers: dict[str, str] = {'Authorization': f"Bearer {access_token}"}
        payload: dict[str, object] = {
            'site': site,
            'stream_name': stream_name,
            'body': message,
            'image_path': image_path,
            'violation_id': violation_id,
            'type': 'violation',
            'title': 'Safety violation alert',
            'deep_link': (
                f"/violations?violation_id={violation_id}"
                if violation_id is not None
                else '/violations'
            ),
            'metadata': {
                'site': site,
                'stream_name': stream_name,
                'violation_id': violation_id,
            },
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
                    headers['Authorization'] = f"Bearer {new_token}"

                    next_delay = await self._next_retry_delay(
                        attempt,
                        backoff_delay,
                    )
                    if next_delay is None:
                        return False
                    backoff_delay = next_delay
                    continue

                response.raise_for_status()
                result: dict[str, object] = response.json()
                return bool(result.get('success', False))

            except httpx.RequestError as exc:
                self.logger.error(
                    f"API request failed (attempt {attempt + 1}): {exc}",
                )
                next_delay = await self._next_retry_delay(
                    attempt,
                    backoff_delay,
                )
                if next_delay is None:
                    return False
                backoff_delay = next_delay

            except httpx.HTTPStatusError as exc:
                if 400 <= exc.response.status_code < 500:
                    # Schema and authorisation rejections cannot succeed later.
                    self.logger.error(
                        'FCM API rejected the notification payload: %d',
                        exc.response.status_code,
                    )
                    return False
                self.logger.error(
                    'API responded with error status (attempt %d): %d',
                    attempt + 1,
                    exc.response.status_code,
                )
                next_delay = await self._next_retry_delay(
                    attempt,
                    backoff_delay,
                )
                if next_delay is None:
                    return False
                backoff_delay = next_delay

        return False

    async def _next_retry_delay(
        self,
        attempt: int,
        delay: int,
    ) -> int | None:
        """Return the next exponential delay, or stop after the final try."""
        if attempt >= self.max_retries:
            return None
        await asyncio.sleep(delay)
        return delay * 2
