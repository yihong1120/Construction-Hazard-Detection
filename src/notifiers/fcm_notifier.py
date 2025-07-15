from __future__ import annotations

import logging

import httpx

from src.utils import TokenManager


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
        api_url: str = 'https://changdar-server.mooo.com/api/fcm',
        max_retries: int = 3,
        timeout: int = 10,
    ) -> None:
        """
        Initialise FCMSender and set API URL and configuration.

        Args:
            api_url (str): The unified API URL for FCM.
            max_retries (int): Maximum number of retry attempts.
            timeout (int): Request timeout in seconds.
        """
        self.api_url = api_url
        self.shared_token: dict[str, str | bool] = {
            'access_token': '',
            'refresh_token': '',
            'is_refreshing': False,
        }
        self.max_retries = max_retries
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

        # Create a TokenManager instance using the shared token state
        self.token_manager = TokenManager(
            shared_token=self.shared_token,
        )

    async def send_fcm_message_to_site(
        self,
        site: str,
        stream_name: str,
        message: dict[str, dict[str, int]],
        image_path: str | None = None,
        violation_id: int | None = None,
        retry_count: int = 0,
    ) -> bool:
        """
        Send FCM push notification to a specific site and stream.

        Args:
            site (str): Name of the construction site.
            stream_name (str): Name of the live stream.
            message (dict[str, dict[str, int]]): Warning data,
                key is warning code, value is a dictionary of parameters
                (e.g. {"count": 3}).
            image_path (Optional[str]):
                Image URL to display in the notification.
            violation_id (Optional[int]): Violation record ID.
            retry_count (int): Current retry attempt count.

        Returns:
            bool:
                True if the API call and push notification succeed,
                False otherwise.
        """
        if retry_count > self.max_retries:
            self.logger.error(
                (
                    f"Exceeded max_retries ({self.max_retries}); "
                    'abort sending FCM.'
                ),
            )
            return False

        # Authenticate if no token is present
        if not self.shared_token or not self.shared_token.get('access_token'):
            await self.token_manager.authenticate(force=True)

        # Retrieve access token from shared state
        api_token: str = str(self.shared_token.get('access_token', ''))
        headers: dict[str, str] = {'Authorization': f'Bearer {api_token}'}
        payload: dict[str, object] = {
            'site': site,
            'stream_name': stream_name,
            'body': message,
            'image_path': image_path,
            'violation_id': violation_id,
        }
        # Use the unified API URL's send_fcm_notification endpoint
        endpoint: str = f"{self.api_url}/send_fcm_notification"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response: httpx.Response = await client.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                )
                if response.status_code == 401:
                    self.logger.warning(
                        (
                            f"FCM API got 401. Attempting to refresh token... "
                            f"(retry_count={retry_count})"
                        ),
                    )
                    await self.token_manager.handle_401(
                        retry_count=retry_count,
                    )
                    # Recursively retry after refreshing token
                    return await self.send_fcm_message_to_site(
                        site=site,
                        stream_name=stream_name,
                        message=message,
                        image_path=image_path,
                        violation_id=violation_id,
                        retry_count=retry_count + 1,
                    )
                response.raise_for_status()
                result: dict[str, object] = response.json()
                return bool(result.get('success', False))
            except httpx.RequestError as exc:
                self.logger.error(
                    f"API request failed: {exc}",
                )
                return False
            except httpx.HTTPStatusError as exc:
                self.logger.error(
                    (
                        f"API responded with error status: "
                        f"{exc.response.status_code}"
                    ),
                )
                return False
