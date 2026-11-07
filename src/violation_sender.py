from __future__ import annotations

import asyncio
import logging

import httpx
from dotenv import load_dotenv

from src.async_http_client import AsyncHttpClientOwner
from src.auth_tokens import TokenManager
from src.evidence import EvidenceMetadata
from src.settings.connections import ViolationTransportSettings

# Load environment variables
load_dotenv()


class ViolationSender(AsyncHttpClientOwner):
    """Responsible for sending violation images and metadata to the backend
    API.

    Handles authentication, token refresh, and retry logic for robust delivery.
    """

    def __init__(
        self,
        api_url: str | None = None,
        max_retries: int = 3,
        timeout: int = 10,
        *,
        token_manager: TokenManager | None = None,
    ) -> None:
        """Initialise the ViolationSender.

        Args:
            api_url (str | None): The base URL for the violation API endpoint.
                If None, uses environment variable.
            max_retries (int): Maximum number of retry attempts for requests.
            timeout (int): Timeout for HTTP requests in seconds.
        """
        self.settings = ViolationTransportSettings.from_overrides(
            api_url=api_url,
        )
        self.base_url: str = self.settings.api_url

        self.max_retries: int = max_retries
        super().__init__(timeout)

        logging.getLogger('httpx').setLevel(logging.WARNING)

        self.token_manager = token_manager or TokenManager()
        self.shared_token = self.token_manager.shared_token
        self._owns_token_manager = token_manager is None

    async def close(self) -> None:
        try:
            await super().close()
        finally:
            if self._owns_token_manager:
                await self.token_manager.close()

    async def send_violation(
        self,
        metadata: EvidenceMetadata,
        image_bytes: bytes,
    ) -> str | None:
        """Upload one evidence image using the shared metadata contract."""
        if bool(metadata.model_id) != bool(metadata.model_version):
            raise ValueError(
                'Model ID and inference version must be provided together',
            )
        access_token = await self.token_manager.get_valid_token()
        if not access_token:
            raise RuntimeError('Failed to obtain valid access token')
        headers, files, data, upload_url = self._build_upload_payload(
            access_token,
            metadata,
            image_bytes,
        )

        # Use shared client connection pool
        client = await self._get_client()

        # Exponential backoff retry strategy
        backoff_delay = 1

        # Attempt to send the violation data with retries
        for attempt in range(self.max_retries):
            try:
                resp = await client.post(
                    upload_url,
                    data=data,
                    files=files,
                    headers=headers,
                )
                resp.raise_for_status()
                return resp.json().get('violation_id')

            except httpx.ConnectTimeout as exc:
                logging.warning(
                    f'[send_violation] Attempt {attempt + 1}: '
                    'Connection timeout, retry...',
                )
                backoff_delay = await self._retry_or_raise(
                    attempt,
                    backoff_delay,
                    exc,
                    RuntimeError(
                        '[send_violation] All retry attempts exhausted due '
                        'to timeout',
                    ),
                )

            except httpx.HTTPStatusError as exc:
                if await self._try_refresh_on_401(exc, attempt, headers):
                    continue
                raise

            except Exception as exc:
                logging.error(f'[send_violation] Unexpected error: {exc}')
                backoff_delay = await self._retry_or_raise(
                    attempt,
                    backoff_delay,
                    exc,
                )

        # If all attempts fail, return None
        return None

    def _build_upload_payload(
        self,
        access_token: str | None,
        metadata: EvidenceMetadata,
        image_bytes: bytes,
    ) -> tuple[
        dict[str, str],
        dict[str, tuple[str, bytes, str]],
        dict[str, str],
        str,
    ]:
        headers = (
            {
                'Authorization': f'Bearer {access_token}',
            }
            if access_token
            else {}
        )
        return (
            headers,
            {'image': ('violation.jpg', image_bytes, 'image/jpeg')},
            metadata.as_form(),
            self.base_url + '/upload',
        )

    async def _retry_or_raise(
        self,
        attempt: int,
        delay: int,
        error: Exception,
        final_error: Exception | None = None,
    ) -> int:
        """Wait for the next retry or raise the appropriate final error."""
        if attempt < self.max_retries - 1:
            await asyncio.sleep(delay)
            return delay * 2
        raise final_error or error

    async def _try_refresh_on_401(
        self,
        exc: httpx.HTTPStatusError,
        attempt: int,
        headers: dict[str, str],
    ) -> bool:
        """Attempt token refresh on 401; update headers and signal retry.

        Args:
            exc (httpx.HTTPStatusError): The HTTP error that occurred.
            attempt (int): The current attempt number.
            headers (dict[str, str]): The headers to update.

        Returns:
            bool:
                True if the caller should retry,
                otherwise False (caller should raise).
        """
        if exc.response is not None and exc.response.status_code == 401:
            logging.warning(
                '[send_violation] Unauthorized. Attempting token refresh...',
            )
            await self.token_manager.refresh_token(
                headers['Authorization'].removeprefix('Bearer '),
            )
            new_token = await self.token_manager.get_valid_token()
            headers['Authorization'] = f'Bearer {new_token}'
            return attempt < self.max_retries - 1
        return False
