from __future__ import annotations

import asyncio
import logging
import os
import time

import aiohttp
import jwt


class TokenManager:
    """
    Manages authentication and token refreshing for API requests.
    """

    def __init__(
        self,
        api_url: str | None = None,
        shared_token: dict[str, str | bool] | None = None,
    ) -> None:
        """
        Initialises the TokenManager instance.

        Args:
            api_url (str | None): The base API URL for authentication.
            shared_token (dict[str, str | bool] | None):
                Shared token dictionary for storing access and refresh tokens.
        """
        # API endpoint for authentication;
        # defaults to environment variable or local address.
        self.api_url: str = api_url or os.getenv(
            'DB_MANAGEMENT_API_URL',
        ) or 'http://127.0.0.1:8005'
        # Shared token dictionary for access/refresh tokens and refresh state.
        self.shared_token: dict[str, str | bool] = shared_token or {
            'access_token': '',
            'refresh_token': '',
            'is_refreshing': False,
        }
        self.logger: logging.Logger = logging.getLogger(__name__)

        # Maximum retries for token refresh attempts.
        self.max_retries: int = 3

    @staticmethod
    def _create_session() -> aiohttp.ClientSession:
        """Create an authentication session with the shared timeout policy."""
        return aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
        )

    async def authenticate(self, force: bool = False) -> None:
        """
        Authenticates with the API and retrieves access/refresh tokens.

        Args:
            force (bool):
                If True, forces re-authentication even if a token exists.

        Raises:
            ValueError: If username or password is missing.
            RuntimeError: If authentication fails.
        """
        # If token exists and not forced, skip authentication.
        if not force and self.shared_token.get('access_token'):
            return

        # Load credentials from environment variables (supports .env)
        username: str = os.getenv('API_USERNAME', '')
        password: str = os.getenv('API_PASSWORD', '')
        hcaptcha_bypass_key: str = os.getenv('HCAPTCHA_BYPASS_KEY', '')

        if not username or not password:
            raise ValueError('Missing API_USERNAME or API_PASSWORD')

        try:
            headers: dict[str, str] = {}
            if hcaptcha_bypass_key:
                headers['X-HCaptcha-Bypass-Key'] = hcaptcha_bypass_key

            async with self._create_session() as session:
                resp: aiohttp.ClientResponse = await session.post(
                    f"{self.api_url}/login",
                    json={'identifier': username, 'password': password},
                    headers=headers,
                )
                if resp.status != 200:
                    error_text = await resp.text()
                    msg: str = (
                        f"Authenticate failed with status {resp.status}: "
                        f"{error_text}"
                    )
                    self.logger.error(msg)
                    raise RuntimeError(msg)

                data: dict = await resp.json()
                self.shared_token['access_token'] = data['access_token']
                self.shared_token['refresh_token'] = data.get(
                    'refresh_token', '',
                )
                self.logger.info(
                    'Successfully authenticated and retrieved token.',
                )

        except aiohttp.ClientError as e:
            self.logger.error(f"Network error during authentication: {e}")
            raise RuntimeError(
                f"Authentication failed due to network error: {e}",
            )
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            raise

    async def refresh_token(self) -> None:
        """
        Refreshes the access token using the refresh token.

        Raises:
            RuntimeError:
                If refresh fails repeatedly or returns unexpected status.
        """
        # If another refresh is in progress, wait up to 10 seconds.
        if self.shared_token.get('is_refreshing'):
            await self._wait_for_refresh_completion()
            return

        refresh_token: str = str(self.shared_token.get('refresh_token', ''))
        if not refresh_token:
            # No refresh token available; force re-authentication.
            await self.authenticate(force=True)
            return

        try:
            # Double check for token changes during wait.
            if self.shared_token.get('refresh_token') != refresh_token:
                return

            self.shared_token['is_refreshing'] = True
            self.logger.warning('Token expired. Attempting to refresh...')

            async with self._create_session() as session:
                # First attempt with the bearer-token header.
                resp = await self._attempt_token_refresh(
                    session, refresh_token, with_auth=True,
                )

                # Retry without header if 401 returned.
                if resp.status == 401:
                    resp = await self._attempt_token_refresh(
                        session, refresh_token, with_auth=False,
                    )

                if resp.status == 200:
                    data: dict = await resp.json()
                    self.shared_token['access_token'] = data['access_token']
                    self.shared_token['refresh_token'] = data['refresh_token']
                    self.logger.info('Token refreshed successfully.')
                else:
                    self.logger.warning(f"Refresh failed: {resp.status}")
                    if resp.status in (401, 403):
                        await self.authenticate(force=True)
                    else:
                        error_text = await resp.text()
                        raise RuntimeError(
                            f"Refresh failed with status {resp.status}: "
                            f"{error_text}",
                        )
        finally:
            self.shared_token['is_refreshing'] = False

    async def _wait_for_refresh_completion(self) -> None:
        """Wait for another refresh operation to complete."""
        wait_time: float = 0.0
        while self.shared_token.get('is_refreshing'):
            await asyncio.sleep(0.1)
            wait_time += 0.1
            if wait_time >= 10:
                self.logger.warning(
                    'Waited 10s for refresh to finish, giving up.',
                )
                return

    async def _attempt_token_refresh(
        self,
        session: aiohttp.ClientSession,
        refresh_token: str,
        with_auth: bool = True,
    ) -> aiohttp.ClientResponse:
        """Attempt to refresh token with or without Authorization header."""
        headers = {}
        if with_auth:
            headers['Authorization'] = (
                f"Bearer {self.shared_token['access_token']}"
            )

        return await session.post(
            f"{self.api_url}/refresh",
            json={'refresh_token': refresh_token},
            headers=headers,
        )

    async def ensure_token_valid(self, retry_count: int = 0) -> None:
        """
        Ensures a valid access token is present, authenticating if necessary.

        Args:
            retry_count (int): Number of previous retries.

        Raises:
            RuntimeError: If maximum retries exceeded.
        """
        if retry_count > self.max_retries:
            raise RuntimeError(
                'Exceeded max_retries in ensure_token_valid, aborting...',
            )

        # Check if token is valid or expired
        if not self.is_token_valid() or self.is_token_expired():
            try:
                if self.shared_token.get('refresh_token'):
                    self.logger.info(
                        'Token expired or missing, attempting refresh...',
                    )
                    await self.refresh_token()
                else:
                    self.logger.info(
                        'No refresh token available, re-authenticating...',
                    )
                    await self.authenticate(force=True)
            except Exception as e:
                self.logger.error(f"Token refresh/authentication failed: {e}")
                if retry_count < self.max_retries:
                    await self.ensure_token_valid(retry_count + 1)
                else:
                    raise

    async def handle_401(self, retry_count: int = 0) -> None:
        """
        Handles HTTP 401 errors by attempting to refresh the token,
        then re-authenticating if needed.

        Args:
            retry_count (int): Number of previous retries.

        Raises:
            RuntimeError: If maximum retries reached.
        """
        if retry_count > self.max_retries:
            raise RuntimeError('Repeated 401 errors, max_retries reached.')

        try:
            await self.refresh_token()
        except Exception as e:
            self.logger.warning(
                f"refresh_token() error: {e}, re-authenticate.",
            )
            await self.authenticate(force=True)

    def is_token_valid(self) -> bool:
        """
        Check if current access token exists and is not empty.

        Returns:
            bool: True if token exists and is not empty, False otherwise.
        """
        return bool(self.shared_token.get('access_token'))

    def is_token_expired(self) -> bool:
        """
        Check if current access token is expired or will expire soon.

        Returns:
            bool:
                True if token is expired or will expire within 60 seconds,
                False otherwise.
        """
        token = self.shared_token.get('access_token')
        if not token:
            return True

        try:
            # Decode JWT token without verifying signature
            # (since we only need to check expiry)
            decoded = jwt.decode(
                str(token), options={
                    'verify_signature': False,
                },
            )
            exp = decoded.get('exp')
            if exp:
                # Check if token is expiring within 60 seconds
                current_time = time.time()
                return current_time >= (exp - 60)  # Refresh 60 seconds early
            return False
        except Exception as e:
            self.logger.warning(
                f"Failed to decode token for expiry check: {e}",
            )
            return True  # Treat tokens that cannot be decoded as expired.

    async def get_valid_token(self) -> str:
        """
        Get a valid access token, refreshing or authenticating if necessary.

        Returns:
            str: A valid access token.

        Raises:
            RuntimeError: If unable to obtain a valid token.
        """
        # Check if token is valid or expired
        if not self.is_token_valid() or self.is_token_expired():
            try:
                if self.shared_token.get('refresh_token'):
                    await self.refresh_token()
                else:
                    await self.authenticate(force=True)
            except Exception as e:
                self.logger.error(f"Failed to refresh/authenticate token: {e}")
                await self.authenticate(force=True)

        token = self.shared_token.get('access_token', '')
        if not token:
            raise RuntimeError('Unable to obtain valid access token')
        return str(token)
