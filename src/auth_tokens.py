from __future__ import annotations

import asyncio
import logging
import os
import time
from urllib.parse import urlsplit

import aiohttp
import jwt


class AuthenticationConfigurationError(RuntimeError):
    """Authentication requires operator configuration rather than retries."""


class TokenManager:
    """Manages authentication and token refreshing for API requests."""

    def __init__(
        self,
        shared_token: dict[str, str | bool] | None = None,
    ) -> None:
        """Initialises the TokenManager instance.

        Args:
            shared_token (dict[str, str | bool] | None):
                Shared dictionary for the provider access token.
        """
        self.shared_token: dict[str, str | bool] = (
            shared_token if shared_token is not None else {
                'access_token': '',
            }
        )
        self.logger: logging.Logger = logging.getLogger(__name__)

        self._token_lock = asyncio.Lock()
        # Maximum retries for token refresh attempts.
        self.max_retries: int = 3

    @staticmethod
    def _create_session() -> aiohttp.ClientSession:
        """Create an authentication session with the shared timeout policy."""
        return aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
        )

    async def authenticate(self, force: bool = False) -> None:
        """Acquire an access token from the identity provider.

        Args:
            force (bool):
                If True, forces re-authentication even if a token exists.

        Raises:
            AuthenticationConfigurationError: If provider settings are missing.
            RuntimeError: If authentication fails.
        """
        # If token exists and not forced, skip authentication.
        if not force and self.shared_token.get('access_token'):
            return

        await self._authenticate_service_account()

    async def _authenticate_service_account(self) -> None:
        """Acquire a short-lived token without a browser or user password."""
        issuer = os.getenv('OIDC_ISSUER_URL', '').rstrip('/')
        endpoint = os.getenv('WORKER_OIDC_TOKEN_ENDPOINT', '') or (
            f'{issuer}/protocol/openid-connect/token' if issuer else ''
        )
        client_id = os.getenv('WORKER_OIDC_CLIENT_ID', '')
        secret = os.getenv('WORKER_OIDC_CLIENT_SECRET', '')
        if not endpoint or not client_id or not secret:
            raise AuthenticationConfigurationError(
                'OIDC worker requires WORKER_OIDC_CLIENT_ID, '
                'WORKER_OIDC_CLIENT_SECRET and WORKER_OIDC_TOKEN_ENDPOINT '
                '(or OIDC_ISSUER_URL).',
            )
        url = urlsplit(endpoint)
        if (
            not url.hostname or url.username or url.password
            or url.query or url.fragment
            or (
                url.scheme != 'https' and not (
                    url.scheme == 'http'
                    and url.hostname in {'localhost', '127.0.0.1', '::1'}
                )
            )
        ):
            raise AuthenticationConfigurationError(
                'Worker token endpoint must use HTTPS or loopback HTTP.',
            )
        async with self._create_session() as session:
            async with session.post(
                endpoint,
                data={
                    'grant_type': 'client_credentials',
                    'client_id': client_id,
                    'client_secret': secret,
                },
                allow_redirects=False,
            ) as response:
                if response.status != 200:
                    # Never log provider response bodies containing
                    # credentials.
                    message = f'OIDC worker authentication failed (HTTP {
                        response.status
                    })'
                    if response.status in {400, 401, 403}:
                        raise AuthenticationConfigurationError(message)
                    raise RuntimeError(message)
                data = await response.json()
        token = data.get('access_token')
        if not isinstance(token, str) or not token:
            raise AuthenticationConfigurationError(
                'OIDC response has no access token',
            )
        if str(data.get('token_type', '')).lower() != 'bearer':
            raise AuthenticationConfigurationError(
                'OIDC response must contain a Bearer token',
            )
        self.shared_token['access_token'] = token

    async def refresh_token(self) -> None:
        """Renew a worker access token with the client-credentials grant."""
        async with self._token_lock:
            await self._authenticate_service_account()

    async def ensure_token_valid(self, retry_count: int = 0) -> None:
        """Ensures a valid access token is present, authenticating if
        necessary.

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
                await self.authenticate(force=True)
            except AuthenticationConfigurationError:
                raise
            except Exception as e:
                self.logger.error(f"Token refresh/authentication failed: {e}")
                if retry_count < self.max_retries:
                    await self.ensure_token_valid(retry_count + 1)
                else:
                    raise

    async def handle_401(self, retry_count: int = 0) -> None:
        """Handles HTTP 401 errors by attempting to refresh the token, then re-
        authenticating if needed.

        Args:
            retry_count (int): Number of previous retries.

        Raises:
            RuntimeError: If maximum retries reached.
        """
        if retry_count > self.max_retries:
            raise RuntimeError('Repeated 401 errors, max_retries reached.')

        try:
            await self.refresh_token()
        except AuthenticationConfigurationError:
            raise
        except Exception as e:
            self.logger.warning(
                f"refresh_token() error: {e}, re-authenticate.",
            )
            await self.authenticate(force=True)

    def is_token_valid(self) -> bool:
        """Check if current access token exists and is not empty.

        Returns:
            bool: True if token exists and is not empty, False otherwise.
        """
        return bool(self.shared_token.get('access_token'))

    def is_token_expired(self) -> bool:
        """Check if current access token is expired or will expire soon.

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
                str(token),
                options={
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
        """Serialize acquisition so concurrent requests reuse the same
        token.
        """
        async with self._token_lock:
            return await self._get_valid_token()

    async def _get_valid_token(self) -> str:
        """Get a valid access token, refreshing or authenticating if necessary.

        Returns:
            str: A valid access token.

        Raises:
            RuntimeError: If unable to obtain a valid token.
        """
        # Check if token is valid or expired
        if not self.is_token_valid() or self.is_token_expired():
            try:
                await self.authenticate(force=True)
            except AuthenticationConfigurationError:
                raise
            except Exception as e:
                self.logger.error(f"Failed to refresh/authenticate token: {e}")
                await self.authenticate(force=True)

        token = self.shared_token.get('access_token', '')
        if not token:
            raise RuntimeError('Unable to obtain valid access token')
        return str(token)
