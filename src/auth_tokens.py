from __future__ import annotations

import asyncio
import logging
import time

import jwt

from src.async_http_client import AsyncHttpClientOwner
from src.settings.worker_auth import WorkerOidcSettings


class AuthenticationConfigurationError(RuntimeError):
    """Authentication requires configuration rather than repeated attempts."""


class TokenManager(AsyncHttpClientOwner):
    """Own a pooled worker credential transport and one refresh operation."""

    trust_env = False

    def __init__(self, shared_token: dict[str, str | bool] | None = None):
        super().__init__(10)
        self.shared_token: dict[str, str | bool] = (
            shared_token
            if shared_token is not None
            else {
                'access_token': '',
            }
        )
        self.logger = logging.getLogger(__name__)
        self._token_lock = asyncio.Lock()
        self._oidc_settings: WorkerOidcSettings | None = None

    async def authenticate(self, force: bool = False) -> None:
        async with self._token_lock:
            if (
                not force
                and self.is_token_valid()
                and not self.is_token_expired()
            ):
                return
            await self._authenticate_service_account()

    async def _authenticate_service_account(self) -> None:
        if self._oidc_settings is None:
            try:
                self._oidc_settings = WorkerOidcSettings()
            except ValueError as config_error:
                raise AuthenticationConfigurationError(
                    str(config_error),
                ) from config_error
        config = self._oidc_settings
        endpoint = config.endpoint
        client_id = config.client_id
        secret = config.client_secret.get_secret_value()
        client = await self._get_client()
        response = await client.post(
            endpoint,
            data={
                'grant_type': 'client_credentials',
                'client_id': client_id,
                'client_secret': secret,
            },
            follow_redirects=False,
        )
        if response.status_code != 200:
            error = (
                'OIDC worker authentication failed '
                f'(HTTP {response.status_code})'
            )
            if response.status_code in {400, 401, 403}:
                raise AuthenticationConfigurationError(error)
            raise RuntimeError(error)
        data = response.json()
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

    async def refresh_token(self, rejected_token: str | None = None) -> None:
        async with self._token_lock:
            if (
                rejected_token is not None
                and self.shared_token.get('access_token') != rejected_token
            ):
                return
            await self._authenticate_service_account()

    async def ensure_token_valid(self) -> None:
        await self.get_valid_token()

    async def handle_401(self, rejected_token: str | None = None) -> None:
        await self.refresh_token(rejected_token)

    def is_token_valid(self) -> bool:
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
        await self.authenticate()
        token = self.shared_token.get('access_token')
        if not token:
            raise RuntimeError('Unable to obtain valid access token')
        return str(token)
