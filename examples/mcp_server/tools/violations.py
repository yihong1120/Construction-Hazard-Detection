from __future__ import annotations

import base64
import inspect
import logging
import os
from typing import Any

import httpx

from examples.mcp_server.config import get_env_var
from src.utils import TokenManager


async def _maybe_await(value: Any) -> Any:
    """Await value if it's awaitable; otherwise return value directly."""
    if inspect.isawaitable(value):
        return await value
    return value


class ViolationsTools:
    """Tools for querying and managing violation records."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._token_manager = None
        self._client: httpx.AsyncClient | None = None
        self._base_url: str = ''

    async def search(
        self,
        site_id: int | None = None,
        keyword: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict:
        """Search violation records with filtering.

        Args:
            site_id: Filter by site ID.
            keyword: Search keyword (supports synonyms).
            start_time: Start of time range filter (ISO 8601).
            end_time: End of time range filter (ISO 8601).
            limit: Maximum number of records to return (1-100).
            offset: Starting record offset.

        Returns:
            dict[str, Any]: A mapping including the total count and an items
            list.
        """
        try:
            await self._ensure_client()
            client = self._client
            assert client is not None

            # Build query parameters
            params: dict[str, Any] = {
                'limit': min(max(limit, 1), 100),
                'offset': max(offset, 0),
            }

            if site_id is not None:
                params['site_id'] = site_id
            if keyword:
                params['keyword'] = keyword
            if start_time:
                params['start_time'] = start_time
            if end_time:
                params['end_time'] = end_time

            # Get authorization headers
            headers = await self._get_auth_headers()

            # Make request
            response = await client.get(
                f"{self._base_url}/violations",
                params=params,
                headers=headers,
            )
            await _maybe_await(response.raise_for_status())
            return await _maybe_await(response.json())

        except Exception as e:
            self.logger.error(f"Failed to search violations: {e}")
            raise

    async def get(self, violation_id: int) -> dict:
        """Get a single violation record by ID.

        Args:
            violation_id: ID of the violation to retrieve.

        Returns:
            dict[str, Any]: A mapping containing violation details.
        """
        try:
            await self._ensure_client()
            client = self._client
            assert client is not None

            # Get authorization headers
            headers = await self._get_auth_headers()

            # Make request
            response = await client.get(
                f"{self._base_url}/violations/{violation_id}",
                headers=headers,
            )
            await _maybe_await(response.raise_for_status())
            return await _maybe_await(response.json())

        except Exception as e:
            self.logger.error(f"Failed to get violation {violation_id}: {e}")
            raise

    async def get_image(
        self,
        image_path: str,
        as_base64: bool = False,
    ) -> dict:
        """Get violation image.

        Args:
            image_path: Path to the image within the static directory.
            as_base64: When true, returns the image content as a base64 string.

        Returns:
            dict[str, Any]: A mapping containing either an ``image_base64``
            with its ``media_type`` or a direct ``url``.
        """
        try:
            await self._ensure_client()
            client = self._client
            assert client is not None

            if as_base64:
                # Get authorization headers
                headers = await self._get_auth_headers()

                # Make request
                response = await client.get(
                    f"{self._base_url}/get_violation_image",
                    params={'image_path': image_path},
                    headers=headers,
                )
                await _maybe_await(response.raise_for_status())

                # Convert response to base64
                image_base64 = base64.b64encode(
                    response.content,
                ).decode('utf-8')
                media_type = response.headers.get('content-type', 'image/jpeg')

                return {
                    'image_base64': image_base64,
                    'media_type': media_type,
                }
            else:
                # Return URL for direct access
                return {
                    'url': (
                        f"{self._base_url}/get_violation_image?"
                        f"image_path={image_path}"
                    ),
                }

        except httpx.HTTPStatusError as http_err:
            status = (
                http_err.response.status_code if http_err.response else None
            )
            self.logger.error(
                f"Failed to get violation image {image_path}: {http_err}",
            )
            return {
                'success': False,
                'status_code': status,
                'message': f"Could not retrieve image: HTTP {status}",
                'image_path': image_path,
            }
        except Exception as e:
            self.logger.error(
                f"Failed to get violation image {image_path}: {e}",
            )
            return {
                'success': False,
                'message': str(e),
                'image_path': image_path,
            }

    async def my_sites(self) -> list[dict]:
        """Return sites accessible to the current user.

        Returns:
            list[dict[str, Any]]: List of site dictionaries.
        """
        try:
            await self._ensure_client()
            client = self._client
            assert client is not None

            # Get authorization headers
            headers = await self._get_auth_headers()

            # Make request
            response = await client.get(
                f"{self._base_url}/my_sites",
                headers=headers,
            )
            await _maybe_await(response.raise_for_status())
            return await _maybe_await(response.json())

        except Exception as e:
            self.logger.error(f"Failed to get my sites: {e}")
            raise

    async def get_image_by_violation_id(
        self,
        violation_id: int,
        as_base64: bool = False,
    ) -> dict:
        """Convenience: fetch image via violation id -> image_path ->
        get_image.

        Args:
            violation_id: Violation record id.
            as_base64: When true, return base64 content.

        Returns:
            dict: Same shape as get_image.
        """
        try:
            details = await self.get(violation_id)
            image_path = details.get('image_path') or details.get('image')
            if not image_path:
                return {
                    'success': False,
                    'message': 'No image_path in violation record',
                    'violation_id': violation_id,
                }
            return await self.get_image(
                image_path=image_path,
                as_base64=as_base64,
            )
        except Exception as e:
            self.logger.error(
                f"Failed to get image by violation id {violation_id}: {e}",
            )
            return {
                'success': False,
                'message': str(e),
                'violation_id': violation_id,
            }

    async def _ensure_client(self) -> None:
        """Ensure the HTTP client and token manager are initialised."""
        if self._client is None:
            self._base_url = get_env_var(
                'VIOLATION_RECORD_API_URL',
            ).rstrip('/')
            # Remove timeout limit
            self._client = httpx.AsyncClient(timeout=None)
        if self._token_manager is None:
            self._token_manager = TokenManager(
                api_url=get_env_var('DB_MANAGEMENT_API_URL'),
            )

    async def _get_auth_headers(self) -> dict[str, str]:
        """Build authorisation headers for authenticated requests."""
        try:
            # Optional no-auth or static bearer overrides
            allow_no_auth = (
                str(os.getenv('MCP_ALLOW_NO_AUTH', '')).lower()
                in {'1', 'true', 'yes'}
            )
            static_bearer = os.getenv('MCP_STATIC_BEARER', '').strip()
            headers = {
                'User-Agent': 'ConstructionHazardDetection-MCP/1.0',
            }
            if static_bearer:
                headers['Authorization'] = f'Bearer {static_bearer}'
                return headers
            if allow_no_auth:
                return headers
            assert self._token_manager is not None
            access_token = await self._token_manager.get_valid_token()
            headers['Authorization'] = f'Bearer {access_token}'
            return headers
        except Exception as e:
            self.logger.error(f"Failed to get auth headers: {e}")
            raise

    async def close(self) -> None:
        """Clean up resources by closing the underlying HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
