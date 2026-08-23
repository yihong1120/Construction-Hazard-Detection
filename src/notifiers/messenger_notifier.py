from __future__ import annotations

import os
from io import BytesIO

import httpx
import numpy as np
from PIL import Image

_MESSENGER_MESSAGES_URL = 'https://graph.facebook.com/v11.0/me/messages'


class MessengerNotifier:
    """Send text or image notifications through Facebook Messenger."""

    def __init__(
        self,
        page_access_token: str | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Perform init.

        Args:
            page_access_token: Value used by this callable.
            client: Value used by this callable.
        """
        self.page_access_token = page_access_token or os.getenv(
            'FACEBOOK_PAGE_ACCESS_TOKEN',
        )
        self._client = client
        self._owns_client = client is None

    async def send_notification(
        self,
        recipient_id: str,
        message: str,
        image: np.ndarray | None = None,
    ) -> int:
        """Send a text message or PNG attachment and return its HTTP status."""
        token = self.page_access_token
        if not token:
            raise ValueError('FACEBOOK_PAGE_ACCESS_TOKEN missing.')

        client = self._http_client()
        headers = {'Authorization': f"Bearer {token}"}
        if image is None:
            response = await client.post(
                _MESSENGER_MESSAGES_URL,
                params={'access_token': token},
                headers=headers,
                json={
                    'message': {'text': message},
                    'recipient': {'id': recipient_id},
                },
            )
        else:
            response = await client.post(
                _MESSENGER_MESSAGES_URL,
                params={'access_token': token},
                headers=headers,
                data={
                    'recipient': f'{{"id":"{recipient_id}"}}',
                    'message': (
                        '{"attachment":{"type":"image","payload":{}}}'
                    ),
                },
                files={
                    'filedata': (
                        'image.png',
                        _png_bytes(image),
                        'image/png',
                    ),
                },
            )
        return response.status_code

    async def aclose(self) -> None:
        """Close a client created for standalone use."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _http_client(self) -> httpx.AsyncClient:
        """Return the injected transport or lazily create one for reuse."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, connect=5.0),
            )
        return self._client


def _png_bytes(image: np.ndarray) -> bytes:
    """Encode an RGB image without blocking the network event loop."""
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format='PNG')
    return buffer.getvalue()
