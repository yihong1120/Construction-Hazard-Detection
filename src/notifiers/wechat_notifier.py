from __future__ import annotations

import asyncio
import os
from io import BytesIO

import httpx
import numpy as np
from PIL import Image

_WECHAT_API_ROOT = 'https://qyapi.weixin.qq.com/cgi-bin'
_TOKEN_REFRESH_MARGIN_SECONDS = 60


class WeChatNotifier:
    """Send text or image notifications through WeChat Work asynchronously."""

    def __init__(
        self,
        corp_id: str | None = None,
        corp_secret: str | None = None,
        agent_id: int | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Perform init.

        Args:
            corp_id: Value used by this callable.
            corp_secret: Value used by this callable.
            agent_id: Value used by this callable.
            client: Value used by this callable.
        """
        self.corp_id = corp_id or os.getenv('WECHAT_CORP_ID')
        self.corp_secret = corp_secret or os.getenv('WECHAT_CORP_SECRET')
        self.agent_id = agent_id or int(os.getenv('WECHAT_AGENT_ID') or 0)
        self._client = client
        self._owns_client = client is None
        self._access_token: str | None = None
        self._access_token_expires_at = 0.0

    async def get_access_token(self, *, force: bool = False) -> str:
        """Return a cached WeChat Work access token, refreshing when needed."""
        if not self.corp_id or not self.corp_secret:
            raise ValueError(
                'WECHAT_CORP_ID and WECHAT_CORP_SECRET are required.',
            )
        if (
            not force
            and self._access_token is not None
            and self._access_token_expires_at
            > asyncio.get_running_loop().time()
        ):
            return self._access_token

        client = self._http_client()
        response = await client.get(
            f"{_WECHAT_API_ROOT}/gettoken",
            params={
                'corpid': self.corp_id,
                'corpsecret': self.corp_secret,
            },
        )
        response.raise_for_status()
        payload = response.json()
        token = payload.get('access_token')
        if not isinstance(token, str) or not token:
            raise ValueError('WeChat Work did not return an access token.')
        expires_in = payload.get('expires_in', 7200)
        seconds = expires_in if isinstance(expires_in, int) else 7200
        self._access_token = token
        self._access_token_expires_at = (
            asyncio.get_running_loop().time()
            + max(
                0,
                seconds - _TOKEN_REFRESH_MARGIN_SECONDS,
            )
        )
        return token

    async def send_notification(
        self,
        user_id: str,
        message: str,
        image: np.ndarray | None = None,
    ) -> dict[str, object]:
        """Send one text or image notification and return the WeChat
        payload."""
        token = await self.get_access_token()
        if image is None:
            payload: dict[str, object] = {
                'touser': user_id,
                'msgtype': 'text',
                'agentid': self.agent_id,
                'text': {'content': message},
                'safe': 0,
            }
        else:
            payload = {
                'touser': user_id,
                'msgtype': 'image',
                'agentid': self.agent_id,
                'image': {
                    'media_id': await self.upload_media(image, token=token),
                },
                'safe': 0,
            }
        client = self._http_client()
        response = await client.post(
            f"{_WECHAT_API_ROOT}/message/send",
            params={'access_token': token},
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    async def upload_media(
        self,
        image: np.ndarray,
        *,
        token: str | None = None,
    ) -> str:
        """Upload an image and return the temporary WeChat media ID."""
        access_token = token or await self.get_access_token()
        client = self._http_client()
        response = await client.post(
            f"{_WECHAT_API_ROOT}/media/upload",
            params={'access_token': access_token, 'type': 'image'},
            files={'media': ('image.png', _png_bytes(image), 'image/png')},
        )
        response.raise_for_status()
        media_id = response.json().get('media_id')
        if not isinstance(media_id, str) or not media_id:
            raise ValueError('WeChat Work did not return a media ID.')
        return media_id

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
    """Encode an RGB image as a PNG byte string."""
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format='PNG')
    return buffer.getvalue()
