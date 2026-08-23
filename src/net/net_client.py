from __future__ import annotations

import asyncio
import logging
import posixpath
from collections.abc import Callable
from collections.abc import Mapping
from urllib.parse import urlparse
from urllib.parse import urlunparse

import aiohttp
import httpx

from src.async_http_client import AsyncHttpClientOwner
from src.auth_tokens import TokenManager


class NetClient(AsyncHttpClientOwner):
    """High-level HTTP and WebSocket client."""

    def __init__(
        self,
        base_url: str,
        token_manager: TokenManager,
        timeout: int = 10,
        reconnect_backoff: float = 1.5,
        ws_heartbeat: int = 30,
        ws_send_timeout: float = 10.0,
        ws_recv_timeout: float = 10.0,
        ws_connect_attempts: int = 3,
        auth_required: bool = True,
    ) -> None:
        """Initialise the NetClient with the given parameters.

        Args:
            base_url (str): The base URL for the client.
            token_manager (TokenManager): The token manager for handling
                authentication.
            timeout (int, optional): The timeout for HTTP requests.
                Defaults to 10.
            reconnect_backoff (float, optional): The backoff factor for
                reconnection attempts. Defaults to 1.5.
            ws_heartbeat (int, optional): The heartbeat interval for
                WebSocket connections. Defaults to 30.
            ws_send_timeout (float, optional): The timeout for sending
                messages over WebSocket. Defaults to 10.0.
            ws_recv_timeout (float, optional): The timeout for receiving
                messages over WebSocket. Defaults to 10.0.
            ws_connect_attempts (int, optional): The number of attempts to
                connect the WebSocket. Defaults to 3.
            auth_required (bool, optional): Whether HTTP/WS requests should
                acquire and refresh JWT credentials. Defaults to True.
        """
        self.base_url: str = base_url.rstrip('/')
        self.token_manager: TokenManager = token_manager
        super().__init__(timeout)
        self.reconnect_backoff: float = reconnect_backoff
        self.ws_heartbeat: int = ws_heartbeat
        self.ws_send_timeout: float = ws_send_timeout
        self.ws_recv_timeout: float = ws_recv_timeout
        self.ws_connect_attempts: int = ws_connect_attempts
        self.auth_required: bool = auth_required

        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._ws_path: str | None = None
        # Logger dedicated to this client (informational + debug on retries).
        self._log: logging.Logger = logging.getLogger(__name__)

    def build_http_url(self, path: str) -> str:
        """Build a full HTTP URL from the base URL and a relative path.

        Args:
            path (str): Relative path (with or without a leading slash).

        Returns:
            str: The absolute HTTP(S) URL.
        """
        if not path.startswith('/'):
            path = '/' + path
        return f"{self.base_url}{path}"

    def build_ws_url(self, path: str) -> str:
        """Build a full WebSocket URL derived from the base URL and path.

        Args:
            path (str): Relative path (with or without a leading slash).

        Returns:
            str: The absolute WebSocket URL.
        """
        if not path.startswith('/'):
            path = '/' + path
        u = urlparse(self.base_url)
        scheme = 'wss' if u.scheme in ('https', 'wss') else 'ws'
        full_path = posixpath.join(u.path.rstrip('/') or '/', path.lstrip('/'))
        return urlunparse((scheme, u.netloc, full_path, '', '', ''))

    async def auth_headers(self) -> dict[str, str]:
        """Compute authorisation headers using the token manager.

        Returns:
            dict[str, str]: A dictionary with standard headers including
                ``Authorization`` and a user agent string.

        Raises:
            Exception: If authentication and token retrieval both fail.
        """
        if not self.auth_required:
            return {'User-Agent': 'ConstructionHazardDetection/1.0'}
        try:
            access_token = await self.token_manager.get_valid_token()
        except Exception:
            await self.token_manager.authenticate(force=True)
            access_token = await self.token_manager.get_valid_token()
        return {
            'Authorization': f"Bearer {access_token}",
            'User-Agent': 'ConstructionHazardDetection/1.0',
        }

    async def http_post(
        self,
        path: str,
        data: Mapping[str, object],
        files: dict[str, tuple[str, bytes, str]] | None = None,
        max_retries: int = 3,
    ) -> dict[str, object]:
        """Perform an HTTP POST request with retries and token refresh
        handling.

        Args:
            path (str): Relative HTTP path to post to.
            data (Mapping[str, object]): Form/field data payload.
            files (dict[str, tuple[str, bytes, str]], optional): Optional
                mapping of file fields for multipart uploads.
            max_retries (int, optional): Max tries including the initial
                attempt. Defaults to 3.

        Returns:
            dict[str, object]: Parsed JSON response as a dictionary.

        Raises:
            httpx.HTTPError: Networking or HTTP errors after retries.
            RuntimeError: If retry budget is exhausted without a response.
        """
        headers = await self.auth_headers()
        if max_retries <= 0:
            raise RuntimeError('HTTP POST retries exhausted')

        client = await self._get_client()
        attempt = 0
        while True:
            try:
                resp = await client.post(
                    self.build_http_url(path),
                    data=data,
                    files=files,
                    headers=headers,
                )
                resp.raise_for_status()
                return resp.json()
            except httpx.ConnectTimeout:
                if attempt == max_retries - 1:
                    raise
                await self._wait_for_http_retry(attempt)
                attempt += 1
            except httpx.HTTPStatusError as exc:
                if self.auth_required and exc.response.status_code in (
                    401,
                    403,
                ):
                    await self.token_manager.refresh_token()
                    headers = await self.auth_headers()
                    if attempt == max_retries - 1:
                        raise
                    await self._wait_for_http_retry(attempt)
                    attempt += 1
                    continue
                raise

    async def _wait_for_http_retry(self, attempt: int) -> None:
        """Apply the bounded HTTP retry backoff for a failed attempt."""
        delay = min(self.reconnect_backoff * (attempt + 1), 10.0)
        await asyncio.sleep(delay)

    async def ensure_ws(
        self,
        ws_path: str,
        headers: dict[str, str] | None = None,
    ) -> aiohttp.ClientWebSocketResponse:
        """Ensure a live WebSocket is connected for a specific path.

        Args:
            ws_path (str): WebSocket path to connect to.
            headers (dict[str, str], optional): Optional extra headers
                merged with authorisation headers.

        Returns:
            aiohttp.ClientWebSocketResponse: A connected WebSocket response
                object.
        """
        # Reuse if already connected to the same path
        if self._ws and not self._ws.closed and self._ws_path == ws_path:
            return self._ws

        # Recreate session and connect
        await self._open_new_session()
        self._ws = await self._connect_with_retries(ws_path, headers)
        self._ws_path = ws_path
        return self._ws

    async def close(self) -> None:
        """Close the HTTP pool and any WebSocket resources."""
        try:
            await self._close_websocket_resources()
        finally:
            await super().close()

    async def _close_websocket_resources(self) -> None:
        """Close the WebSocket transport without discarding the HTTP pool."""
        try:
            if self._ws is not None and not self._ws.closed:
                await self._ws.close()
        finally:
            self._ws = None
            try:
                if self._session is not None and not self._session.closed:
                    await self._session.close()
            finally:
                self._session = None

    async def _open_new_session(self) -> None:
        """(Re)initialise the aiohttp session with configured timeouts."""
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=self.timeout,
                connect=self.timeout,
            ),
        )

    async def _connect_with_retries(
        self,
        ws_path: str,
        headers: dict[str, str] | None,
    ) -> aiohttp.ClientWebSocketResponse:
        """Connect to a WebSocket URL with retries and token refresh on 401.

        Args:
            ws_path (str): WebSocket path to connect.
            headers (dict[str, str], optional): Optional extra headers.

        Returns:
            aiohttp.ClientWebSocketResponse: A connected WebSocket response
                object.

        Raises:
            ConnectionError: If all connection attempts fail.
        """
        attempts = self.ws_connect_attempts
        ws_url = self.build_ws_url(ws_path)
        for attempt in range(1, attempts + 1):
            try:
                headers_final = await self._make_ws_headers(headers)
                self._log.info(
                    '[WS] Connecting to %s (attempt %d)...',
                    ws_url,
                    attempt,
                )
                session = self._session
                # Ensure session is not None after _open_new_session
                assert session is not None
                ws = await session.ws_connect(
                    ws_url,
                    headers=headers_final,
                    heartbeat=self.ws_heartbeat,
                    compress=0,
                )
                self._log.info('[WS] Connection established')
                return ws
            except aiohttp.ClientResponseError as e:
                self._log.warning(
                    '[WS] connection response error from %s: status=%s %s',
                    ws_url,
                    e.status,
                    e.message,
                )
                if self.auth_required and e.status == 401:
                    await self.token_manager.refresh_token()
                elif e.status in (403, 404):
                    raise
            except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
                self._log.warning(
                    '[WS] connection attempt %d/%d to %s failed: %s',
                    attempt,
                    attempts,
                    ws_url,
                    e,
                )
            if attempt < attempts:
                delay = min(self.reconnect_backoff * attempt, 10.0)
                await asyncio.sleep(delay)
        raise ConnectionError('WebSocket connection retries exhausted')

    async def _make_ws_headers(
        self,
        extra: dict[str, str] | None,
    ) -> dict[str, str]:
        """Merge base authorisation headers with any provided extras.

        Args:
            extra (dict[str, str], optional): Additional headers to merge.

        Returns:
            dict[str, str]: Final headers including authorisation and
                extras.
        """
        headers_final = await self.auth_headers()
        if extra:
            headers_final.update(extra)
        return headers_final

    async def ws_send_and_receive(
        self,
        ws_path: str,
        payload: bytes,
        headers: dict[str, str] | None = None,
        ignore_messages: Callable[[object], bool] | None = None,
    ) -> object | None:
        """Send bytes over WS and await a JSON response.

        Args:
            ws_path: WS path for ensure/connect.
            payload: Bytes to send over the socket.
            headers: Optional extra headers.
            ignore_messages: Optional predicate used to skip control
                messages and continue waiting for the actual response
                payload.

        Returns:
            Parsed JSON payload, or ``None`` if sending/receiving failed.
        """
        ws = await self.ensure_ws(ws_path, headers=headers)
        if ws.closed:
            reconnected_ws = await self._reconnect_ws(ws_path, headers)
            if reconnected_ws is None:
                return None
            ws = reconnected_ws

        if not await self._send_ws_bytes(ws, payload):
            return None

        return await self._receive_ws_json_until(ws, ignore_messages)

    async def _reconnect_ws(
        self,
        ws_path: str,
        headers: dict[str, str] | None,
    ) -> aiohttp.ClientWebSocketResponse | None:
        """Attempt to reconnect the WS by closing and ensuring again.

        Args:
            ws_path (str): The WebSocket path to reconnect to.
            headers (dict[str, str] | None): Optional headers to include.

        Returns:
            aiohttp.ClientWebSocketResponse | None:
                The reconnected WebSocket or None on failure.
        """
        await self._close_websocket_resources()
        try:
            return await self.ensure_ws(ws_path, headers=headers)
        except Exception:
            return None

    async def _send_ws_bytes(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        payload: bytes,
    ) -> bool:
        """Send a bytes payload over WS with a timeout.

        Args:
            ws (aiohttp.ClientWebSocketResponse):
                The WebSocket connection to send data over.
            payload (bytes):
                The bytes payload to send.
        """
        try:
            await asyncio.wait_for(
                ws.send_bytes(payload),
                timeout=self.ws_send_timeout,
            )
            return True
        except (
            asyncio.TimeoutError,
            aiohttp.ClientConnectionError,
            ConnectionResetError,
            RuntimeError,
        ) as e:
            # RuntimeError: Cannot write to closing transport
            self._log.debug('[WS] send failed: %s', e)
            await self.close()
            return False

    async def _receive_ws_json_until(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        ignore_messages: Callable[[object], bool] | None = None,
    ) -> object | None:
        """Receive JSON messages until one should be returned to the caller."""
        while True:
            data = await self._receive_ws_json(ws)
            if data is None:
                return None
            if ignore_messages and ignore_messages(data):
                self._log.debug(
                    '[WS] skipping control message on %s',
                    self._ws_path,
                )
                continue
            return data

    async def _receive_ws_json(
        self,
        ws: aiohttp.ClientWebSocketResponse,
    ) -> object | None:
        """Receive a WS message and parse JSON payloads.

        Args:
          ws: The connected WS instance to receive from.

        Returns:
          Parsed JSON payload, or ``None`` on failure/end.
        """
        try:
            resp = await asyncio.wait_for(
                ws.receive(),
                timeout=self.ws_recv_timeout,
            )
        except (asyncio.TimeoutError, aiohttp.ClientConnectionError) as e:
            self._log.debug('[WS] receive failed: %s', e)
            await self.close()
            return None

        if resp.type == aiohttp.WSMsgType.TEXT:
            import json as _json

            return _json.loads(resp.data)

        if resp.type == aiohttp.WSMsgType.BINARY:
            import json as _json

            return _json.loads(resp.data.decode('utf-8'))

        if resp.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING):
            self._log.warning(
                '[WS] connection closed on %s: code=%s extra=%s',
                self._ws_path,
                resp.data,
                getattr(resp, 'extra', ''),
            )
            # 1008: policy violation/auth error
            if self.auth_required and resp.data == 1008:
                await self.token_manager.refresh_token()
            await self.close()
            return None

        if resp.type == aiohttp.WSMsgType.ERROR:
            self._log.warning(
                '[WS] connection error on %s: %s',
                self._ws_path,
                ws.exception(),
            )
            await self.close()
            return None

        return None
