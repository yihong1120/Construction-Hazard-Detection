from __future__ import annotations

import os
import re
import time
from collections.abc import Mapping
from typing import cast
from urllib.parse import parse_qs
from urllib.parse import parse_qsl
from urllib.parse import quote
from urllib.parse import unquote
from urllib.parse import urlencode
from urllib.parse import urljoin
from urllib.parse import urlsplit

import httpx
from fastapi import HTTPException
from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.models import USER_STATUS_ACTIVE
from examples.auth.session_store import media_session_cameras
from examples.auth.user_service import load_user_access_context
from examples.streaming_web.media_paths import build_media_path
from examples.streaming_web.media_paths import build_preview_media_path
from examples.streaming_web.media_paths import encode_media_segment
from examples.streaming_web.media_paths import parse_annotated_media_path

MEDIA_PUBLISHER_IDLE_GRACE_SECONDS = max(
    30,
    int(os.getenv('MEDIA_PUBLISHER_IDLE_GRACE_SECONDS', '180')),
)
MEDIA_INTERNAL_HLS_BASE_URL = os.getenv(
    'MEDIA_INTERNAL_HLS_BASE_URL',
    'http://127.0.0.1:8890',
).rstrip('/')
MEDIA_INTERNAL_HLS_TIMEOUT_SECONDS = float(
    os.getenv('MEDIA_INTERNAL_HLS_TIMEOUT_SECONDS', '10'),
)


def media_auth_401(detail: str) -> HTTPException:
    """Build a machine-readable media-authorisation failure."""
    return HTTPException(
        status_code=401,
        detail=detail,
        headers={
            'WWW-Authenticate': f'Bearer error="{detail}"',
            'X-Media-Auth-Error': detail,
        },
    )


async def authorise_label_access(
    credentials: JwtAuthorizationCredentials,
    db: AsyncSession,
    label: str,
) -> None:
    """Ensure the authenticated user may view the requested site label."""
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    user, user_site_names, user_role = await load_user_access_context(
        db,
        username,
    )
    if getattr(user, 'status', None) != USER_STATUS_ACTIVE:
        raise media_auth_401('inactive_user')
    if user_role != 'super_admin' and label not in user_site_names:
        raise HTTPException(status_code=403, detail='Access denied')


def extract_opaque_media_token(request: Request) -> str:
    """Extract only an opaque media capability from a proxy request."""
    for query_name in ('mt', 'media_token'):
        query_token = request.query_params.get(query_name)
        if query_token:
            return query_token

    for header_name in ('x-original-uri', 'x-forwarded-uri'):
        original_uri = request.headers.get(header_name)
        if not original_uri:
            continue
        original_query = parse_qs(urlsplit(original_uri).query)
        for query_name in ('mt', 'media_token'):
            original_tokens = original_query.get(query_name) or []
            if original_tokens and original_tokens[0]:
                return original_tokens[0]
    return ''


def media_session_demand_ttl(session: Mapping[str, object]) -> int:
    """Derive a bounded producer lease from capability expiry."""
    remaining = int(cast(str | int, session['expires_at'])) - int(time.time())
    return max(MEDIA_PUBLISHER_IDLE_GRACE_SECONDS, remaining)


def split_hls_playlist_query(query: str) -> tuple[str, str]:
    """Separate capability parameters from upstream HLS query parameters."""
    auth_items: list[tuple[str, str]] = []
    media_items: list[tuple[str, str]] = []
    for key, value in parse_qsl(query, keep_blank_values=True):
        if key in {'mt', 'media_token'}:
            auth_items.append((key, value))
        else:
            media_items.append((key, value))
    return urlencode(auth_items), urlencode(media_items)


def append_query(url: str, query: str) -> str:
    """Append a non-empty query string to a URL."""
    if not query:
        return url
    separator = '&' if '?' in url else '?'
    return f"{url}{separator}{query}"


def rewrite_hls_uri(uri: str, media_path: str, auth_query: str) -> str:
    """Rewrite one HLS URI so child reads retain media authorisation."""
    if not auth_query:
        return uri
    public_base_path = f"/hazard/media/{quote(media_path, safe='')}/"
    parts = urlsplit(uri)
    path = parts.path
    if path.startswith('/hazard/media/'):
        rewritten = path
    else:
        path_parts = [unquote(part) for part in path.split('/') if part]
        if path_parts and path_parts[0] == media_path:
            rewritten = public_base_path + '/'.join(
                quote(part, safe='') for part in path_parts[1:]
            )
        elif parts.scheme or parts.netloc or path.startswith('/'):
            filename = path_parts[-1] if path_parts else ''
            rewritten = urljoin(public_base_path, quote(filename, safe=''))
        else:
            rewritten = urljoin(public_base_path, uri)
    if parts.query and '?' not in rewritten:
        rewritten = append_query(rewritten, parts.query)
    return append_query(rewritten, auth_query)


def rewrite_hls_playlist_media_urls(
    playlist: str,
    media_path: str,
    auth_query: str,
) -> str:
    """Append opaque authorisation to all playlist URI values."""
    uri_attr = re.compile(r'URI="([^"]+)"')

    def rewrite_uri_attr(match: re.Match[str]) -> str:
        """Perform rewrite uri attr.

        Args:
            match: Value used by this callable.

        Returns:
            The callable result.
        """
        return (
            f'URI="{rewrite_hls_uri(match.group(1), media_path, auth_query)}"'
        )

    rewritten_lines: list[str] = []
    for line in playlist.splitlines():
        if not line:
            rewritten_lines.append(line)
        elif line.startswith('#'):
            rewritten_lines.append(uri_attr.sub(rewrite_uri_attr, line))
        else:
            rewritten_lines.append(
                rewrite_hls_uri(
                    line,
                    media_path,
                    auth_query,
                ),
            )
    suffix = '\n' if playlist.endswith('\n') else ''
    return '\n'.join(rewritten_lines) + suffix


def media_hls_session_cookie(
    media_path: str,
    session_value: str | None,
) -> str | None:
    """Build a secure, stream-scoped browser cookie for MediaMTX HLS."""
    if session_value is None:
        return None
    if not re.fullmatch(r'[A-Za-z0-9._~-]+', session_value):
        raise HTTPException(status_code=502, detail='invalid_hls_session')
    public_path = f"/hazard/media/{quote(media_path, safe='')}/"
    return (
        f"hlsSession={session_value}; Path={public_path}; "
        'Secure; HttpOnly; SameSite=None; Partitioned'
    )


async def fetch_internal_hls_playlist(
    media_path: str,
    media_query: str,
    http_client: httpx.AsyncClient | None = None,
) -> tuple[str, str | None]:
    """Fetch an HLS playlist and return its scoped MediaMTX session cookie."""
    playlist_path = f"{quote(media_path, safe='')}/index.m3u8"
    url = append_query(
        f"{MEDIA_INTERNAL_HLS_BASE_URL}/{playlist_path}", media_query,
    )
    try:
        if http_client is not None:
            response = await http_client.get(url)
        else:
            async with httpx.AsyncClient(
                timeout=MEDIA_INTERNAL_HLS_TIMEOUT_SECONDS,
                follow_redirects=True,
            ) as client:
                response = await client.get(url)
    except (httpx.TimeoutException, httpx.NetworkError) as exc:
        raise HTTPException(
            status_code=502,
            detail='media_playlist_unavailable',
        ) from exc
    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail='media_playlist_unavailable',
        )
    if not response.text.strip():
        raise HTTPException(status_code=503, detail='media_playlist_not_ready')
    return response.text, media_hls_session_cookie(
        media_path,
        response.cookies.get('hlsSession'),
    )


def extract_media_path_from_uri(uri: str) -> str:
    """Extract the decoded MediaMTX stream path from an external URI."""
    segments = [
        unquote(part) for part in uri.split('?', 1)[0].split('/') if part
    ]
    try:
        path_index = segments.index('media') + 1
    except ValueError:
        return ''
    if path_index < len(segments) and segments[path_index] == 'webrtc':
        path_index += 1
    return segments[path_index] if path_index < len(segments) else ''


def media_path_matches_site(media_path: str, site_name: str) -> bool:
    """Check whether a MediaMTX path belongs to the supplied site."""
    return media_path.startswith(f"hazard_{encode_media_segment(site_name)}_")


def opaque_media_session_allows_path(
    session: Mapping[str, object],
    media_path: str,
) -> bool:
    """Enforce the site, camera, profile, and quality scope of a capability."""
    site = str(session.get('site') or '')
    quality = session.get('quality')
    if quality not in {'detail', 'preview'}:
        return False
    base_paths = {
        build_media_path(site, camera)
        for camera in media_session_cameras(session)
    }
    if quality == 'preview':
        base_paths = {build_preview_media_path(path) for path in base_paths}
    if not base_paths:
        return False
    if session.get('profile') == 'clean':
        return media_path in base_paths
    if session.get('profile') == 'overlay':
        parsed = parse_annotated_media_path(media_path)
        return parsed is not None and parsed[0] in base_paths
    return False
