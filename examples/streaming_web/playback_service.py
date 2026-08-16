from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import time
from collections.abc import Mapping
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import cast
from urllib.parse import parse_qs
from urllib.parse import parse_qsl
from urllib.parse import quote
from urllib.parse import unquote
from urllib.parse import urlencode
from urllib.parse import urljoin
from urllib.parse import urlsplit

import httpx
import redis.asyncio as redis
from fastapi import HTTPException
from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.models import USER_STATUS_ACTIVE
from examples.auth.session_store import media_session_cameras
from examples.auth.user_service import load_user_access_context
from examples.local_notification_server.lang_config import LANGUAGES
from examples.streaming_web.media_paths import build_annotated_media_path
from examples.streaming_web.media_paths import build_clean_demand_key
from examples.streaming_web.media_paths import build_media_hls_url
from examples.streaming_web.media_paths import build_media_path
from examples.streaming_web.media_paths import build_overlay_demand_key
from examples.streaming_web.media_paths import build_overlay_ready_key
from examples.streaming_web.media_paths import build_preview_media_path
from examples.streaming_web.media_paths import decode_media_segment
from examples.streaming_web.media_paths import encode_media_segment
from examples.streaming_web.media_paths import OVERLAY_DEMAND_PREFIX
from examples.streaming_web.media_paths import parse_annotated_media_path
from examples.streaming_web.metadata_keys import encode_stream_id
from examples.streaming_web.overlay_renderer import CLASS_LABELS
from examples.streaming_web.overlay_renderer import LANGUAGE_ALIASES
from examples.streaming_web.overlay_renderer import normalise_label_language
from examples.streaming_web.overlay_renderer import SUPPORTED_LABEL_LANGUAGES
from examples.streaming_web.overlay_renderer import WARNING_LABELS
from examples.streaming_web.schemas import CleanPlaybackSession
from examples.streaming_web.schemas import OverlayLanguageInfo
from examples.streaming_web.schemas import OverlayLanguageListResponse
from examples.streaming_web.schemas import OverlayPlaybackSession
from examples.streaming_web.schemas import PlaybackProfile
from examples.streaming_web.schemas import PlaybackRendition
from examples.streaming_web.schemas import PlaybackSession
from examples.streaming_web.schemas import PlaybackSessionResponse
from examples.streaming_web.schemas import PlaybackSessionState


def _allowed_overlay_languages() -> tuple[str, ...]:
    """Return the configured overlay languages enabled for playback.

    Returns:
        Ordered unique canonical language codes.

    Raises:
        ValueError: If configuration is empty or names unsupported languages.
    """
    configured = tuple(
        dict.fromkeys(
            language.strip()
            for language in os.getenv(
                'MEDIA_OVERLAY_ALLOWED_LANGUAGES',
                ','.join(SUPPORTED_LABEL_LANGUAGES),
            ).split(',')
            if language.strip()
        ),
    )
    if not configured or set(configured) - set(SUPPORTED_LABEL_LANGUAGES):
        raise ValueError(
            'MEDIA_OVERLAY_ALLOWED_LANGUAGES must contain supported codes',
        )
    return configured


OVERLAY_LANGUAGE_DETAILS: dict[str, dict[str, str | list[str]]] = {
    'zh-TW': {
        'name': 'Traditional Chinese',
        'native_name': '繁體中文',
        'aliases': ['zh', 'zh-Hant', 'zh_TW', 'zh-HK'],
    },
    'en': {
        'name': 'English',
        'native_name': 'English',
        'aliases': ['en-US', 'en-GB'],
    },
    'zh-CN': {
        'name': 'Simplified Chinese',
        'native_name': '简体中文',
        'aliases': ['zh-Hans', 'zh_CN', 'zh-SG'],
    },
    'ja': {
        'name': 'Japanese',
        'native_name': '日本語',
        'aliases': ['jp', 'ja-JP'],
    },
    'vi': {
        'name': 'Vietnamese',
        'native_name': 'Tiếng Việt',
        'aliases': ['vi-VN'],
    },
    'id': {
        'name': 'Indonesian',
        'native_name': 'Bahasa Indonesia',
        'aliases': ['id-ID'],
    },
    'fr': {
        'name': 'French',
        'native_name': 'Français',
        'aliases': ['fr-FR', 'fr-CA'],
    },
    'th': {
        'name': 'Thai',
        'native_name': 'ไทย',
        'aliases': ['th-TH'],
    },
}
OVERLAY_TO_NOTIFICATION_LANGUAGE: dict[str, str] = {
    'en': 'en-GB',
    'zh-TW': 'zh-TW',
    'zh-CN': 'zh-CN',
    'ja': 'ja-JP',
    'vi': 'vi-VN',
    'id': 'id-ID',
    'fr': 'fr-FR',
    'th': 'th-TH',
}


def _default_overlay_language() -> str:
    """Return the configured default overlay language.

    Returns:
        Canonical language code selected by default.

    Raises:
        ValueError: If the configured default is not allowed.
    """
    language = os.getenv('MEDIA_DEFAULT_OVERLAY_LANGUAGE', 'zh-TW')
    if language not in _allowed_overlay_languages():
        raise ValueError(
            'MEDIA_DEFAULT_OVERLAY_LANGUAGE must be an allowed language',
        )
    return language


def _language_alias_map() -> dict[str, str]:
    """Build normalised language aliases for API consumers.

    Returns:
        Mapping from recognised aliases to canonical overlay language codes.
    """
    aliases = dict(LANGUAGE_ALIASES)
    for code, details in OVERLAY_LANGUAGE_DETAILS.items():
        aliases[code] = code
        for alias in details['aliases']:
            aliases[alias] = code
            aliases[alias.lower()] = code
    return aliases


def _notification_language_code(code: str) -> str:
    """Map an overlay language to its notification language code.

    Args:
        code: Canonical overlay language code.

    Returns:
        Corresponding notification-template language code.
    """
    return OVERLAY_TO_NOTIFICATION_LANGUAGE[code]


def _overlay_language_options(
    allowed_languages: tuple[str, ...] | None = None,
) -> list[OverlayLanguageInfo]:
    """Build language-option metadata for the frontend.

    Args:
        allowed_languages: Optional prevalidated ordered language codes.

    Returns:
        Display, label, and notification details for each allowed language.
    """
    languages: list[OverlayLanguageInfo] = []
    codes = (
        _allowed_overlay_languages()
        if allowed_languages is None
        else allowed_languages
    )
    for code in codes:
        details = OVERLAY_LANGUAGE_DETAILS[code]
        notification_code = _notification_language_code(code)
        languages.append(
            OverlayLanguageInfo(
                code=code,
                notification_code=notification_code,
                display_name=details['name'],
                native_name=details['native_name'],
                is_default=code == _default_overlay_language(),
                class_labels=CLASS_LABELS[code],
                warning_labels=WARNING_LABELS[code],
                notification_templates=LANGUAGES[notification_code],
            ),
        )
    return languages


OVERLAY_DEMAND_TTL_SECONDS = int(
    os.getenv('MEDIA_OVERLAY_DEMAND_TTL_SECONDS', '90'),
)
MEDIA_PUBLISHER_IDLE_GRACE_SECONDS = max(
    30,
    int(os.getenv('MEDIA_PUBLISHER_IDLE_GRACE_SECONDS', '180')),
)
CLEAN_DEMAND_TTL_SECONDS = int(
    os.getenv(
        'MEDIA_CLEAN_DEMAND_TTL_SECONDS',
        str(OVERLAY_DEMAND_TTL_SECONDS),
    ),
)
OVERLAY_MAX_ACTIVE_LANGUAGES = max(
    1,
    int(os.getenv('MEDIA_OVERLAY_MAX_ACTIVE_LANGUAGES_PER_STREAM', '5')),
)
STREAM_PLAYBACK_SESSION_TTL_SECONDS = int(
    os.getenv(
        'STREAM_PLAYBACK_SESSION_TTL_SECONDS',
        str(75 * 60),
    ),
)
STREAM_PLAYBACK_SESSION_PREFIX = 'stream_playback_session'
STREAM_PLAYBACK_MEDIA_SESSION_PREFIX = 'stream_playback_media_session'
STREAM_PLAYBACK_PUBLIC_BASE_PATH = os.getenv(
    'STREAM_PLAYBACK_PUBLIC_BASE_PATH',
    '/hazard/api/stream-playback',
).rstrip('/')
STREAM_PLAYBACK_STARTUP_WAIT_SECONDS = max(
    0.0,
    float(os.getenv('STREAM_PLAYBACK_STARTUP_WAIT_SECONDS', '3.0')),
)
MEDIA_INTERNAL_HLS_BASE_URL = os.getenv(
    'MEDIA_INTERNAL_HLS_BASE_URL',
    'http://127.0.0.1:8890',
).rstrip('/')
MEDIA_INTERNAL_HLS_TIMEOUT_SECONDS = float(
    os.getenv('MEDIA_INTERNAL_HLS_TIMEOUT_SECONDS', '10'),
)


async def _authorise_label_access(
    credentials: JwtAuthorizationCredentials,
    db: AsyncSession,
    label: str,
) -> None:
    """Ensure the authenticated user may view a site label.

    Args:
        credentials: Verified JWT credentials for the caller.
        db: Database session used to load the access context.
        label: Requested site label.

    Raises:
        HTTPException: If the token, user status, or site access is invalid.
    """
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    user, user_site_names, user_role = await load_user_access_context(
        db,
        username,
    )
    if user.status != USER_STATUS_ACTIVE:
        raise _media_auth_401('inactive_user')
    if user_role != 'super_admin' and label not in user_site_names:
        raise HTTPException(status_code=403, detail='Access denied')


def _extract_opaque_media_token(request: Request) -> str:
    """Extract the opaque media capability without accepting a main JWT.

    Args:
        request: Incoming media-proxy authorisation request.

    Returns:
        Capability token from the original request URI, or an empty string.
    """
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


def _media_auth_401(detail: str) -> HTTPException:
    """Build a media-authorisation error with a machine-readable reason.

    Args:
        detail: Stable failure code exposed to the media proxy.

    Returns:
        Configured HTTP 401 exception with challenge headers.
    """
    return HTTPException(
        status_code=401,
        detail=detail,
        headers={
            'WWW-Authenticate': f'Bearer error="{detail}"',
            'X-Media-Auth-Error': detail,
        },
    )


def _media_session_demand_ttl(session: dict[str, object]) -> int:
    """Derive a bounded producer lease from capability expiry.

    Args:
        session: Trusted opaque media session payload.

    Returns:
        Lease duration that outlives the capability by the idle grace period.
    """
    remaining = int(cast(str | int, session['expires_at'])) - int(time.time())
    return max(MEDIA_PUBLISHER_IDLE_GRACE_SECONDS, remaining)


def _split_hls_playlist_query(query: str) -> tuple[str, str]:
    """Split capability parameters from upstream HLS query parameters.

    Args:
        query: Raw incoming playlist query string.

    Returns:
        Encoded capability query and encoded MediaMTX-only query.
    """
    auth_items: list[tuple[str, str]] = []
    media_items: list[tuple[str, str]] = []
    for key, value in parse_qsl(query, keep_blank_values=True):
        if key in {'mt', 'media_token'}:
            auth_items.append((key, value))
        else:
            media_items.append((key, value))
    return urlencode(auth_items), urlencode(media_items)


def _append_query(url: str, query: str) -> str:
    """Append a non-empty query string to a URL.

    Args:
        url: URL that may already contain query parameters.
        query: Encoded query string to append.

    Returns:
        URL containing all query parameters.
    """
    if not query:
        return url
    separator = '&' if '?' in url else '?'
    return f'{url}{separator}{query}'


def _rewrite_hls_uri(uri: str, media_path: str, auth_query: str) -> str:
    """Rewrite one playlist URI so fragment reads retain authorisation.

    Args:
        uri: Relative or absolute URI found in an HLS playlist.
        media_path: Authorised MediaMTX stream path.
        auth_query: Encoded opaque media-capability parameters.

    Returns:
        Public proxy URI carrying the original and authorisation queries.
    """
    if not auth_query:
        return uri
    public_base_path = f'/hazard/media/{quote(media_path, safe="")}/'
    parts = urlsplit(uri)
    path = parts.path

    if path.startswith('/hazard/media/'):
        rewritten = path
    else:
        path_parts = [unquote(part) for part in path.split('/') if part]
        if path_parts and path_parts[0] == media_path:
            suffix_parts = path_parts[1:]
            rewritten = public_base_path + '/'.join(
                quote(part, safe='')
                for part in suffix_parts
            )
        elif parts.scheme or parts.netloc or path.startswith('/'):
            filename = path_parts[-1] if path_parts else ''
            rewritten = urljoin(public_base_path, quote(filename, safe=''))
        else:
            rewritten = urljoin(public_base_path, uri)

    if parts.query and '?' not in rewritten:
        rewritten = _append_query(rewritten, parts.query)
    return _append_query(rewritten, auth_query)


def _rewrite_hls_playlist_media_urls(
    playlist: str,
    media_path: str,
    auth_query: str,
) -> str:
    """Append an opaque media capability to every M3U8 media URI.

    Args:
        playlist: Complete M3U8 playlist body.
        media_path: Authorised MediaMTX stream path.
        auth_query: Encoded opaque media-capability parameters.

    Returns:
        Playlist with both URI attributes and media lines rewritten.
    """
    rewritten_lines: list[str] = []
    uri_attr = re.compile(r'URI="([^"]+)"')

    def rewrite_uri_attr(match: re.Match[str]) -> str:
        """Rewrite one quoted URI attribute inside an HLS directive.

        Args:
            match: Regular-expression match containing the original URI.

        Returns:
            Replacement URI attribute carrying opaque media authorisation.
        """
        uri = _rewrite_hls_uri(
            match.group(1),
            media_path,
            auth_query,
        )
        return f'URI="{uri}"'

    for line in playlist.splitlines():
        if not line:
            rewritten_lines.append(line)
            continue
        if line.startswith('#'):
            rewritten_lines.append(
                uri_attr.sub(
                    rewrite_uri_attr,
                    line,
                ),
            )
            continue
        rewritten_lines.append(_rewrite_hls_uri(line, media_path, auth_query))

    suffix = '\n' if playlist.endswith('\n') else ''
    return '\n'.join(rewritten_lines) + suffix


def _media_hls_session_cookie(
    media_path: str,
    session_value: str | None,
) -> str | None:
    """Build the browser cookie required by MediaMTX HLS child requests.

    Args:
        media_path: Authorised MediaMTX stream path.
        session_value: Session value returned by the upstream playlist.

    Returns:
        Complete scoped ``Set-Cookie`` value, or ``None`` when absent.

    Raises:
        HTTPException: If the upstream session value has unsafe characters.
    """
    if session_value is None:
        return None
    if not re.fullmatch(r'[A-Za-z0-9._~-]+', session_value):
        raise HTTPException(status_code=502, detail='invalid_hls_session')
    public_path = f'/hazard/media/{quote(media_path, safe="")}/'
    return (
        f'hlsSession={session_value}; Path={public_path}; '
        'Secure; HttpOnly; SameSite=None; Partitioned'
    )


async def _fetch_internal_hls_playlist(
    media_path: str,
    media_query: str,
) -> tuple[str, str | None]:
    """Fetch the current MediaMTX playlist for an authorised stream path.

    Args:
        media_path: Authorised MediaMTX stream path.
        media_query: Encoded upstream-only playlist query parameters.

    Returns:
        Upstream playlist body and an optional scoped HLS session cookie.

    Raises:
        HTTPException: If MediaMTX is unavailable, rejects, or has no playlist.
    """
    url = (
        f'{MEDIA_INTERNAL_HLS_BASE_URL}/'
        f'{quote(media_path, safe="")}/index.m3u8'
    )
    url = _append_query(url, media_query)
    try:
        async with httpx.AsyncClient(
            timeout=MEDIA_INTERNAL_HLS_TIMEOUT_SECONDS,
            # Recent MediaMTX releases use a one-time cookie-check redirect
            # before serving HLS.  Without following it httpx exposes the
            # empty 302 body as a seemingly-successful playlist.
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
        raise HTTPException(
            status_code=503,
            detail='media_playlist_not_ready',
        )
    # MediaMTX v1.18+ establishes the HLS session on index.m3u8.  The
    # browser must receive that same cookie before it fetches child playlists.
    hls_session = response.cookies.get('hlsSession')
    return response.text, _media_hls_session_cookie(media_path, hls_session)


def _extract_media_path_from_uri(uri: str) -> str:
    """Extract the MediaMTX stream path from an external media URI.

    Args:
        uri: Public HLS or WHEP URI, optionally with a query string.

    Returns:
        Decoded stream path, or an empty string when the URI is not media.
    """
    path = uri.split('?', 1)[0]
    segments = [unquote(part) for part in path.split('/') if part]
    try:
        media_index = segments.index('media')
    except ValueError:
        return ''

    path_index = media_index + 1
    if path_index < len(segments) and segments[path_index] == 'webrtc':
        path_index += 1
    if path_index >= len(segments):
        return ''
    return segments[path_index]


def _media_path_matches_site(media_path: str, site_name: str) -> bool:
    """Determine whether a MediaMTX path belongs to a site.

    Args:
        media_path: MediaMTX stream path to inspect.
        site_name: Decoded site label expected in the path.

    Returns:
        ``True`` when the path uses the encoded site prefix.
    """
    return media_path.startswith(
        f'hazard_{encode_media_segment(site_name)}_',
    )


def _opaque_media_session_allows_path(
    session: Mapping[str, object],
    media_path: str,
) -> bool:
    """Enforce the site, camera, profile, and quality scope of a capability.

    Args:
        session: Trusted opaque media session payload.
        media_path: Requested MediaMTX stream path.

    Returns:
        ``True`` only when the requested path is covered by the capability.
    """
    site = str(session.get('site') or '')
    quality = session.get('quality')
    if quality not in {'detail', 'preview'}:
        return False
    base_paths = {
        build_media_path(site, camera)
        for camera in media_session_cameras(session)
    }
    if quality == 'preview':
        base_paths = {
            build_preview_media_path(path)
            for path in base_paths
        }
    if not base_paths:
        return False
    profile = session.get('profile')
    if profile == 'clean':
        return media_path in base_paths
    if profile == 'overlay':
        parsed = parse_annotated_media_path(media_path)
        return parsed is not None and parsed[0] in base_paths
    return False


async def _active_overlay_languages(
    rds: redis.Redis,
    media_path: str,
) -> set[str]:
    """Return overlay languages currently demanded for a media path.

    Args:
        rds: Redis connection used to scan demand leases.
        media_path: Base clean-stream MediaMTX path.

    Returns:
        Canonical active overlay languages for the stream.
    """
    pattern = f'{OVERLAY_DEMAND_PREFIX}:{media_path}:*'
    languages: set[str] = set()
    async for raw_key in rds.scan_iter(match=pattern):
        key = raw_key.decode('utf-8')
        encoded_language = key.rsplit(':', 1)[-1]
        language = normalise_label_language(
            decode_media_segment(encoded_language),
        )
        if language in _allowed_overlay_languages():
            languages.add(language)
    return languages


async def _touch_overlay_demand(
    rds: redis.Redis,
    media_path: str,
    label_language: str,
    ttl_seconds: int = OVERLAY_DEMAND_TTL_SECONDS,
) -> None:
    """Renew the shared overlay producer demand lease.

    Args:
        rds: Redis connection used to write the demand lease.
        media_path: Base clean-stream MediaMTX path.
        label_language: Canonical overlay label language.
        ttl_seconds: Lease duration to set in seconds.
    """
    await rds.set(
        build_overlay_demand_key(media_path, label_language),
        str(int(time.time())).encode('ascii'),
        ex=ttl_seconds,
    )


async def _touch_clean_demand(
    rds: redis.Redis,
    media_path: str,
    ttl_seconds: int = CLEAN_DEMAND_TTL_SECONDS,
) -> None:
    """Renew the shared clean producer demand lease.

    Args:
        rds: Redis connection used to write the demand lease.
        media_path: Clean-stream MediaMTX path.
        ttl_seconds: Lease duration to set in seconds.
    """
    await rds.set(
        build_clean_demand_key(media_path),
        str(int(time.time())).encode('ascii'),
        ex=ttl_seconds,
    )


async def _touch_overlay_demand_from_media_path(
    rds: redis.Redis,
    media_path: str,
    ttl_seconds: int = OVERLAY_DEMAND_TTL_SECONDS,
) -> None:
    """Refresh overlay demand from an annotated media path.

    Args:
        rds: Redis connection used to write the demand lease.
        media_path: Annotated MediaMTX stream path.
        ttl_seconds: Lease duration to set in seconds.
    """
    parsed = parse_annotated_media_path(media_path)
    if parsed is None:
        return
    base_path, raw_language = parsed
    language = normalise_label_language(raw_language)
    if language not in _allowed_overlay_languages():
        return
    await _touch_overlay_demand(
        rds,
        base_path,
        language,
        ttl_seconds=ttl_seconds,
    )


async def _touch_media_demand_from_media_path(
    rds: redis.Redis,
    media_path: str,
    ttl_seconds: int | None = None,
) -> None:
    """Refresh the demand key matching a clean or overlay media path.

    Args:
        rds: Redis connection used to write the demand lease.
        media_path: Requested clean or annotated MediaMTX path.
        ttl_seconds: Optional lease duration overriding the profile default.
    """
    parsed = parse_annotated_media_path(media_path)
    if parsed is not None:
        await _touch_overlay_demand_from_media_path(
            rds,
            media_path,
            ttl_seconds=(ttl_seconds or OVERLAY_DEMAND_TTL_SECONDS),
        )
        return
    if media_path.startswith('hazard_'):
        await _touch_clean_demand(
            rds,
            media_path,
            ttl_seconds=(ttl_seconds or CLEAN_DEMAND_TTL_SECONDS),
        )


async def _overlay_is_ready(
    rds: redis.Redis,
    overlay_media_path: str,
) -> bool:
    """Determine whether a producer recently published an overlay path.

    Args:
        rds: Redis connection used to inspect readiness.
        overlay_media_path: Language-specific annotated MediaMTX path.

    Returns:
        ``True`` while the producer's ready marker has not expired.
    """
    return bool(await rds.exists(build_overlay_ready_key(overlay_media_path)))


def _normalise_playback_profile(profile: str | None) -> PlaybackProfile:
    """Normalise and validate a requested playback profile.

    Args:
        profile: Optional raw profile supplied by the client.

    Returns:
        Canonical clean or overlay profile.

    Raises:
        HTTPException: If the profile is unsupported.
    """
    value = (profile or 'clean').strip().lower()
    if value in {'clean', 'overlay'}:
        return cast(PlaybackProfile, value)
    raise HTTPException(status_code=422, detail='unsupported_profile')


def _normalise_playback_rendition(
    rendition: str | None,
) -> PlaybackRendition:
    """Normalise and validate a requested HLS rendition.

    Args:
        rendition: Optional raw rendition supplied by the client.

    Returns:
        Canonical detail or preview rendition.

    Raises:
        HTTPException: If the rendition is unsupported.
    """
    value = (rendition or 'detail').strip().lower()
    if value in {'detail', 'preview'}:
        return cast(PlaybackRendition, value)
    raise HTTPException(status_code=422, detail='unsupported_rendition')


def _playback_session_key(session_id: str) -> str:
    """Build the Redis key for a browser-facing playback session.

    Args:
        session_id: Opaque playback session identifier.

    Returns:
        Canonical Redis key for the session payload.
    """
    return f'{STREAM_PLAYBACK_SESSION_PREFIX}:{session_id}'


def _playback_media_session_key(media_path: str, session_id: str) -> str:
    """Build the reverse Redis index for a session and media path.

    Args:
        media_path: Selected clean or annotated MediaMTX path.
        session_id: Opaque playback session identifier.

    Returns:
        Canonical Redis reverse-index key.
    """
    return (
        f'{STREAM_PLAYBACK_MEDIA_SESSION_PREFIX}:'
        f'{media_path}:{session_id}'
    )


def _playback_media_session_pattern(media_path: str) -> str:
    """Build the Redis scan pattern for a media path's session indexes.

    Args:
        media_path: Selected clean or annotated MediaMTX path.

    Returns:
        Redis scan pattern for reverse indexes of that path.
    """
    return f'{STREAM_PLAYBACK_MEDIA_SESSION_PREFIX}:{media_path}:*'


def _build_session_playback_url(session_id: str) -> str:
    """Build the stable public playlist URL for a playback session.

    Args:
        session_id: Opaque playback session identifier.

    Returns:
        Relative public HLS playlist URL.
    """
    return (
        f'{STREAM_PLAYBACK_PUBLIC_BASE_PATH}/sessions/'
        f'{session_id}/index.m3u8'
    )


def _decode_playback_session_payload(
    raw: bytes | None,
) -> PlaybackSession | None:
    """Decode an optional Redis playback-session payload.

    Args:
        raw: Optional UTF-8 JSON payload retrieved from Redis.

    Returns:
        Typed playback session, or ``None`` when no Redis value exists.
    """
    if raw is None:
        return None
    return json.loads(raw)


async def _load_playback_session(
    rds: redis.Redis,
    session_id: str,
) -> PlaybackSession | None:
    """Load one short-lived playback session from Redis.

    Args:
        rds: Redis connection used to retrieve the session.
        session_id: Opaque playback session identifier.

    Returns:
        Decoded session, or ``None`` when it has expired or never existed.
    """
    return _decode_playback_session_payload(
        await rds.get(_playback_session_key(session_id)),
    )


def _session_selected_media_path(session: PlaybackSession) -> str:
    """Return the media path currently selected by a playback session.

    Args:
        session: Clean or overlay playback session.

    Returns:
        Clean base path or language-specific annotated path.
    """
    if session['profile'] == 'overlay':
        return session['overlay_media_path']
    return session['base_media_path']


def _session_possible_media_paths(session: PlaybackSession) -> set[str]:
    """Return media paths that may have reverse indexes for a session.

    Args:
        session: Clean or overlay playback session.

    Returns:
        Set containing the base path and the current selected path.
    """
    return {
        session['base_media_path'],
        _session_selected_media_path(session),
    }


async def _delete_playback_session_media_indexes(
    rds: redis.Redis,
    session: PlaybackSession,
) -> None:
    """Remove all reverse indexes for one playback session.

    Args:
        rds: Redis connection used to delete reverse indexes.
        session: Session whose indexed media paths are removed.
    """
    session_id = session['session_id']
    for media_path in _session_possible_media_paths(session):
        await rds.delete(_playback_media_session_key(media_path, session_id))


async def _register_playback_session_media_path(
    rds: redis.Redis,
    session: PlaybackSession,
    media_path: str,
) -> None:
    """Index a media path so media authorisation can renew a session.

    Args:
        rds: Redis connection used to store the reverse index.
        session: Session to renew when the path is read.
        media_path: Current selected MediaMTX path.
    """
    session_id = session['session_id']
    await rds.set(
        _playback_media_session_key(media_path, session_id),
        b'1',
        ex=STREAM_PLAYBACK_SESSION_TTL_SECONDS,
    )


async def _refresh_playback_sessions_for_media_path(
    rds: redis.Redis,
    media_path: str,
) -> None:
    """Renew sessions whose selected proxied HLS media is being read.

    Args:
        rds: Redis connection used to scan and renew session indexes.
        media_path: MediaMTX path just authorised by the media proxy.
    """
    pattern = _playback_media_session_pattern(media_path)
    async for raw_key in rds.scan_iter(match=pattern):
        key = raw_key.decode('utf-8')
        session_id = key.rsplit(':', 1)[-1]
        session = await _load_playback_session(rds, session_id)
        if session is None:
            await rds.delete(key)
            continue
        if _session_selected_media_path(session) != media_path:
            await rds.delete(key)
            continue
        await _refresh_playback_session_ttl(rds, session_id)
        await rds.expire(key, STREAM_PLAYBACK_SESSION_TTL_SECONDS)


async def _create_or_update_playback_session(
    rds: redis.Redis,
    session_id: str | None,
    username: str,
    label: str,
    stream_name: str,
    profile: PlaybackProfile,
    rendition: PlaybackRendition,
    language: str | None,
) -> PlaybackSession:
    """Create or update a short-lived playback session for one camera.

    Args:
        rds: Redis connection used to store session state and indexes.
        session_id: Optional existing session identifier to update.
        username: Authenticated owner of the session.
        label: Site label containing the stream.
        stream_name: Validated configured stream name.
        profile: Requested clean or overlay profile.
        rendition: Requested detail or preview rendition.
        language: Canonical overlay language, if the profile is overlay.

    Returns:
        Newly created or updated typed playback session.

    Raises:
        HTTPException: If an existing session is absent or owned by another user.
    """
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=STREAM_PLAYBACK_SESSION_TTL_SECONDS)

    if session_id:
        existing = await _load_playback_session(rds, session_id)
        if existing is None:
            raise HTTPException(status_code=404, detail='session_not_found')
        if existing.get('username') != username:
            raise HTTPException(status_code=403, detail='session_forbidden')
        await _delete_playback_session_media_indexes(rds, existing)
    else:
        # Generate server-side identifiers; clients may only refresh a session
        # that is already present and owned by the authenticated user.
        session_id = secrets.token_urlsafe(24)

    detail_media_path = build_media_path(label, stream_name)
    base_media_path = (
        build_preview_media_path(detail_media_path)
        if rendition == 'preview'
        else detail_media_path
    )
    session: PlaybackSession
    if profile == 'clean':
        session = CleanPlaybackSession(
            session_id=session_id,
            username=username,
            label=label,
            stream_name=stream_name,
            stream_id=encode_stream_id(stream_name),
            profile='clean',
            rendition=rendition,
            language=None,
            base_media_path=base_media_path,
            overlay_media_path=None,
            created_at=now.isoformat(),
            expires_at=expires_at.isoformat(),
        )
    else:
        if language is None:
            raise HTTPException(status_code=422, detail='overlay_language_required')
        session = OverlayPlaybackSession(
            session_id=session_id,
            username=username,
            label=label,
            stream_name=stream_name,
            stream_id=encode_stream_id(stream_name),
            profile='overlay',
            rendition=rendition,
            language=language,
            base_media_path=base_media_path,
            overlay_media_path=build_annotated_media_path(
                base_media_path,
                language,
            ),
            created_at=now.isoformat(),
            expires_at=expires_at.isoformat(),
        )
    await rds.set(
        _playback_session_key(session_id),
        json.dumps(session, ensure_ascii=False).encode('utf-8'),
        ex=STREAM_PLAYBACK_SESSION_TTL_SECONDS,
    )
    return session


async def _refresh_playback_session_ttl(
    rds: redis.Redis,
    session_id: str,
) -> None:
    """Renew a playback session while its HLS playlist is read.

    Args:
        rds: Redis connection used to extend the session expiry.
        session_id: Opaque session identifier to renew.
    """
    await rds.expire(
        _playback_session_key(session_id),
        STREAM_PLAYBACK_SESSION_TTL_SECONDS,
    )


async def _select_session_playback(
    rds: redis.Redis,
    session: PlaybackSession,
) -> PlaybackSessionState:
    """Resolve the current media path for a stable playback session.

    Args:
        rds: Redis connection used to renew demand and readiness state.
        session: Clean or overlay playback session to resolve.

    Returns:
        Current media path, HLS URL, and client-visible readiness state.
    """
    if session['profile'] == 'clean':
        base_media_path = session['base_media_path']
        await _touch_clean_demand(rds, base_media_path)
        await _register_playback_session_media_path(
            rds, session, base_media_path,
        )
        return {
            'status': 'ready',
            'state': 'ready',
            'overlay_ready': False,
            'media_path': base_media_path,
            'hls_url': build_media_hls_url(base_media_path),
        }

    base_media_path = session['base_media_path']
    language = session['language']
    overlay_media_path = session['overlay_media_path']
    overlay_hls_url = build_media_hls_url(overlay_media_path)
    await _touch_overlay_demand(rds, base_media_path, language)
    await _register_playback_session_media_path(
        rds, session, overlay_media_path,
    )
    ready = await _overlay_is_ready(rds, overlay_media_path)
    return {
        'status': 'ready' if ready else 'starting',
        'state': 'ready' if ready else 'starting',
        'overlay_ready': ready,
        'media_path': overlay_media_path,
        'hls_url': overlay_hls_url,
    }


async def _wait_for_session_startup(session: PlaybackSession) -> None:
    """Wait briefly for an on-demand publisher to create its HLS playlist.

    Args:
        session: Session whose creation time bounds the startup wait.
    """
    if STREAM_PLAYBACK_STARTUP_WAIT_SECONDS <= 0:
        return
    created = datetime.fromisoformat(session['created_at'])
    elapsed = (datetime.now(timezone.utc) - created).total_seconds()
    remaining = STREAM_PLAYBACK_STARTUP_WAIT_SECONDS - elapsed
    if remaining > 0:
        await asyncio.sleep(remaining)


async def _build_playback_session_response_body(
    rds: redis.Redis,
    session: PlaybackSession,
) -> PlaybackSessionResponse:
    """Build the API payload returned to frontend playback controllers.

    Args:
        rds: Redis connection used to resolve live demand and readiness.
        session: Clean or overlay playback session to expose.

    Returns:
        Public stable playback descriptor for the frontend.
    """
    state = await _select_session_playback(rds, session)
    session_id = session['session_id']
    stable_url = _build_session_playback_url(session_id)

    body: PlaybackSessionResponse = {
        'session_id': session_id,
        'stream_id': session['stream_id'],
        'key': session['stream_name'],
        'label': session['label'],
        'transport': 'hls',
        'status': state['status'],
        'state': state['state'],
        'profile': session['profile'],
        'rendition': session['rendition'],
        'playback_ready': True,
        'playback_url': stable_url,
        'media_hls_url': state['hls_url'],
        'language': session['language'],
        'overlay_ready': state['overlay_ready'],
        'media_path': state['media_path'],
        'expires_at': session['expires_at'],
        'expires_in': STREAM_PLAYBACK_SESSION_TTL_SECONDS,
        'demand_ttl_seconds': OVERLAY_DEMAND_TTL_SECONDS,
    }
    return body


async def _has_other_playback_session(
    rds: redis.Redis,
    released_session_id: str,
    base_media_path: str,
    profile: PlaybackProfile,
    language: str | None = None,
) -> bool:
    """Determine whether another active session uses the same producer demand.

    Args:
        rds: Redis connection used to scan active playback sessions.
        released_session_id: Session being released and excluded from the scan.
        base_media_path: Shared clean base path to compare.
        profile: Clean or overlay profile to compare.
        language: Overlay language to compare for overlay sessions.

    Returns:
        ``True`` when another session still requires the same producer lease.
    """
    pattern = f'{STREAM_PLAYBACK_SESSION_PREFIX}:*'
    async for raw_key in rds.scan_iter(match=pattern):
        key = raw_key.decode('utf-8')
        if key == _playback_session_key(released_session_id):
            continue
        session = _decode_playback_session_payload(await rds.get(key))
        if (
            session
            and session['profile'] == profile
            and session['base_media_path'] == base_media_path
            and (profile != 'overlay' or session['language'] == language)
        ):
            return True
    return False


def _build_overlay_language_response() -> OverlayLanguageListResponse:
    """Build the public overlay-language capability response.

    Returns:
        Canonical language options, aliases, and lease limits for clients.
    """
    allowed_languages = _allowed_overlay_languages()
    return OverlayLanguageListResponse(
        default_language=_default_overlay_language(),
        allowed_language_codes=list(allowed_languages),
        supported_languages=list(allowed_languages),
        aliases=_language_alias_map(),
        languages=_overlay_language_options(allowed_languages),
        stream_playback_endpoint='/hazard/api/stream-playback',
        playback_endpoint='/hazard/api/stream-playback',
        max_active_languages_per_stream=OVERLAY_MAX_ACTIVE_LANGUAGES,
        demand_ttl_seconds=OVERLAY_DEMAND_TTL_SECONDS,
        ready_ttl_seconds=int(
            os.getenv('MEDIA_OVERLAY_READY_TTL_SECONDS', '15'),
        ),
    )
