from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import secrets
import time
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from urllib.parse import parse_qs
from urllib.parse import parse_qsl
from urllib.parse import quote
from urllib.parse import unquote
from urllib.parse import urlencode
from urllib.parse import urljoin
from urllib.parse import urlsplit

import httpx
import redis.asyncio as redis
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from fastapi import Security
from fastapi import WebSocket
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.config import Settings
from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.models import Site
from examples.auth.models import StreamConfig as StreamConfigModel
from examples.auth.models import USER_STATUS_ACTIVE
from examples.auth.redis_pool import get_redis_pool
from examples.auth.redis_pool import get_redis_pool_ws
from examples.auth.session_store import get_media_session
from examples.auth.session_store import media_session_cameras
from examples.auth.user_service import load_user_access_context
from examples.local_notification_server.lang_config import LANGUAGES
from examples.streaming_web.media_paths import (
    build_annotated_media_path,
)
from examples.streaming_web.media_paths import build_clean_demand_key
from examples.streaming_web.media_paths import build_media_hls_url
from examples.streaming_web.media_paths import build_media_path
from examples.streaming_web.media_paths import build_media_webrtc_url
from examples.streaming_web.media_paths import (
    build_overlay_demand_key,
)
from examples.streaming_web.media_paths import build_overlay_ready_key
from examples.streaming_web.media_paths import build_preview_media_path
from examples.streaming_web.media_paths import decode_media_segment
from examples.streaming_web.media_paths import encode_media_segment
from examples.streaming_web.media_paths import OVERLAY_DEMAND_PREFIX
from examples.streaming_web.media_paths import (
    parse_annotated_media_path,
)
from examples.streaming_web.overlay_renderer import CLASS_LABELS
from examples.streaming_web.overlay_renderer import LANGUAGE_ALIASES
from examples.streaming_web.overlay_renderer import (
    normalise_label_language,
)
from examples.streaming_web.overlay_renderer import (
    normalise_overlay_mode,
)
from examples.streaming_web.overlay_renderer import (
    SUPPORTED_LABEL_LANGUAGES,
)
from examples.streaming_web.overlay_renderer import WARNING_LABELS
from examples.streaming_web.schemas import LabelListResponse
from examples.streaming_web.schemas import MAX_STREAM_PLAYBACK_BATCH_STREAMS
from examples.streaming_web.schemas import OverlayLanguageInfo
from examples.streaming_web.schemas import OverlayLanguageListResponse
from examples.streaming_web.schemas import StreamPlaybackBatchRequest
from examples.streaming_web.schemas import StreamPlaybackRequest
from examples.streaming_web.utils import Utils
from examples.streaming_web.webrtc_service import (
    get_public_ice_servers,
)
from examples.streaming_web.ws_handlers import (
    handle_metadata_stream_id_ws,
)
from examples.streaming_web.ws_handlers import (
    metadata_stream_generator,
)

# Module-level alias retained for test patching
get_user_and_sites = load_user_access_context


# Settings and router
settings: Settings = Settings()
router: APIRouter = APIRouter()


def _csv_env(name: str, default: str) -> list[str]:
    """Read a comma-separated environment setting."""
    value = os.getenv(name, default)
    return [item.strip() for item in value.split(',') if item.strip()]


def _allowed_overlay_languages() -> tuple[str, ...]:
    """Return overlay languages enabled for playback."""
    configured = _csv_env(
        'MEDIA_OVERLAY_ALLOWED_LANGUAGES',
        ','.join(SUPPORTED_LABEL_LANGUAGES),
    )
    allowed = []
    for language in configured:
        normalised = normalise_label_language(language)
        if (
            normalised not in allowed
            and normalised in SUPPORTED_LABEL_LANGUAGES
        ):
            allowed.append(normalised)
    return tuple(allowed or ('en',))


OVERLAY_LANGUAGE_DETAILS: dict[str, dict[str, object]] = {
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
    """Return the default overlay language accepted by the server."""
    language = normalise_label_language(
        os.getenv('MEDIA_DEFAULT_OVERLAY_LANGUAGE', 'zh-TW'),
    )
    allowed = _allowed_overlay_languages()
    return language if language in allowed else allowed[0]


def _language_alias_map() -> dict[str, str]:
    """Return normalised language aliases for API consumers."""
    aliases = dict(LANGUAGE_ALIASES)
    for code, details in OVERLAY_LANGUAGE_DETAILS.items():
        aliases[code] = code
        detail_aliases = details.get('aliases', ())
        if not isinstance(detail_aliases, (list, tuple)):
            continue
        for alias in detail_aliases:
            if isinstance(alias, str):
                aliases[alias] = code
                aliases[alias.lower()] = code
    return aliases


def _notification_language_code(code: str) -> str:
    """Map an overlay language to a notification language code."""
    notification_code = OVERLAY_TO_NOTIFICATION_LANGUAGE.get(code, 'en-GB')
    if notification_code in LANGUAGES:
        return notification_code
    return 'en-GB'


def _overlay_language_options(
    allowed_languages: tuple[str, ...] | None = None,
) -> list[OverlayLanguageInfo]:
    """Build language option metadata for the frontend."""
    languages: list[OverlayLanguageInfo] = []
    for code in allowed_languages or _allowed_overlay_languages():
        details = OVERLAY_LANGUAGE_DETAILS.get(code, {})
        notification_code = _notification_language_code(code)
        languages.append(
            OverlayLanguageInfo(
                code=code,
                notification_code=notification_code,
                display_name=str(details.get('name', code)),
                native_name=str(details.get('native_name', code)),
                is_default=code == _default_overlay_language(),
                class_labels=CLASS_LABELS.get(code, CLASS_LABELS['en']),
                warning_labels=WARNING_LABELS.get(code, WARNING_LABELS['en']),
                notification_templates=LANGUAGES.get(
                    notification_code,
                    LANGUAGES['en-GB'],
                ),
            ),
        )
    return languages


def _overlay_language_option_payloads(
    allowed_languages: tuple[str, ...] | None = None,
) -> list[dict[str, object]]:
    """Return JSONResponse-safe language option dictionaries."""
    return [
        language.model_dump()
        for language in _overlay_language_options(allowed_languages)
    ]


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
    """Raise an HTTP error when the authenticated user cannot view a label."""
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    user, user_site_names, user_role = await get_user_and_sites(db, username)
    if getattr(user, 'status', USER_STATUS_ACTIVE) != USER_STATUS_ACTIVE:
        raise _media_auth_401('inactive_user')
    if user_role != 'super_admin' and label not in user_site_names:
        raise HTTPException(status_code=403, detail='Access denied')


def _extract_opaque_media_token(request: Request) -> str:
    """Return the dedicated opaque media capability, never a main JWT."""
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
    """Build a media-auth 401 with a machine-readable reason."""
    return HTTPException(
        status_code=401,
        detail=detail,
        headers={
            'WWW-Authenticate': f'Bearer error="{detail}"',
            'X-Media-Auth-Error': detail,
        },
    )


def _media_session_demand_ttl(session: dict[str, object]) -> int:
    """Return a bounded producer lease from trusted capability expiry."""
    try:
        remaining = int(session.get('expires_at') or 0) - int(time.time())
    except (TypeError, ValueError):
        remaining = 0
    return max(MEDIA_PUBLISHER_IDLE_GRACE_SECONDS, remaining)


def _split_hls_playlist_query(query: str) -> tuple[str, str]:
    """Return auth-token query and MediaMTX playlist query separately."""
    auth_items: list[tuple[str, str]] = []
    media_items: list[tuple[str, str]] = []
    for key, value in parse_qsl(query, keep_blank_values=True):
        if key in {'mt', 'media_token'}:
            auth_items.append((key, value))
        else:
            media_items.append((key, value))
    return urlencode(auth_items), urlencode(media_items)


def _append_query(url: str, query: str) -> str:
    if not query:
        return url
    separator = '&' if '?' in url else '?'
    return f'{url}{separator}{query}'


def _rewrite_hls_uri(uri: str, media_path: str, auth_query: str) -> str:
    """Rewrite one playlist URI so HLS fragment reads carry media auth."""
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
    *,
    media_path: str,
    auth_query: str,
) -> str:
    """Append the short-lived media token to every media URI in an M3U8."""
    rewritten_lines: list[str] = []
    uri_attr = re.compile(r'URI="([^"]+)"')

    def rewrite_uri_attr(match: re.Match[str]) -> str:
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
    session_value: object,
) -> str | None:
    """Build the browser cookie required by MediaMTX HLS child requests."""
    if not isinstance(session_value, str):
        return None
    if not re.fullmatch(r'[A-Za-z0-9._~-]+', session_value):
        return None
    public_path = f'/hazard/media/{quote(media_path, safe="")}/'
    return (
        f'hlsSession={session_value}; Path={public_path}; '
        'Secure; HttpOnly; SameSite=None; Partitioned'
    )


async def _fetch_internal_hls_playlist(
    media_path: str,
    *,
    media_query: str,
) -> tuple[str, str | None]:
    """Fetch the current MediaMTX playlist for a stream path."""
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
    """Extract the MediaMTX stream path from an external media URI."""
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
    """Return whether a MediaMTX path belongs to a site name."""
    return media_path.startswith(
        f'hazard_{encode_media_segment(site_name)}_',
    )


def _opaque_media_session_allows_path(
    session: dict[str, object],
    media_path: str,
) -> bool:
    """Enforce the site/camera/profile scope embedded in a media session."""
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


def _decode_redis_key(raw_key: object) -> str:
    """Decode a Redis key returned by async scan."""
    if isinstance(raw_key, bytes):
        return raw_key.decode('utf-8')
    return str(raw_key)


async def _active_overlay_languages(
    rds: redis.Redis,
    media_path: str,
) -> set[str]:
    """Return active overlay languages currently demanded for a stream."""
    pattern = f'{OVERLAY_DEMAND_PREFIX}:{media_path}:*'
    languages: set[str] = set()
    async for raw_key in rds.scan_iter(match=pattern):
        key = _decode_redis_key(raw_key)
        encoded_language = key.rsplit(':', 1)[-1]
        try:
            language = normalise_label_language(
                decode_media_segment(encoded_language),
            )
        except Exception:
            continue
        if language in _allowed_overlay_languages():
            languages.add(language)
    return languages


async def _touch_overlay_demand(
    rds: redis.Redis,
    media_path: str,
    label_language: str,
    ttl_seconds: int = OVERLAY_DEMAND_TTL_SECONDS,
) -> None:
    """Keep a shared overlay profile alive while a viewer uses it."""
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
    """Keep a shared clean profile alive while a viewer uses it."""
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
    """Refresh overlay demand using an annotated media path."""
    parsed = parse_annotated_media_path(media_path)
    if parsed is None:
        return
    base_path, raw_language = parsed
    language = normalise_label_language(raw_language)
    if language not in _allowed_overlay_languages():
        return
    try:
        await _touch_overlay_demand(
            rds,
            base_path,
            language,
            ttl_seconds=ttl_seconds,
        )
    except Exception as exc:
        print(
            f"Failed to renew overlay demand for {media_path}: {exc}",
            flush=True,
        )


async def _touch_media_demand_from_media_path(
    rds: redis.Redis,
    media_path: str,
    ttl_seconds: int | None = None,
) -> None:
    """Refresh the demand key that matches a clean or overlay media path."""
    parsed = parse_annotated_media_path(media_path)
    if parsed is not None:
        await _touch_overlay_demand_from_media_path(
            rds,
            media_path,
            ttl_seconds=(ttl_seconds or OVERLAY_DEMAND_TTL_SECONDS),
        )
        return
    if media_path.startswith('hazard_'):
        try:
            await _touch_clean_demand(
                rds,
                media_path,
                ttl_seconds=(ttl_seconds or CLEAN_DEMAND_TTL_SECONDS),
            )
        except Exception as exc:
            print(
                f"Failed to renew clean demand for {media_path}: {exc}",
                flush=True,
            )


async def _overlay_is_ready(
    rds: redis.Redis,
    overlay_media_path: str,
) -> bool:
    """Return whether the producer recently published the overlay path."""
    return bool(await rds.exists(build_overlay_ready_key(overlay_media_path)))


def _normalise_playback_profile(profile: str | None) -> str:
    """Return the requested playback profile for a session."""
    value = (profile or 'clean').strip().lower()
    if value in {'clean', 'overlay'}:
        return value
    raise HTTPException(status_code=422, detail='unsupported_profile')


def _normalise_playback_rendition(rendition: str | None) -> str:
    """Return the exact HLS rendition selected by a playback session."""
    value = (rendition or 'detail').strip().lower()
    if value in {'detail', 'preview'}:
        return value
    raise HTTPException(status_code=422, detail='unsupported_rendition')


def _playback_session_key(session_id: str) -> str:
    return f'{STREAM_PLAYBACK_SESSION_PREFIX}:{session_id}'


def _playback_media_session_key(media_path: str, session_id: str) -> str:
    return (
        f'{STREAM_PLAYBACK_MEDIA_SESSION_PREFIX}:'
        f'{media_path}:{session_id}'
    )


def _playback_media_session_pattern(media_path: str) -> str:
    return f'{STREAM_PLAYBACK_MEDIA_SESSION_PREFIX}:{media_path}:*'


def _build_session_playback_url(session_id: str) -> str:
    return (
        f'{STREAM_PLAYBACK_PUBLIC_BASE_PATH}/sessions/'
        f'{session_id}/index.m3u8'
    )


def _decode_playback_session_payload(raw: object) -> dict[str, object] | None:
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8')
    if not isinstance(raw, str):
        raw = str(raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


async def _load_playback_session(
    rds: redis.Redis,
    session_id: str,
) -> dict[str, object] | None:
    """Load one short-lived playback session from Redis."""
    return _decode_playback_session_payload(
        await rds.get(_playback_session_key(session_id)),
    )


def _session_selected_media_path(session: dict[str, object]) -> str | None:
    """Return the media path currently selected by a playback session."""
    base_media_path = session.get('base_media_path')
    if not isinstance(base_media_path, str) or not base_media_path:
        return None

    profile = str(session.get('profile') or 'clean')
    if profile != 'overlay':
        return base_media_path

    overlay_media_path = session.get('overlay_media_path')
    if isinstance(overlay_media_path, str) and overlay_media_path:
        return overlay_media_path

    language = str(session.get('language') or _default_overlay_language())
    return build_annotated_media_path(base_media_path, language)


def _session_possible_media_paths(session: dict[str, object]) -> set[str]:
    """Return media paths that may have reverse indexes for this session."""
    paths: set[str] = set()
    base_media_path = session.get('base_media_path')
    if isinstance(base_media_path, str) and base_media_path:
        paths.add(base_media_path)
    selected = _session_selected_media_path(session)
    if selected:
        paths.add(selected)
    return paths


async def _delete_playback_session_media_indexes(
    rds: redis.Redis,
    session: dict[str, object],
) -> None:
    """Remove reverse indexes for one playback session."""
    session_id = session.get('session_id')
    if not isinstance(session_id, str) or not session_id:
        return
    for media_path in _session_possible_media_paths(session):
        await rds.delete(_playback_media_session_key(media_path, session_id))


async def _register_playback_session_media_path(
    rds: redis.Redis,
    session: dict[str, object],
    media_path: str,
) -> None:
    """Index a media path so media-auth can renew playback sessions."""
    session_id = session.get('session_id')
    if not isinstance(session_id, str) or not session_id:
        return
    await rds.set(
        _playback_media_session_key(media_path, session_id),
        b'1',
        ex=STREAM_PLAYBACK_SESSION_TTL_SECONDS,
    )


async def _scan_redis_keys(
    rds: redis.Redis,
    pattern: str,
):
    """Yield Redis keys from clients whose scan_iter may be sync or async."""
    iterator = rds.scan_iter(match=pattern)
    if inspect.isawaitable(iterator):
        iterator = await iterator
    async for raw_key in iterator:
        yield _decode_redis_key(raw_key)


async def _refresh_playback_sessions_for_media_path(
    rds: redis.Redis,
    media_path: str,
) -> None:
    """Keep playback sessions alive when their proxied HLS media is read."""
    pattern = _playback_media_session_pattern(media_path)
    try:
        async for key in _scan_redis_keys(rds, pattern):
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
    except Exception as exc:
        print(
            f'Failed to refresh playback sessions for {media_path}: {exc}',
            flush=True,
        )


async def _create_or_update_playback_session(
    rds: redis.Redis,
    *,
    session_id: str | None,
    username: str,
    label: str,
    stream_name: str,
    profile: str,
    rendition: str,
    language: str | None,
) -> dict[str, object]:
    """Create or update a playback session for one camera."""
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
        session_id = secrets.token_urlsafe(24)

    detail_media_path = build_media_path(label, stream_name)
    base_media_path = (
        build_preview_media_path(detail_media_path)
        if rendition == 'preview'
        else detail_media_path
    )
    overlay_language = language if profile == 'overlay' else None
    overlay_media_path = (
        build_annotated_media_path(base_media_path, overlay_language)
        if overlay_language
        else None
    )
    session: dict[str, object] = {
        'session_id': session_id,
        'username': username,
        'label': label,
        'stream_name': stream_name,
        'stream_id': Utils.encode(stream_name),
        'profile': profile,
        'rendition': rendition,
        'language': overlay_language,
        'base_media_path': base_media_path,
        'overlay_media_path': overlay_media_path,
        'created_at': now.isoformat(),
        'expires_at': expires_at.isoformat(),
    }
    await rds.set(
        _playback_session_key(session_id),
        json.dumps(session, ensure_ascii=False),
        ex=STREAM_PLAYBACK_SESSION_TTL_SECONDS,
    )
    return session


async def _refresh_playback_session_ttl(
    rds: redis.Redis,
    session_id: str,
) -> None:
    """Keep the playback session alive while its HLS playlist is read."""
    try:
        await rds.expire(
            _playback_session_key(session_id),
            STREAM_PLAYBACK_SESSION_TTL_SECONDS,
        )
    except Exception as exc:
        print(
            f'Failed to refresh playback session {session_id}: {exc}',
            flush=True,
        )


async def _select_session_playback(
    rds: redis.Redis,
    session: dict[str, object],
) -> dict[str, object]:
    """Resolve the current media path for a stable playback session."""
    profile = str(session.get('profile') or 'clean')
    base_media_path = str(session['base_media_path'])
    if profile != 'overlay':
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

    language = str(session.get('language') or _default_overlay_language())
    overlay_media_path = str(
        session.get('overlay_media_path')
        or build_annotated_media_path(base_media_path, language),
    )
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


async def _wait_for_session_startup(session: dict[str, object]) -> None:
    """Give an on-demand publisher time to create its HLS playlist."""
    if STREAM_PLAYBACK_STARTUP_WAIT_SECONDS <= 0:
        return
    created_at = session.get('created_at')
    if not isinstance(created_at, str):
        return
    try:
        created = datetime.fromisoformat(created_at)
    except ValueError:
        return
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    elapsed = (datetime.now(timezone.utc) - created).total_seconds()
    remaining = STREAM_PLAYBACK_STARTUP_WAIT_SECONDS - elapsed
    if remaining > 0:
        await asyncio.sleep(remaining)


async def _build_playback_session_response_body(
    rds: redis.Redis,
    session: dict[str, object],
) -> dict[str, object]:
    """Build the API payload returned to frontend playback controllers."""
    state = await _select_session_playback(rds, session)
    profile = str(session.get('profile') or 'clean')
    session_id = str(session['session_id'])
    stable_url = _build_session_playback_url(session_id)

    body: dict[str, object] = {
        'session_id': session_id,
        'stream_id': session.get('stream_id'),
        'key': session.get('stream_name'),
        'label': session.get('label'),
        'transport': 'hls',
        'status': state['status'],
        'state': state['state'],
        'profile': profile,
        'rendition': session.get('rendition', 'detail'),
        'playback_ready': True,
        'playback_url': stable_url,
        'media_hls_url': state['hls_url'],
        'language': session.get('language'),
        'overlay_ready': state['overlay_ready'],
        'media_path': state['media_path'],
        'expires_at': session.get('expires_at'),
        'expires_in': STREAM_PLAYBACK_SESSION_TTL_SECONDS,
        'demand_ttl_seconds': OVERLAY_DEMAND_TTL_SECONDS,
    }
    return body


async def _has_other_overlay_sessions(
    rds: redis.Redis,
    *,
    released_session_id: str,
    base_media_path: str,
    language: str,
) -> bool:
    """Return whether another active playback session wants this overlay."""
    pattern = f'{STREAM_PLAYBACK_SESSION_PREFIX}:*'
    try:
        async for raw_key in rds.scan_iter(match=pattern):
            key = _decode_redis_key(raw_key)
            if key == _playback_session_key(released_session_id):
                continue
            session = _decode_playback_session_payload(await rds.get(key))
            if not session:
                continue
            if (
                session.get('profile') == 'overlay'
                and session.get('base_media_path') == base_media_path
                and session.get('language') == language
            ):
                return True
    except Exception as exc:
        print(f'Failed to scan playback sessions: {exc}', flush=True)
        return True
    return False


async def _has_other_clean_sessions(
    rds: redis.Redis,
    *,
    released_session_id: str,
    base_media_path: str,
) -> bool:
    """Return whether another active playback session wants clean video."""
    pattern = f'{STREAM_PLAYBACK_SESSION_PREFIX}:*'
    try:
        async for raw_key in rds.scan_iter(match=pattern):
            key = _decode_redis_key(raw_key)
            if key == _playback_session_key(released_session_id):
                continue
            session = _decode_playback_session_payload(await rds.get(key))
            if not session:
                continue
            if (
                session.get('profile') == 'clean'
                and session.get('base_media_path') == base_media_path
            ):
                return True
    except Exception as exc:
        print(f'Failed to scan playback sessions: {exc}', flush=True)
        return True
    return False


def _normalise_stream_id(value: str) -> str:
    """Decode stream ids generated by older and newer listing payloads."""
    decoded = Utils.decode(value)
    if decoded != value:
        return decoded
    try:
        return decode_media_segment(value)
    except Exception:
        return value


def _visible_stream_names_query(label: str):
    """Select streams currently enabled for live playback."""
    return (
        select(StreamConfigModel.stream_name)
        .join(Site)
        .where(
            Site.name == label,
            StreamConfigModel.recognition_enabled.is_(True),
        )
    )


async def _resolve_configured_stream_name(
    db: AsyncSession,
    label: str,
    stream_id: str | None,
    key: str | None,
) -> str:
    """Resolve and validate a stream name enabled for live playback."""
    requested_name = key or (
        _normalise_stream_id(
            stream_id,
        ) if stream_id else ''
    )
    if not requested_name:
        raise HTTPException(
            status_code=422, detail='stream_id_or_key_required',
        )

    result = await db.execute(_visible_stream_names_query(label))
    stream_names = set(result.scalars().all())
    if requested_name not in stream_names:
        raise HTTPException(status_code=404, detail='stream_not_found')
    return requested_name


@router.get(
    '/labels',
    response_model=LabelListResponse,
)
async def get_labels_route(
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
    db: AsyncSession = Depends(get_db),
) -> LabelListResponse:
    """Return stream labels visible to the authenticated user."""
    try:
        username: str | None = credentials.subject.get('username')
        if not username:
            raise HTTPException(
                status_code=401, detail='Invalid token: no subject',
            )

        _, user_site_names, user_role = await get_user_and_sites(db, username)
        print(
            f"User {username} has access to sites: {user_site_names}",
        )
        result = await db.execute(select(Site.name).order_by(Site.name))
        all_labels = list(result.scalars().all())
        print(f"Configured labels: {all_labels}")
        filtered_labels = Utils.filter_labels(
            all_labels, user_role, user_site_names,
        )
        return LabelListResponse(labels=filtered_labels)
    except Exception as e:
        print(f"Failed to fetch labels: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch labels: {str(e)}",
        )


def _build_overlay_language_response() -> OverlayLanguageListResponse:
    """Build the overlay language response payload."""
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


@router.get('/overlay-languages', response_model=OverlayLanguageListResponse)
async def get_overlay_languages(
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
) -> OverlayLanguageListResponse:
    """Return canonical overlay language codes supported by this backend."""
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    return _build_overlay_language_response()


@router.get(
    '/stream-playback/languages',
    response_model=OverlayLanguageListResponse,
)
async def get_stream_playback_languages(
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
) -> OverlayLanguageListResponse:
    """Return language options for stream-playback overlay requests."""
    return await get_overlay_languages(credentials)


@router.get('/media-auth', include_in_schema=False)
async def authorise_media_request(
    request: Request,
    rds: redis.Redis = Depends(get_redis_pool),
) -> Response:
    """Authorise MediaMTX requests from a scoped Redis capability."""
    original_uri = (
        request.headers.get('x-original-uri')
        or request.headers.get('x-forwarded-uri')
        or str(request.url.path)
    )
    media_path = _extract_media_path_from_uri(original_uri)
    if not media_path.startswith('hazard_'):
        raise HTTPException(status_code=403, detail='Invalid media path')

    opaque_token = _extract_opaque_media_token(request)
    if not opaque_token:
        raise _media_auth_401('missing_media_token')
    opaque_session = await get_media_session(rds, opaque_token)
    if opaque_session is None:
        raise _media_auth_401('expired_media_session')
    if opaque_session.get('user_active') is False:
        raise _media_auth_401('inactive_user')
    if not _opaque_media_session_allows_path(opaque_session, media_path):
        raise HTTPException(status_code=403, detail='media_scope_denied')
    await _touch_media_demand_from_media_path(
        rds,
        media_path,
        ttl_seconds=_media_session_demand_ttl(opaque_session),
    )
    await _refresh_playback_sessions_for_media_path(rds, media_path)
    return Response(
        status_code=204,
        headers={
            'Cache-Control': 'no-store',
            'X-Media-Auth-Mode': 'opaque_media_session',
        },
    )


@router.get('/stream-playback/sessions/{session_id}/index.m3u8')
async def stream_playback_session_playlist(
    session_id: str,
    request: Request,
    rds: redis.Redis = Depends(get_redis_pool),
) -> Response:
    """Serve a stable playlist whose fragments keep the media auth token."""
    session = await _load_playback_session(rds, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='session_not_found')

    auth_query, media_query = _split_hls_playlist_query(request.url.query)
    if not auth_query:
        raise _media_auth_401('missing_media_token')

    await _refresh_playback_session_ttl(rds, session_id)
    state = await _select_session_playback(rds, session)
    await _wait_for_session_startup(session)
    target_url = str(state['hls_url'])
    media_path = _extract_media_path_from_uri(target_url)
    if not media_path.startswith('hazard_'):
        raise HTTPException(status_code=502, detail='invalid_media_playlist')
    playlist, hls_session_cookie = await _fetch_internal_hls_playlist(
        media_path,
        media_query=media_query,
    )
    rewritten = _rewrite_hls_playlist_media_urls(
        playlist,
        media_path=media_path,
        auth_query=auth_query,
    )
    has_media_uri = any(
        line and not line.startswith('#')
        for line in rewritten.splitlines()
    ) or 'URI="/hazard/media/' in rewritten
    has_media_uri_header = 'true' if has_media_uri else 'false'
    response = Response(
        content=rewritten,
        media_type='application/vnd.apple.mpegurl',
        headers={
            'Cache-Control': 'no-store',
            'X-Playback-Session': session_id,
            'X-Playback-Profile': str(session.get('profile') or 'clean'),
            'X-HLS-Media-Path': media_path,
            'X-HLS-Playlist-Lines': str(len(rewritten.splitlines())),
            'X-HLS-Playlist-Has-Media-URI': has_media_uri_header,
        },
    )
    if hls_session_cookie:
        response.headers.append('Set-Cookie', hls_session_cookie)
    return response


async def _negotiate_stream_playback(
    request_body: StreamPlaybackRequest,
    *,
    username: str,
    credentials: JwtAuthorizationCredentials,
    db: AsyncSession,
    rds: redis.Redis,
) -> tuple[dict[str, object], int]:
    """Create or update one playback session and return its API payload."""
    if not request_body.label:
        raise HTTPException(status_code=422, detail='label_required')

    await _authorise_label_access(
        credentials=credentials,
        db=db,
        label=request_body.label,
    )
    stream_name = await _resolve_configured_stream_name(
        db,
        request_body.label,
        stream_id=request_body.stream_id,
        key=request_body.key,
    )

    profile = _normalise_playback_profile(request_body.profile)
    rendition = _normalise_playback_rendition(request_body.rendition)
    language: str | None = None
    if profile == 'overlay':
        language = normalise_label_language(request_body.language)
        allowed_languages = _allowed_overlay_languages()
        if language not in allowed_languages:
            raise HTTPException(status_code=422, detail='unsupported_language')

        media_path = build_media_path(request_body.label, stream_name)
        if rendition == 'preview':
            media_path = build_preview_media_path(media_path)
        active_languages = await _active_overlay_languages(rds, media_path)
        if (
            language not in active_languages
            and len(active_languages) >= OVERLAY_MAX_ACTIVE_LANGUAGES
        ):
            raise HTTPException(
                status_code=429,
                detail='overlay_language_limit_reached',
            )

    session = await _create_or_update_playback_session(
        rds,
        session_id=request_body.session_id,
        username=username,
        label=request_body.label,
        stream_name=stream_name,
        profile=profile,
        rendition=rendition,
        language=language,
    )
    response_body = await _build_playback_session_response_body(rds, session)
    response_body['webrtc_url'] = build_media_webrtc_url(
        str(session['base_media_path']),
    )
    return (
        response_body,
        202 if response_body['state'] == 'starting' else 200,
    )


def _build_stream_playback_batch_response(
    *,
    items: list[dict[str, object]],
    status_code: int,
) -> JSONResponse:
    """Return a batch envelope whose items match single playback payloads."""
    response = JSONResponse(
        {
            'items': items,
            'count': len(items),
            'stream_playback_endpoint': STREAM_PLAYBACK_PUBLIC_BASE_PATH,
            'batch_endpoint': f'{STREAM_PLAYBACK_PUBLIC_BASE_PATH}/batch',
            'release_endpoint': f'{STREAM_PLAYBACK_PUBLIC_BASE_PATH}/release',
            'max_streams': MAX_STREAM_PLAYBACK_BATCH_STREAMS,
        },
        status_code=status_code,
    )
    return response


@router.post('/stream-playback')
async def request_stream_playback(
    request_body: StreamPlaybackRequest,
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
    db: AsyncSession = Depends(get_db),
    rds: redis.Redis = Depends(get_redis_pool),
) -> JSONResponse:
    """Return a clean or shared overlay playback URL for one camera."""
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')
    response_body, status_code = await _negotiate_stream_playback(
        request_body,
        username=username,
        credentials=credentials,
        db=db,
        rds=rds,
    )
    return JSONResponse(response_body, status_code=status_code)


def _model_field_was_set(model: object, field_name: str) -> bool:
    """Return whether a Pydantic model field was provided by the caller."""
    fields_set = getattr(model, 'model_fields_set', None)
    if fields_set is None:
        fields_set = getattr(model, '__fields_set__', set())
    return (
        isinstance(fields_set, (set, frozenset))
        and field_name in fields_set
    )


def _inherit_batch_playback_defaults(
    item: StreamPlaybackRequest,
    batch: StreamPlaybackBatchRequest,
) -> StreamPlaybackRequest:
    """Apply batch-level playback options to one explicit stream request."""
    return StreamPlaybackRequest(
        label=item.label or batch.label,
        stream_id=item.stream_id,
        key=item.key,
        session_id=item.session_id,
        profile=(
            item.profile
            if _model_field_was_set(item, 'profile')
            else batch.profile
        ),
        rendition=(
            item.rendition
            if _model_field_was_set(item, 'rendition')
            else batch.rendition
        ),
        language=(
            item.language
            if _model_field_was_set(item, 'language')
            else batch.language
        ),
        transport=(
            item.transport
            if _model_field_was_set(item, 'transport')
            else batch.transport
        ),
    )


async def _build_batch_playback_requests(
    request_body: StreamPlaybackBatchRequest,
    db: AsyncSession,
) -> list[StreamPlaybackRequest]:
    """Expand an explicit or site-level batch request into stream requests."""
    if request_body.streams:
        return [
            _inherit_batch_playback_defaults(item, request_body)
            for item in request_body.streams
        ]

    if not request_body.label:
        raise HTTPException(status_code=422, detail='label_required')

    result = await db.execute(
        _visible_stream_names_query(request_body.label),
    )
    stream_names = list(result.scalars().all())
    return [
        StreamPlaybackRequest(
            label=request_body.label,
            key=stream_name,
            profile=request_body.profile,
            rendition=request_body.rendition,
            language=request_body.language,
            transport=request_body.transport,
        )
        for stream_name in stream_names
    ]


def _enforce_stream_playback_batch_limit(
    requests: list[StreamPlaybackRequest],
) -> None:
    count = len(requests)
    if count <= MAX_STREAM_PLAYBACK_BATCH_STREAMS:
        return
    raise HTTPException(
        status_code=422,
        detail={
            'code': 'stream_batch_limit_exceeded',
            'count': count,
            'max_streams': MAX_STREAM_PLAYBACK_BATCH_STREAMS,
        },
    )


@router.post('/stream-playback/batch')
async def request_stream_playback_batch(
    request_body: StreamPlaybackBatchRequest,
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
    db: AsyncSession = Depends(get_db),
    rds: redis.Redis = Depends(get_redis_pool),
) -> JSONResponse:
    """Create stable playback sessions for multiple cameras at once."""
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    requests = await _build_batch_playback_requests(request_body, db)
    _enforce_stream_playback_batch_limit(requests)
    items: list[dict[str, object]] = []
    status_code = 200
    for stream_request in requests:
        body, item_status = await _negotiate_stream_playback(
            stream_request,
            username=username,
            credentials=credentials,
            db=db,
            rds=rds,
        )
        items.append(body)
        if item_status == 202:
            status_code = 202

    return _build_stream_playback_batch_response(
        items=items,
        status_code=status_code,
    )


@router.post('/stream-playback/release')
async def release_stream_playback(
    request_body: StreamPlaybackRequest,
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
    rds: redis.Redis = Depends(get_redis_pool),
) -> JSONResponse:
    """Stop refreshing a shared overlay request for one camera."""
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    if request_body.session_id:
        session = await _load_playback_session(rds, request_body.session_id)
        if session is None:
            raise HTTPException(status_code=404, detail='session_not_found')
        if session.get('username') != username:
            raise HTTPException(status_code=403, detail='session_forbidden')

        await _delete_playback_session_media_indexes(rds, session)
        await rds.delete(_playback_session_key(request_body.session_id))
        language = session.get('language')
        base_media_path = str(session.get('base_media_path') or '')
        if session.get('profile') == 'overlay' and isinstance(language, str):
            has_other_session = await _has_other_overlay_sessions(
                rds,
                released_session_id=request_body.session_id,
                base_media_path=base_media_path,
                language=language,
            )
            if not has_other_session:
                await rds.delete(
                    build_overlay_demand_key(base_media_path, language),
                )
        elif session.get('profile') == 'clean' and base_media_path:
            has_other_session = await _has_other_clean_sessions(
                rds,
                released_session_id=request_body.session_id,
                base_media_path=base_media_path,
            )
            if not has_other_session:
                await rds.delete(build_clean_demand_key(base_media_path))

        return JSONResponse({
            'status': 'released',
            'session_id': request_body.session_id,
            'profile': session.get('profile'),
        })

    raise HTTPException(status_code=422, detail='session_id_required')


@router.get('/streams/{label}')
async def get_streams_for_label_route(
    label: str,
    overlay: str | None = None,
    language: str | None = None,
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
    db: AsyncSession = Depends(get_db),
    rds: redis.Redis = Depends(get_redis_pool),
) -> JSONResponse:
    """Return stream display names and stable IDs for a site label."""
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    await _authorise_label_access(
        credentials=credentials,
        db=db,
        label=label,
    )

    overlay_mode = normalise_overlay_mode(overlay)
    overlay_language = normalise_label_language(language)
    if (
        overlay_mode == 'backend'
        and overlay_language not in _allowed_overlay_languages()
    ):
        raise HTTPException(status_code=422, detail='unsupported_language')

    profile = 'overlay' if overlay_mode == 'backend' else 'clean'
    language = overlay_language if profile == 'overlay' else None
    try:
        result = await db.execute(_visible_stream_names_query(label))
        stream_names = list(result.scalars().all())
    except Exception:
        stream_names = []

    streams = []
    for stream_name in stream_names:
        session = await _create_or_update_playback_session(
            rds,
            session_id=None,
            username=username,
            label=label,
            stream_name=stream_name,
            profile=profile,
            rendition='detail',
            language=language,
        )
        body = await _build_playback_session_response_body(rds, session)
        body['webrtc_url'] = build_media_webrtc_url(
            str(session['base_media_path']),
        )
        streams.append(body)
    return JSONResponse({'streams': streams})


async def _get_configured_media_streams(
    db: AsyncSession,
    label: str,
    rds: redis.Redis | None = None,
    overlay_mode: str = 'none',
    overlay_language: str | None = None,
) -> list[dict[str, object]]:
    """Return DB-configured streams with media-server playback URLs."""
    try:
        result = await db.execute(_visible_stream_names_query(label))
        stream_names = list(result.scalars().all())
    except Exception:
        return []
    streams: list[dict[str, object]] = []
    selected_language = normalise_label_language(overlay_language)
    for stream_name in stream_names:
        stream = _build_stream_listing(
            label,
            stream_name,
            Utils.encode(stream_name),
        )
        if overlay_mode == 'backend' and rds is not None:
            await _apply_overlay_listing_state(
                rds=rds,
                stream=stream,
                language=selected_language,
            )
        streams.append(stream)
    return streams


async def _apply_overlay_listing_state(
    rds: redis.Redis,
    stream: dict[str, object],
    language: str,
) -> None:
    """Request and describe the shared overlay stream for one listing."""
    media_path = str(stream['media_path'])
    overlay_path = build_annotated_media_path(media_path, language)
    overlay_hls_url = build_media_hls_url(overlay_path)

    await _touch_overlay_demand(rds, media_path, language)
    ready = await _overlay_is_ready(rds, overlay_path)

    stream.update(
        {
            'language': language,
            'profile': 'overlay',
            'status': 'ready' if ready else 'starting',
            'state': 'ready' if ready else 'starting',
            'overlay_ready': ready,
            'playback_ready': True,
            'media_path': overlay_path,
            'media_hls_url': overlay_hls_url,
            'demand_ttl_seconds': OVERLAY_DEMAND_TTL_SECONDS,
            'playback_url': overlay_hls_url,
        },
    )


def _build_stream_listing(
    label: str,
    key: str,
    stream_id: str,
) -> dict[str, object]:
    """Build the public stream listing payload for one camera."""
    media_path = build_media_path(label, key)
    stream: dict[str, object] = {
        'key': key,
        'stream_id': stream_id,
        'media_path': media_path,
    }
    hls_url = build_media_hls_url(media_path)
    webrtc_url = build_media_webrtc_url(media_path)
    playback_url = hls_url
    stream.update(
        {
            'transport': 'hls',
            'playback_url': playback_url,
            'media_hls_url': hls_url,
            'webrtc_url': webrtc_url,
            'profile': 'clean',
            'status': 'ready',
            'state': 'ready',
            'playback_ready': True,
        },
    )
    return stream


@router.get('/webrtc/ice-servers')
async def get_webrtc_ice_servers(
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
) -> dict[str, list[dict[str, object]]]:
    """Return authenticated ICE servers for Flutter/web peer setup."""
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')
    return {'iceServers': get_public_ice_servers(username)}


@router.get('/metadata/stream-id/{label}/{stream_id}')
async def metadata_stream_id(
    request: Request,
    label: str,
    stream_id: str,
    overlay: str | None = None,
    language: str | None = None,
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
    db: AsyncSession = Depends(get_db),
    rds: redis.Redis = Depends(get_redis_pool),
) -> StreamingResponse:
    """Stream warning/time metadata by stable stream id as SSE."""
    await _authorise_label_access(credentials, db, label)
    # The SSE generator uses Redis only; do not hold a DB connection open.
    await db.close()
    redis_key = f"stream_metadata:{Utils.encode(label)}|{stream_id}"
    overlay_ready_key: str | None = None
    overlay_ready_payload: dict[str, object] | None = None
    overlay_demand_key: str | None = None
    overlay_mode = normalise_overlay_mode(overlay)
    if overlay_mode == 'backend':
        overlay_language = normalise_label_language(language)
        if overlay_language not in _allowed_overlay_languages():
            raise HTTPException(status_code=422, detail='unsupported_language')
        stream_name = _normalise_stream_id(stream_id)
        media_path = build_media_path(label, stream_name)
        overlay_path = build_annotated_media_path(
            media_path,
            overlay_language,
        )
        overlay_hls_url = build_media_hls_url(overlay_path)
        overlay_ready_key = build_overlay_ready_key(overlay_path)
        overlay_demand_key = build_overlay_demand_key(
            media_path,
            overlay_language,
        )
        overlay_ready_payload = {
            'profile': 'overlay',
            'status': 'ready',
            'state': 'ready',
            'playback_ready': True,
            'overlay_ready': True,
            'language': overlay_language,
            'media_path': overlay_path,
            'playback_url': overlay_hls_url,
            'media_hls_url': overlay_hls_url,
        }
    return StreamingResponse(
        metadata_stream_generator(
            request,
            rds,
            redis_key,
            overlay_ready_key=overlay_ready_key,
            overlay_ready_payload=overlay_ready_payload,
            overlay_demand_key=overlay_demand_key,
            overlay_demand_ttl_seconds=OVERLAY_DEMAND_TTL_SECONDS,
            overlay_demand_refresh_seconds=max(
                1.0,
                OVERLAY_DEMAND_TTL_SECONDS / 2,
            ),
        ),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-store, no-cache, must-revalidate',
            'Pragma': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )


@router.websocket('/ws/metadata-id/{label}/{stream_id}')
async def websocket_metadata_stream_id(
    websocket: WebSocket,
    label: str,
    stream_id: str,
    rds: redis.Redis = Depends(get_redis_pool_ws),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Stream metadata updates for a configured stream id."""
    await handle_metadata_stream_id_ws(
        websocket=websocket,
        label=label,
        stream_id=stream_id,
        rds=rds,
        settings=settings,
        db=db,
    )
