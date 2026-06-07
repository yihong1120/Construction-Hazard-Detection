from __future__ import annotations

import os
import time
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Literal
from urllib.parse import parse_qs
from urllib.parse import unquote
from urllib.parse import urlsplit

import jwt
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
from jwt.exceptions import ExpiredSignatureError
from jwt.exceptions import InvalidTokenError
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
from examples.auth.user_service import load_user_access_context
from examples.local_notification_server.lang_config import LANGUAGES
from examples.streaming_web.backend.media_paths import (
    build_annotated_media_path,
)
from examples.streaming_web.backend.media_paths import build_media_hls_url
from examples.streaming_web.backend.media_paths import build_media_path
from examples.streaming_web.backend.media_paths import build_media_webrtc_url
from examples.streaming_web.backend.media_paths import (
    build_overlay_demand_key,
)
from examples.streaming_web.backend.media_paths import build_overlay_ready_key
from examples.streaming_web.backend.media_paths import decode_media_segment
from examples.streaming_web.backend.media_paths import encode_media_segment
from examples.streaming_web.backend.media_paths import OVERLAY_DEMAND_PREFIX
from examples.streaming_web.backend.media_paths import (
    parse_annotated_media_path,
)
from examples.streaming_web.backend.overlay_renderer import CLASS_LABELS
from examples.streaming_web.backend.overlay_renderer import LANGUAGE_ALIASES
from examples.streaming_web.backend.overlay_renderer import (
    normalise_label_language,
)
from examples.streaming_web.backend.overlay_renderer import (
    normalise_overlay_mode,
)
from examples.streaming_web.backend.overlay_renderer import (
    SUPPORTED_LABEL_LANGUAGES,
)
from examples.streaming_web.backend.overlay_renderer import WARNING_LABELS
from examples.streaming_web.backend.schemas import LabelListResponse
from examples.streaming_web.backend.schemas import OverlayLanguageInfo
from examples.streaming_web.backend.schemas import OverlayLanguageListResponse
from examples.streaming_web.backend.schemas import StreamPlaybackRequest
from examples.streaming_web.backend.utils import Utils
from examples.streaming_web.backend.webrtc_service import (
    get_public_ice_servers,
)
from examples.streaming_web.backend.ws_handlers import (
    handle_metadata_stream_id_ws,
)
from examples.streaming_web.backend.ws_handlers import (
    metadata_stream_generator,
)

# Module-level alias retained for test patching
get_user_and_sites = load_user_access_context


# Settings and router
settings: Settings = Settings()
router: APIRouter = APIRouter()

MEDIA_TOKEN_GRACE_SECONDS = int(
    os.getenv('HAZARD_MEDIA_TOKEN_GRACE_SECONDS', '300'),
)


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment setting.

    Args:
        name: Environment variable name.
        default: Value used when the setting is absent.

    Returns:
        Parsed boolean value.
    """
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


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


MEDIA_SESSION_TTL_SECONDS = int(
    os.getenv('HAZARD_MEDIA_SESSION_TTL_SECONDS', str(75 * 60)),
)
MEDIA_SESSION_COOKIE_NAME = os.getenv(
    'HAZARD_MEDIA_SESSION_COOKIE_NAME',
    'hazard_media_session',
)
MEDIA_SESSION_COOKIE_SECURE = _env_bool(
    'HAZARD_MEDIA_SESSION_COOKIE_SECURE',
    True,
)
MEDIA_SESSION_COOKIE_SAMESITE = os.getenv(
    'HAZARD_MEDIA_SESSION_COOKIE_SAMESITE',
    'lax',
).lower()
MEDIA_SESSION_EXPOSE_TOKEN = _env_bool(
    'HAZARD_MEDIA_SESSION_EXPOSE_TOKEN',
    False,
)
OVERLAY_DEMAND_TTL_SECONDS = int(
    os.getenv('MEDIA_OVERLAY_DEMAND_TTL_SECONDS', '90'),
)
OVERLAY_MAX_ACTIVE_LANGUAGES = max(
    1,
    int(os.getenv('MEDIA_OVERLAY_MAX_ACTIVE_LANGUAGES_PER_STREAM', '5')),
)
MediaAuthKind = Literal['access', 'media_session']


async def _authorise_label_access(
    credentials: JwtAuthorizationCredentials,
    db: AsyncSession,
    label: str,
) -> None:
    """Raise an HTTP error when the authenticated user cannot view a label."""
    username: str | None = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    _, user_site_names, user_role = await get_user_and_sites(db, username)
    if user_role != 'super_admin' and label not in user_site_names:
        raise HTTPException(status_code=403, detail='Access denied')


def _extract_media_auth_token(request: Request) -> str:
    """Return a viewer JWT from header, query string, or same-origin cookie."""
    auth_header = request.headers.get('authorization', '')
    scheme, _, value = auth_header.partition(' ')
    if scheme.lower() == 'bearer' and value:
        return value.strip()

    query_token = request.query_params.get('token')
    if query_token:
        return query_token

    for header_name in ('x-original-uri', 'x-forwarded-uri'):
        original_uri = request.headers.get(header_name)
        if not original_uri:
            continue
        original_query = parse_qs(urlsplit(original_uri).query)
        original_tokens = original_query.get('token') or []
        if original_tokens and original_tokens[0]:
            return original_tokens[0]

    cookie_token = request.cookies.get('hazard_access_token')
    if cookie_token:
        return cookie_token
    return request.cookies.get('access_token') or ''


def _extract_media_session_token(request: Request) -> str:
    """Return the signed media session token carried by cookie or query."""
    query_token = request.query_params.get('media_session')
    if query_token:
        return query_token

    for header_name in ('x-original-uri', 'x-forwarded-uri'):
        original_uri = request.headers.get(header_name)
        if not original_uri:
            continue
        original_query = parse_qs(urlsplit(original_uri).query)
        original_tokens = original_query.get('media_session') or []
        if original_tokens and original_tokens[0]:
            return original_tokens[0]

    return request.cookies.get(MEDIA_SESSION_COOKIE_NAME) or ''


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


def _create_media_session_token(username: str) -> str:
    """Create a signed token dedicated to long-running HLS media playback."""
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=MEDIA_SESSION_TTL_SECONDS)
    payload = {
        'typ': 'hazard_media_session',
        'sub': username,
        'iat': now,
        'exp': expires_at,
    }
    return jwt.encode(
        payload,
        settings.authjwt_secret_key,
        algorithm=settings.ALGORITHM,
    )


def _decode_media_session_token(token: str) -> dict[str, object]:
    """Decode a signed media session token and enforce its intended use."""
    try:
        payload = jwt.decode(
            token,
            settings.authjwt_secret_key,
            algorithms=[settings.ALGORITHM],
        )
    except ExpiredSignatureError as exc:
        raise _media_auth_401('expired_media_session') from exc
    except InvalidTokenError as exc:
        raise _media_auth_401('invalid_media_session') from exc

    if payload.get('typ') != 'hazard_media_session':
        raise _media_auth_401('invalid_media_session')
    username = payload.get('sub')
    if not isinstance(username, str) or not username:
        raise _media_auth_401('invalid_media_session')
    return payload


def _decode_media_auth_token(token: str) -> dict[str, object]:
    """Decode a media token, allowing a short expiry grace for HLS races."""
    try:
        return jwt_access.decode_token(token)
    except ExpiredSignatureError as exc:
        try:
            payload = jwt_access.decode_token(token, verify_exp=False)
        except InvalidTokenError as invalid_exc:
            raise _media_auth_401('invalid_token') from invalid_exc

        exp = payload.get('exp')
        if not isinstance(exp, (int, float)):
            raise _media_auth_401('invalid_token') from exc
        if time.time() - float(exp) > MEDIA_TOKEN_GRACE_SECONDS:
            raise _media_auth_401('expired_token') from exc
        return payload
    except InvalidTokenError as exc:
        raise _media_auth_401('invalid_token') from exc


def _resolve_media_auth_identity(
    request: Request,
) -> tuple[str, MediaAuthKind]:
    """Resolve the viewer identity from access token or media session token."""
    access_token = _extract_media_auth_token(request)
    if access_token:
        payload = _decode_media_auth_token(access_token)
        subject = payload.get('subject')
        if not isinstance(subject, dict):
            subject = {}
        username = subject.get('username') or payload.get('sub')
        if not isinstance(username, str) or not username:
            raise _media_auth_401('invalid_token')
        return username, 'access'

    media_session = _extract_media_session_token(request)
    if media_session:
        payload = _decode_media_session_token(media_session)
        return str(payload['sub']), 'media_session'

    raise _media_auth_401('missing_token')


def _set_media_session_cookie(response: Response, token: str) -> None:
    """Attach a media session cookie suitable for HLS fragment requests."""
    response.set_cookie(
        key=MEDIA_SESSION_COOKIE_NAME,
        value=token,
        max_age=MEDIA_SESSION_TTL_SECONDS,
        httponly=True,
        secure=MEDIA_SESSION_COOKIE_SECURE,
        samesite=MEDIA_SESSION_COOKIE_SAMESITE,  # type: ignore[arg-type]
        path='/',
    )


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
) -> None:
    """Keep a shared overlay profile alive while a viewer uses it."""
    await rds.set(
        build_overlay_demand_key(media_path, label_language),
        str(int(time.time())).encode('ascii'),
        ex=OVERLAY_DEMAND_TTL_SECONDS,
    )


async def _touch_overlay_demand_from_media_path(
    rds: redis.Redis,
    media_path: str,
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
        await _touch_overlay_demand(rds, base_path, language)
    except Exception as exc:
        print(
            f"Failed to renew overlay demand for {media_path}: {exc}",
            flush=True,
        )


async def _overlay_is_ready(
    rds: redis.Redis,
    overlay_media_path: str,
) -> bool:
    """Return whether the producer recently published the overlay path."""
    return bool(await rds.exists(build_overlay_ready_key(overlay_media_path)))


def _normalise_stream_id(value: str) -> str:
    """Decode stream ids generated by older and newer listing payloads."""
    decoded = Utils.decode(value)
    if decoded != value:
        return decoded
    try:
        return decode_media_segment(value)
    except Exception:
        return value


async def _resolve_configured_stream_name(
    db: AsyncSession,
    label: str,
    stream_id: str | None,
    key: str | None,
) -> str:
    """Resolve and validate a stream name configured for a site."""
    requested_name = key or (
        _normalise_stream_id(
            stream_id,
        ) if stream_id else ''
    )
    if not requested_name:
        raise HTTPException(
            status_code=422, detail='stream_id_or_key_required',
        )

    result = await db.execute(
        select(StreamConfigModel.stream_name).join(Site).where(
            Site.name == label,
        ),
    )
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


@router.post('/media-session', include_in_schema=False)
async def issue_media_session(
    response: Response,
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
    db: AsyncSession = Depends(get_db),
) -> dict[str, int | str]:
    """
    Issue or renew a signed HLS media session for long-running playback.

    Clients should call this after login and after each access-token refresh.
    HLS requests can then be authorised by the HttpOnly cookie even when a
    native player cannot reliably attach Authorization headers to fragments.
    """
    username: str | None = credentials.subject.get('username')
    if not username:
        raise _media_auth_401('invalid_token')

    user, _, _ = await get_user_and_sites(db, username)
    if getattr(user, 'status', USER_STATUS_ACTIVE) != USER_STATUS_ACTIVE:
        raise _media_auth_401('inactive_user')

    token = _create_media_session_token(username)
    _set_media_session_cookie(response, token)
    result: dict[str, int | str] = {
        'expires_in': MEDIA_SESSION_TTL_SECONDS,
        'token_type': 'hazard_media_session',
    }
    if MEDIA_SESSION_EXPOSE_TOKEN:
        result['media_session_token'] = token
    return result


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
    db: AsyncSession = Depends(get_db),
    rds: redis.Redis = Depends(get_redis_pool),
) -> Response:
    """Authorise Nginx auth_request subrequests for MediaMTX playback."""
    username, auth_kind = _resolve_media_auth_identity(request)

    original_uri = (
        request.headers.get('x-original-uri')
        or request.headers.get('x-forwarded-uri')
        or str(request.url.path)
    )
    media_path = _extract_media_path_from_uri(original_uri)
    if not media_path.startswith('hazard_'):
        raise HTTPException(status_code=403, detail='Invalid media path')

    user, user_site_names, user_role = await get_user_and_sites(db, username)
    if getattr(user, 'status', USER_STATUS_ACTIVE) != USER_STATUS_ACTIVE:
        raise _media_auth_401('inactive_user')

    if user_role == 'super_admin' or any(
        _media_path_matches_site(media_path, site)
        for site in user_site_names
    ):
        await _touch_overlay_demand_from_media_path(rds, media_path)
        return Response(
            status_code=204,
            headers={
                'Cache-Control': 'no-store',
                'X-Media-Auth-Mode': auth_kind,
            },
        )

    raise HTTPException(status_code=403, detail='Access denied')


@router.post('/stream-playback')
async def request_stream_playback(
    request_body: StreamPlaybackRequest,
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
    db: AsyncSession = Depends(get_db),
    rds: redis.Redis = Depends(get_redis_pool),
) -> JSONResponse:
    """Return a clean or shared overlay playback URL for one camera."""
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

    media_path = build_media_path(request_body.label, stream_name)
    hls_url = build_media_hls_url(media_path)
    response_body: dict[str, object] = {
        'transport': 'hls',
        'status': 'ready',
        'overlay': False,
        'language': None,
        'media_path': media_path,
        'playback_url': hls_url,
        'hls_url': hls_url,
        'clean_hls_url': hls_url,
        'webrtc_url': build_media_webrtc_url(media_path),
        'default_language': _default_overlay_language(),
        'supported_languages': list(_allowed_overlay_languages()),
        'language_options': _overlay_language_option_payloads(),
    }

    overlay_mode = normalise_overlay_mode(
        str(request_body.overlay)
        if request_body.overlay is not None
        else 'none',
    )
    if overlay_mode != 'backend':
        return JSONResponse(response_body)

    language = normalise_label_language(
        request_body.language or request_body.lang,
    )
    allowed_languages = _allowed_overlay_languages()
    if language not in allowed_languages:
        raise HTTPException(status_code=422, detail='unsupported_language')

    active_languages = await _active_overlay_languages(rds, media_path)
    if (
        language not in active_languages
        and len(active_languages) >= OVERLAY_MAX_ACTIVE_LANGUAGES
    ):
        raise HTTPException(
            status_code=429,
            detail='overlay_language_limit_reached',
        )

    await _touch_overlay_demand(rds, media_path, language)
    overlay_path = build_annotated_media_path(media_path, language)
    overlay_hls_url = build_media_hls_url(overlay_path)
    ready = await _overlay_is_ready(rds, overlay_path)
    response_body.update(
        {
            'status': 'ready' if ready else 'starting',
            'overlay': True,
            'language': language,
            'media_path': overlay_path,
            'base_media_path': media_path,
            'playback_url': overlay_hls_url,
            'hls_url': overlay_hls_url,
            'overlay_hls_url': overlay_hls_url,
            'overlay_media_path': overlay_path,
            'demand_ttl_seconds': OVERLAY_DEMAND_TTL_SECONDS,
            'retry_after_seconds': 2 if not ready else 0,
        },
    )
    return JSONResponse(
        response_body,
        status_code=200 if ready else 202,
    )


@router.get('/streams/{label}')
async def get_streams_for_label_route(
    label: str,
    credentials: JwtAuthorizationCredentials = Security(jwt_access),
    db: AsyncSession = Depends(get_db),
) -> dict[str, list[dict[str, str]]]:
    """Return stream display names and stable IDs for a site label."""
    await _authorise_label_access(
        credentials=credentials,
        db=db,
        label=label,
    )

    streams = await _get_configured_media_streams(db, label)
    return {'streams': streams}


async def _get_configured_media_streams(
    db: AsyncSession,
    label: str,
) -> list[dict[str, str]]:
    """Return DB-configured streams with media-server playback URLs."""
    try:
        result = await db.execute(
            select(StreamConfigModel.stream_name).join(Site).where(
                Site.name == label,
            ),
        )
        stream_names = list(result.scalars().all())
    except Exception:
        return []
    return [
        _build_stream_listing(label, stream_name, Utils.encode(stream_name))
        for stream_name in stream_names
    ]


def _build_stream_listing(
    label: str,
    key: str,
    stream_id: str,
) -> dict[str, str]:
    """Build the public stream listing payload for one camera."""
    media_path = build_media_path(label, key)
    stream = {
        'key': key,
        'stream_id': stream_id,
        'media_path': media_path,
    }
    hls_url = build_media_hls_url(media_path)
    webrtc_url = build_media_webrtc_url(media_path)
    overlay_language = normalise_label_language(
        os.getenv('MEDIA_DEFAULT_OVERLAY_LANGUAGE', 'zh-TW'),
    )
    annotated_path = build_annotated_media_path(
        media_path,
        overlay_language,
    )
    annotated_hls_url = build_media_hls_url(annotated_path)
    has_annotated_stream = _env_bool('MEDIA_PUBLISH_ANNOTATED_STREAM', True)
    require_annotated_playback = _env_bool(
        'MEDIA_REQUIRE_ANNOTATED_PLAYBACK',
        False,
    )
    playback_url = (
        annotated_hls_url
        if has_annotated_stream and require_annotated_playback
        else hls_url
    )
    stream.update(
        {
            'transport': 'hls',
            'playback_url': playback_url,
            'hls_url': hls_url,
            'webrtc_url': webrtc_url,
            'annotated_media_path': annotated_path,
            'annotated_hls_url': annotated_hls_url,
            'annotated_playback_url': annotated_hls_url,
            'overlay_language': overlay_language,
            'overlay_playback_endpoint': '/hazard/api/stream-playback',
            'supported_overlay_languages': ','.join(
                _allowed_overlay_languages(),
            ),
            'has_annotated_stream': str(has_annotated_stream).lower(),
            'require_annotated_playback': str(
                require_annotated_playback,
            ).lower(),
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
    rds: redis.Redis = Depends(get_redis_pool),
) -> StreamingResponse:
    """Stream warning/time metadata by stable stream id as SSE."""
    redis_key = f"stream_metadata:{Utils.encode(label)}|{stream_id}"
    return StreamingResponse(
        metadata_stream_generator(request, rds, redis_key),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-store, no-cache, must-revalidate',
            'Pragma': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
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
