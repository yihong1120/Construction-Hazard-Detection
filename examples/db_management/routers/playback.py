from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl
from urllib.parse import urlencode
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import httpx
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from fastapi.responses import JSONResponse
from jwt.exceptions import InvalidTokenError
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.database import get_db
from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.models import User
from examples.auth.redis_pool import get_redis_pool
from examples.auth.session_store import auth_session_key
from examples.auth.session_store import create_media_session
from examples.auth.session_store import delete_media_session
from examples.auth.session_store import get_auth_session
from examples.auth.session_store import MEDIA_SESSION_TTL_SECONDS
from examples.auth.session_store import renew_media_session
from examples.bff.proxy import get_proxy_access_token
from examples.bff.security import check_csrf
from examples.bff.security import SESSION_COOKIE
from examples.db_management.schemas.playback import PlaybackProfile
from examples.db_management.schemas.playback import PlaybackRenewRequest
from examples.db_management.schemas.playback import PlaybackSessionRequest
from examples.db_management.schemas.playback import PlaybackWallRequest
from examples.streaming_web.media_paths import build_clean_demand_key
from examples.streaming_web.media_paths import build_media_path
from examples.streaming_web.media_paths import build_overlay_demand_key
from examples.streaming_web.media_paths import build_preview_media_path
from examples.streaming_web.overlay_renderer import normalise_label_language

router = APIRouter(prefix='/api/playback', tags=['playback'])

STREAMING_PLAYBACK_API_URL = os.getenv(
    'PLAYBACK_STREAMING_API_URL',
    'http://127.0.0.1:8800',
).rstrip('/')
PLAYBACK_PUBLIC_BASE_PATH = '/hazard/api/db_management/api/playback'
PLAYBACK_UPSTREAM_TIMEOUT_SECONDS = float(
    os.getenv('PLAYBACK_UPSTREAM_TIMEOUT_SECONDS', '20'),
)


@dataclass(frozen=True)
class PlaybackPrincipal:
    username: str
    user_id: int
    parent: str
    platform: str
    access_token: str


def _bearer_token(request: Request) -> str | None:
    auth_header = request.headers.get('authorization', '')
    scheme, _, value = auth_header.partition(' ')
    if scheme.lower() == 'bearer' and value:
        return value.strip()
    return None


def _decode_access_token(token: str) -> JwtAuthorizationCredentials:
    try:
        payload = jwt_access.decode_token(token)
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=401,
            detail='Could not validate credentials',
            headers={'WWW-Authenticate': 'Bearer'},
        ) from exc

    subject = payload.get('subject')
    if not isinstance(subject, dict):
        sub = payload.get('sub')
        subject = {'username': sub} if isinstance(sub, str) else {}
    if not subject:
        raise HTTPException(
            status_code=401,
            detail='Could not validate credentials',
            headers={'WWW-Authenticate': 'Bearer'},
        )
    return JwtAuthorizationCredentials(
        subject=subject,
        payload=payload,
        token=token,
    )


def _subject_user_id(subject: dict[str, Any]) -> int | None:
    raw_user_id = subject.get('user_id')
    try:
        user_id = int(str(raw_user_id)) if raw_user_id is not None else 0
    except (TypeError, ValueError):
        user_id = 0
    return user_id if user_id > 0 else None


async def _load_user_id_by_username(
    db: AsyncSession,
    username: str,
) -> int:
    user_id = await db.scalar(select(User.id).where(User.username == username))
    try:
        value = int(user_id or 0)
    except (TypeError, ValueError):
        value = 0
    if value <= 0:
        raise HTTPException(status_code=401, detail='invalid_user')
    return value


async def _resolve_playback_principal(
    request: Request,
    db: AsyncSession,
    redis: Redis,
) -> PlaybackPrincipal:
    bearer = _bearer_token(request)
    if bearer:
        credentials = _decode_access_token(bearer)
        username = credentials.subject.get('username')
        if not isinstance(username, str) or not username:
            raise HTTPException(status_code=401, detail='Invalid token')
        user_id = _subject_user_id(credentials.subject)
        if user_id is None:
            user_id = await _load_user_id_by_username(db, username)
        return PlaybackPrincipal(
            username=username,
            user_id=user_id,
            parent=f'native:user:{user_id}',
            platform='native',
            access_token=bearer,
        )

    session_id = request.cookies.get(SESSION_COOKIE)
    app_session = await get_auth_session(redis, session_id)
    if not session_id or app_session is None:
        raise HTTPException(status_code=401, detail='app_session_expired')

    check_csrf(request, app_session, request.headers.get('x-csrf-token'))
    access_token, _ = await get_proxy_access_token(redis, session_id)
    credentials = _decode_access_token(access_token)
    username = credentials.subject.get('username')
    if not isinstance(username, str) or not username:
        raise HTTPException(status_code=401, detail='Invalid token')

    raw_user = app_session.get('user')
    session_user = raw_user if isinstance(raw_user, dict) else {}
    user_id = _subject_user_id(credentials.subject)
    if user_id is None:
        try:
            user_id = int(session_user.get('id') or 0)
        except (TypeError, ValueError):
            user_id = 0
    if user_id <= 0:
        user_id = await _load_user_id_by_username(db, username)

    return PlaybackPrincipal(
        username=username,
        user_id=user_id,
        parent=auth_session_key(session_id),
        platform='web',
        access_token=access_token,
    )


def _streaming_detail(response: httpx.Response) -> object:
    try:
        body = response.json()
    except ValueError:
        return response.text or 'streaming_upstream_error'
    if isinstance(body, dict) and 'detail' in body:
        return body['detail']
    return body


async def _post_streaming_playback(
    path: str,
    *,
    principal: PlaybackPrincipal,
    payload: dict[str, object],
) -> tuple[dict[str, object], int]:
    url = f'{STREAMING_PLAYBACK_API_URL}{path}'
    try:
        async with httpx.AsyncClient(
            timeout=PLAYBACK_UPSTREAM_TIMEOUT_SECONDS,
        ) as client:
            response = await client.post(
                url,
                json=payload,
                headers={'Authorization': f'Bearer {principal.access_token}'},
            )
    except (httpx.TimeoutException, httpx.NetworkError) as exc:
        raise HTTPException(
            status_code=502,
            detail='streaming_upstream_unavailable',
        ) from exc

    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail=_streaming_detail(response),
        )
    try:
        body = response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail='invalid_streaming_upstream_response',
        ) from exc
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=502,
            detail='invalid_streaming_upstream_response',
        )
    return body, response.status_code


def _normalise_profile(profile: PlaybackProfile | str | None) -> str:
    value = (profile or 'clean').strip().lower()
    if value in {'clean', 'overlay'}:
        return value
    raise HTTPException(status_code=422, detail='unsupported_profile')


def _with_media_token(url: object, media_token: str) -> str:
    parts = urlsplit(str(url))
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key not in {'mt', 'media_token'}
    ]
    query.append(('mt', media_token))
    return urlunsplit((
        parts.scheme,
        parts.netloc,
        parts.path,
        urlencode(query),
        parts.fragment,
    ))


def _signed_stream_item(
    item: dict[str, object],
    media_token: str,
) -> dict[str, object]:
    signed = dict(item)
    media_hls_url = _with_media_token(
        item.get('media_hls_url', ''),
        media_token,
    )
    signed['media_hls_url'] = media_hls_url
    if item.get('playback_url'):
        signed['playback_url'] = _with_media_token(
            item['playback_url'],
            media_token,
        )
        signed['hls_url'] = signed['playback_url']
    else:
        signed['hls_url'] = media_hls_url
    return signed


def _playback_endpoints() -> dict[str, str]:
    return {
        'single_endpoint': f'{PLAYBACK_PUBLIC_BASE_PATH}/sessions',
        'wall_endpoint': f'{PLAYBACK_PUBLIC_BASE_PATH}/walls',
        'renew_endpoint': f'{PLAYBACK_PUBLIC_BASE_PATH}/sessions/renew',
    }


async def _create_scoped_media_session(
    redis: Redis,
    *,
    principal: PlaybackPrincipal,
    site: str,
    cameras: list[str],
    profile: str,
    quality: str,
    language: str | None,
) -> tuple[str, dict[str, Any]]:
    kwargs: dict[str, Any] = {
        'user_id': principal.user_id,
        'username': principal.username,
        'site': site,
        'profile': profile,
        'parent': principal.parent,
        'platform': principal.platform,
        'language': language,
        'quality': quality,
        'purpose': 'playback',
        'demand_keys': _playback_demand_keys(
            site=site,
            cameras=cameras,
            profile=profile,
            quality=quality,
            language=language,
        ),
    }
    if len(cameras) == 1:
        kwargs['camera'] = cameras[0]
    else:
        kwargs['cameras'] = cameras
    return await create_media_session(redis, **kwargs)


def _playback_demand_keys(
    *,
    site: str,
    cameras: list[str],
    profile: str,
    quality: str,
    language: str | None,
) -> list[str]:
    """Build exact producer leases for one scoped playback capability."""
    media_paths = [build_media_path(site, camera) for camera in cameras]
    if quality == 'preview':
        media_paths = [build_preview_media_path(path) for path in media_paths]
    if profile == 'clean':
        return [build_clean_demand_key(path) for path in media_paths]
    if profile == 'overlay':
        label_language = normalise_label_language(language)
        return [
            build_overlay_demand_key(path, label_language)
            for path in media_paths
        ]
    raise ValueError(f'unsupported playback profile: {profile}')


def _single_response_body(
    *,
    media_session: dict[str, object],
    stream_item: dict[str, object],
    media_token: str,
    site: str,
    camera: str,
) -> dict[str, object]:
    signed = _signed_stream_item(stream_item, media_token)
    return {
        **_playback_endpoints(),
        **signed,
        'id': str(media_session['id']),
        'mode': 'single',
        'site': site,
        'camera': camera,
        'title': camera,
        'quality': 'detail',
        'token_transport': 'query',
        'expires_in': MEDIA_SESSION_TTL_SECONDS,
    }


def _wall_response_body(
    *,
    media_session: dict[str, object],
    stream_items: list[dict[str, object]],
    media_token: str,
    site: str,
    profile: str,
    max_streams: int | None,
) -> dict[str, object]:
    items: list[dict[str, object]] = []
    for item in stream_items:
        signed = _signed_stream_item(item, media_token)
        camera = str(signed.get('key') or '')
        items.append({
            'camera': camera,
            'title': camera,
            'detail_camera': camera,
            'stream_id': signed.get('stream_id'),
            'session_id': signed.get('session_id'),
            'status': signed.get('status'),
            'state': signed.get('state'),
            'profile': signed.get('profile'),
            'language': signed.get('language'),
            'overlay_ready': signed.get('overlay_ready'),
            'preview_hls_url': signed['hls_url'],
            'hls_url': signed['hls_url'],
            'playback_url': signed.get('playback_url'),
        })

    return {
        **_playback_endpoints(),
        'id': str(media_session['id']),
        'mode': 'multi_stream',
        'layout': 'responsive',
        'site': site,
        'quality': 'preview',
        'profile': profile,
        'token_transport': 'query',
        'expires_in': MEDIA_SESSION_TTL_SECONDS,
        'count': len(items),
        'max_streams': max_streams,
        'items': items,
    }


async def _create_single_playback(
    *,
    payload: PlaybackSessionRequest,
    principal: PlaybackPrincipal,
    redis: Redis,
) -> tuple[dict[str, object], int]:
    profile = _normalise_profile(payload.profile)
    stream_item, status_code = await _post_streaming_playback(
        '/stream-playback',
        principal=principal,
        payload={
            'label': payload.site,
            'key': payload.camera,
            'session_id': payload.session_id,
            'profile': profile,
            'rendition': 'detail',
            'language': payload.language,
            'transport': payload.transport,
        },
    )
    camera = str(stream_item.get('key') or payload.camera)
    media_token, media_session = await _create_scoped_media_session(
        redis,
        principal=principal,
        site=payload.site,
        cameras=[camera],
        profile=profile,
        quality='detail',
        language=(
            str(stream_item['language'])
            if stream_item.get('language')
            else payload.language
        ),
    )
    return (
        _single_response_body(
            media_session=media_session,
            stream_item=stream_item,
            media_token=media_token,
            site=payload.site,
            camera=camera,
        ),
        status_code,
    )


def _wall_upstream_payload(
    payload: PlaybackWallRequest,
    profile: str,
) -> dict[str, object]:
    body: dict[str, object] = {
        'label': payload.site,
        'profile': profile,
        'rendition': 'preview',
        'language': payload.language,
        'transport': payload.transport,
    }
    if payload.cameras:
        body.pop('label', None)
        body['streams'] = [
            {
                'label': payload.site,
                'key': camera,
                'profile': profile,
                'rendition': 'preview',
                'language': payload.language,
                'transport': payload.transport,
            }
            for camera in payload.cameras
        ]
    return body


async def _create_wall_playback(
    *,
    payload: PlaybackWallRequest,
    principal: PlaybackPrincipal,
    redis: Redis,
) -> tuple[dict[str, object], int]:
    profile = _normalise_profile(payload.profile)
    upstream, status_code = await _post_streaming_playback(
        '/stream-playback/batch',
        principal=principal,
        payload=_wall_upstream_payload(payload, profile),
    )
    raw_items = upstream.get('items')
    if not isinstance(raw_items, list):
        raise HTTPException(
            status_code=502,
            detail='invalid_streaming_upstream_response',
        )
    stream_items = [
        item for item in raw_items
        if isinstance(item, dict)
    ]
    cameras = [
        str(item.get('key') or '')
        for item in stream_items
        if item.get('key')
    ]
    if not cameras:
        raise HTTPException(status_code=404, detail='cameras_not_found')
    media_token, media_session = await _create_scoped_media_session(
        redis,
        principal=principal,
        site=payload.site,
        cameras=cameras,
        profile=profile,
        quality='preview',
        language=next(
            (
                str(item['language'])
                for item in stream_items
                if item.get('language')
            ), payload.language,
        ),
    )
    max_streams = upstream.get('max_streams')
    return (
        _wall_response_body(
            media_session=media_session,
            stream_items=stream_items,
            media_token=media_token,
            site=payload.site,
            profile=profile,
            max_streams=(
                int(max_streams)
                if isinstance(max_streams, int)
                else None
            ),
        ),
        status_code,
    )


@router.post('/sessions')
async def create_playback_session(
    payload: PlaybackSessionRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis_pool),
) -> JSONResponse:
    principal = await _resolve_playback_principal(request, db, redis)
    body, status_code = await _create_single_playback(
        payload=payload,
        principal=principal,
        redis=redis,
    )
    return JSONResponse(
        body,
        status_code=status_code,
        headers={'Cache-Control': 'no-store'},
    )


@router.post('/walls')
async def create_playback_wall(
    payload: PlaybackWallRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis_pool),
) -> JSONResponse:
    principal = await _resolve_playback_principal(request, db, redis)
    body, status_code = await _create_wall_playback(
        payload=payload,
        principal=principal,
        redis=redis,
    )
    return JSONResponse(
        body,
        status_code=status_code,
        headers={'Cache-Control': 'no-store'},
    )


@router.post('/sessions/renew')
async def renew_playback_session(
    payload: PlaybackRenewRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis_pool),
) -> JSONResponse:
    principal = await _resolve_playback_principal(request, db, redis)
    current = await renew_media_session(
        redis,
        payload.id,
        owner=principal.parent,
    )
    if current is None:
        raise HTTPException(status_code=401, detail='expired_media_session')
    return JSONResponse(
        {
            'id': str(current['id']),
            'mode': (
                'multi_stream'
                if current.get('scope') == 'batch'
                else 'single'
            ),
            'renew_endpoint': _playback_endpoints()['renew_endpoint'],
            'expires_in': MEDIA_SESSION_TTL_SECONDS,
            'renewed': True,
            'hls_urls_changed': False,
        },
        headers={'Cache-Control': 'no-store'},
    )


@router.delete('/sessions/{session_id}', status_code=204)
async def delete_playback_session(
    session_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis_pool),
) -> Response:
    principal = await _resolve_playback_principal(request, db, redis)
    deleted = await delete_media_session(
        redis,
        session_id,
        owner=principal.parent,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail='session_not_found')
    return Response(status_code=204, headers={'Cache-Control': 'no-store'})
