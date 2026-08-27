from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl
from urllib.parse import urlencode
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import httpx
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from fastapi.responses import JSONResponse
from jwt.exceptions import InvalidTokenError
from pydantic import TypeAdapter
from pydantic import ValidationError
from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.deployment_context import DeploymentBinding
from examples.auth.deployment_context import resolve_request_deployment
from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.session_store import auth_session_key
from examples.auth.session_store import create_media_session
from examples.auth.session_store import delete_media_session
from examples.auth.session_store import get_auth_session
from examples.auth.session_store import MEDIA_SESSION_TTL_SECONDS
from examples.auth.session_store import renew_media_session
from examples.auth.token_revocation import is_access_token_revoked
from examples.bff.proxy import get_proxy_access_token
from examples.bff.security import check_csrf
from examples.bff.security import SESSION_COOKIE
from examples.db_management.schemas.playback import PlaybackProfile
from examples.db_management.schemas.playback import PlaybackRenewRequest
from examples.db_management.schemas.playback import PlaybackSessionRequest
from examples.db_management.schemas.playback import PlaybackWallRequest
from examples.db_management.schemas.playback import (
    StreamingPlaybackBatchResponse,
)
from examples.db_management.schemas.playback import (
    StreamingPlaybackErrorResponse,
)
from examples.db_management.schemas.playback import StreamingPlaybackItem
from examples.streaming_web.media_paths import build_clean_demand_key
from examples.streaming_web.media_paths import build_media_path
from examples.streaming_web.media_paths import build_overlay_demand_key
from examples.streaming_web.media_paths import build_preview_media_path
from examples.streaming_web.overlay_renderer import normalise_label_language
from src.http_client_pool import HttpClientPool

_STREAMING_RESPONSE = TypeAdapter(dict[str, object])
STREAMING_PLAYBACK_API_URL = os.getenv(
    'PLAYBACK_STREAMING_API_URL',
    'http://127.0.0.1:8800',
).rstrip('/')
PLAYBACK_PUBLIC_BASE_PATH = '/hazard/api/db_management/api/playback'
PLAYBACK_UPSTREAM_TIMEOUT_SECONDS = float(
    os.getenv('PLAYBACK_UPSTREAM_TIMEOUT_SECONDS', '20'),
)


async def _request_http_client(
    request: Request,
) -> httpx.AsyncClient | None:
    """Return the app-lifetime streaming client when the lifespan is active."""
    http_clients = getattr(request.app.state, 'http_clients', None)
    if not isinstance(http_clients, HttpClientPool):
        return None
    return await http_clients.get(
        'streaming-playback',
        timeout=PLAYBACK_UPSTREAM_TIMEOUT_SECONDS,
    )


@dataclass(frozen=True)
class PlaybackPrincipal:
    """Represent an authenticated identity authorised for playback.

    Attributes:
        username: Authenticated account username.
        user_id: Database identifier of the authenticated user.
        parent: Session or native-user namespace owning media sessions.
        platform: Client platform used to select security behaviour.
        access_token: Validated access token forwarded to the streaming
            service.
    """

    username: str
    user_id: int
    parent: str
    platform: str
    access_token: str


def _bearer_token(request: Request) -> str | None:
    """Extract a bearer token from an API request.

    Args:
        request: Request whose authorisation header is inspected.

    Returns:
        Stripped bearer token, or ``None`` for a non-bearer header.
    """
    scheme, _, value = request.headers.get('authorization', '').partition(' ')
    return value.strip() if scheme.lower() == 'bearer' and value else None


async def _decode_access_token(
    token: str,
    redis: Redis,
    db: AsyncSession,
    deployment: DeploymentBinding,
) -> JwtAuthorizationCredentials:
    """Validate a non-revoked access token for playback.

    Args:
        token: Raw bearer access token.
        redis: Redis connection used for revocation checks.

    Returns:
        Decoded credentials with validated subject and payload.

    Raises:
        HTTPException: If the token is invalid, revoked, or revocation state is
            unavailable.
    """
    try:
        credentials = await jwt_access.decode_access_token_for_deployment(
            token,
            db,
            deployment,
        )
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=401,
            detail='Could not validate credentials',
            headers={'WWW-Authenticate': 'Bearer'},
        ) from exc
    try:
        if await is_access_token_revoked(
            redis,
            {'jti': credentials.subject['jti']},
        ):
            raise HTTPException(
                status_code=401,
                detail='Could not validate credentials',
                headers={'WWW-Authenticate': 'Bearer'},
            )
    except RedisError as exc:
        raise HTTPException(
            status_code=503,
            detail='Authentication revocation service unavailable',
        ) from exc
    return credentials


async def _resolve_playback_principal(
    request: Request,
    redis: Redis,
    db: AsyncSession,
) -> PlaybackPrincipal:
    """Resolve native bearer or BFF-cookie credentials to playback scope.

    Args:
        request: Request containing a bearer token or BFF session cookie.
        redis: Redis connection holding session and revocation state.

    Returns:
        Authenticated playback principal bound to a media-session owner.

    Raises:
        HTTPException: If authentication, session lookup, or CSRF validation
            fails.
    """
    deployment = await resolve_request_deployment(request, db)
    bearer = _bearer_token(request)
    if bearer:
        credentials = await _decode_access_token(
            bearer,
            redis,
            db,
            deployment,
        )
        user_id = credentials.subject['user_id']
        return PlaybackPrincipal(
            username=credentials.subject['username'],
            user_id=user_id,
            parent=f"native:user:{user_id}",
            platform='native',
            access_token=bearer,
        )
    session_id = request.cookies.get(SESSION_COOKIE)
    app_session = await get_auth_session(redis, session_id)
    if not session_id or app_session is None:
        raise HTTPException(status_code=401, detail='app_session_expired')
    check_csrf(request, app_session, request.headers.get('x-csrf-token'))
    access_token, _ = await get_proxy_access_token(
        redis,
        session_id,
        deployment=deployment,
    )
    credentials = await _decode_access_token(
        access_token,
        redis,
        db,
        deployment,
    )
    return PlaybackPrincipal(
        username=credentials.subject['username'],
        user_id=credentials.subject['user_id'],
        parent=auth_session_key(session_id),
        platform='web',
        access_token=access_token,
    )


def _streaming_detail(response: httpx.Response) -> object:
    """Extract meaningful error detail from a streaming-service response.

    Args:
        response: Upstream HTTP response containing an error body.

    Returns:
        Parsed structured detail, raw JSON body, or response text.
    """
    try:
        body = response.json()
    except ValueError:
        return response.text or 'streaming_upstream_error'
    try:
        return StreamingPlaybackErrorResponse.model_validate(body).detail
    except ValidationError:
        return body


async def _post_streaming_playback(
    path: str,
    *,
    principal: PlaybackPrincipal,
    payload: dict[str, object],
    http_client: httpx.AsyncClient | None = None,
) -> tuple[dict[str, object], int]:
    """Call the streaming playback API using the caller's access token.

    Args:
        path: Upstream playback endpoint path.
        principal: Authenticated caller whose token is forwarded upstream.
        payload: Validated upstream request body.

    Returns:
        Validated JSON response object and upstream HTTP status code.

    Raises:
        HTTPException: If the upstream service is unavailable, rejects the
            request, or returns malformed JSON.
    """
    try:
        if http_client is not None:
            response = await http_client.post(
                f"{STREAMING_PLAYBACK_API_URL}{path}",
                json=payload,
                headers={'Authorization': f"Bearer {principal.access_token}"},
            )
        else:
            async with httpx.AsyncClient(
                timeout=PLAYBACK_UPSTREAM_TIMEOUT_SECONDS,
            ) as client:
                response = await client.post(
                    f"{STREAMING_PLAYBACK_API_URL}{path}",
                    json=payload,
                    headers={
                        'Authorization': f"Bearer {principal.access_token}",
                    },
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
        return (
            _STREAMING_RESPONSE.validate_python(response.json()),
            response.status_code,
        )
    except (ValueError, ValidationError) as exc:
        raise HTTPException(
            status_code=502,
            detail='invalid_streaming_upstream_response',
        ) from exc


def _normalise_profile(profile: PlaybackProfile | str | None) -> str:
    """Normalise a playback profile to a supported render mode.

    Args:
        profile: Requested profile value from client or stored state.

    Returns:
        Normalised ``clean`` or ``overlay`` profile.

    Raises:
        HTTPException: If the supplied profile is unsupported.
    """
    value = (profile or 'clean').strip().lower()
    if value not in {'clean', 'overlay'}:
        raise HTTPException(status_code=422, detail='unsupported_profile')
    return value


def _with_media_token(url: object, media_token: str) -> str:
    """Attach the scoped media token to a playback URL.

    Args:
        url: Upstream playback URL to sign.
        media_token: Scoped token authorising access to the media session.

    Returns:
        URL with stale media-token parameters removed and the new token added.
    """
    parts = urlsplit(str(url))
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key not in {'mt', 'media_token'}
    ]
    query.append(('mt', media_token))
    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urlencode(
                query,
            ),
            parts.fragment,
        ),
    )


def _signed_stream_item(
    item: dict[str, object],
    media_token: str,
) -> dict[str, object]:
    """Add the scoped media token to HLS and playback URLs.

    Args:
        item: Upstream stream item containing media URLs.
        media_token: Scoped token authorising access to the media session.

    Returns:
        Copy of the stream item with signed HLS and playback URLs.
    """
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
    """Return the public playback endpoint URLs.

    Returns:
        Mapping of single-session, wall, and renewal endpoint URLs.
    """
    return {
        'single_endpoint': f"{PLAYBACK_PUBLIC_BASE_PATH}/sessions",
        'wall_endpoint': f"{PLAYBACK_PUBLIC_BASE_PATH}/walls",
        'renew_endpoint': f"{PLAYBACK_PUBLIC_BASE_PATH}/sessions/renew",
    }


def _streaming_session_descriptors(
    stream_items: list[StreamingPlaybackItem],
) -> dict[str, dict[str, object]]:
    """Build trusted upstream session data for later HLS recovery.

    Args:
        stream_items: Validated stream items supplied by the upstream service.

    Returns:
        Mapping keyed by upstream session identifier.
    """
    return {
        item.session_id: {
            'label': item.label,
            'stream_name': item.key,
            'profile': item.profile,
            'rendition': item.rendition,
            'language': item.language,
        }
        for item in stream_items
    }


def _playback_demand_keys(
    *,
    site: str,
    cameras: list[str],
    profile: str,
    quality: str,
    language: str | None,
) -> list[str]:
    """Build producer-demand leases for a scoped playback capability.

    Args:
        site: Site containing the requested cameras.
        cameras: Camera keys authorised by the upstream response.
        profile: Normalised clean or overlay render profile.
        quality: Detail or preview rendition quality.
        language: Optional label language for overlay rendering.

    Returns:
        Exact demand keys required while the media session remains active.

    Raises:
        ValueError: If an unsupported profile reaches this internal helper.
    """
    paths = [build_media_path(site, camera) for camera in cameras]
    if quality == 'preview':
        paths = [build_preview_media_path(path) for path in paths]
    if profile == 'clean':
        return [build_clean_demand_key(path) for path in paths]
    if profile == 'overlay':
        label_language = normalise_label_language(language)
        return [
            build_overlay_demand_key(path, label_language) for path in paths
        ]
    raise ValueError(f"unsupported playback profile: {profile}")


async def _create_scoped_media_session(
    redis: Redis,
    *,
    principal: PlaybackPrincipal,
    site: str,
    cameras: list[str],
    profile: str,
    quality: str,
    language: str | None,
    playback_sessions: dict[str, dict[str, object]],
) -> tuple[str, dict[str, Any]]:
    """Create a media session scoped to validated playback streams.

    Args:
        redis: Redis connection used to store the media session.
        principal: Authenticated caller that owns the media session.
        site: Site containing the authorised cameras.
        cameras: Camera keys authorised by the upstream response.
        profile: Normalised clean or overlay render profile.
        quality: Detail or preview rendition quality.
        language: Optional label language for overlay rendering.
        playback_sessions: Trusted upstream sessions bound to this capability.

    Returns:
        Scoped media token and the persisted media-session record.
    """
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
        'playback_sessions': playback_sessions,
    }
    if len(cameras) == 1:
        kwargs['camera'] = cameras[0]
    else:
        kwargs['cameras'] = cameras
    return await create_media_session(redis, **kwargs)


def _single_response_body(
    *,
    media_session: dict[str, object],
    stream_item: dict[str, object],
    media_token: str,
    site: str,
    camera: str,
) -> dict[str, object]:
    """Build the public response body for one playback stream.

    Args:
        media_session: Persisted scoped media session.
        stream_item: Validated upstream stream item.
        media_token: Scoped token used to sign media URLs.
        site: Requested site label.
        camera: Resolved camera key.

    Returns:
        Single-camera playback response body.
    """
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
    """Build the public response body for a multi-camera playback wall.

    Args:
        media_session: Persisted scoped media session.
        stream_items: Validated upstream stream items.
        media_token: Scoped token used to sign media URLs.
        site: Requested site label.
        profile: Normalised render profile.
        max_streams: Upstream maximum concurrent streams.

    Returns:
        Responsive multi-camera playback response body.
    """
    items: list[dict[str, object]] = []
    for item in stream_items:
        signed = _signed_stream_item(item, media_token)
        camera = str(signed.get('key') or '')
        items.append(
            {
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
            },
        )
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
    payload: PlaybackSessionRequest,
    principal: PlaybackPrincipal,
    redis: Redis,
    http_client: httpx.AsyncClient | None = None,
) -> tuple[dict[str, object], int]:
    """Create one detail-quality playback media capability.

    Args:
        payload: Validated single-camera playback request.
        principal: Authenticated caller that will own the session.
        redis: Redis connection used to create media-session state.

    Returns:
        Signed single-playback body and upstream HTTP status code.

    Raises:
        HTTPException: If upstream validation or media-session creation fails.
    """
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
        http_client=http_client,
    )
    try:
        stream = StreamingPlaybackItem.model_validate(stream_item)
    except ValidationError as exc:
        raise HTTPException(
            status_code=502,
            detail='invalid_streaming_upstream_response',
        ) from exc
    token, session = await _create_scoped_media_session(
        redis,
        principal=principal,
        site=payload.site,
        cameras=[stream.key],
        profile=profile,
        quality='detail',
        language=stream.language or payload.language,
        playback_sessions=_streaming_session_descriptors([stream]),
    )
    return (
        _single_response_body(
            media_session=session,
            stream_item=stream.model_dump(),
            media_token=token,
            site=payload.site,
            camera=stream.key,
        ),
        status_code,
    )


def _wall_upstream_payload(
    payload: PlaybackWallRequest,
    profile: str,
) -> dict[str, object]:
    """Build the streaming-service request for a preview playback wall.

    Args:
        payload: Validated multi-camera wall request.
        profile: Normalised clean or overlay render profile.

    Returns:
        Upstream request body for either default or explicitly scoped cameras.
    """
    body: dict[str, object] = {
        'label': payload.site,
        'profile': profile,
        'rendition': 'preview',
        'language': payload.language,
        'transport': payload.transport,
    }
    if payload.cameras:
        del body['label']
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
    payload: PlaybackWallRequest,
    principal: PlaybackPrincipal,
    redis: Redis,
    http_client: httpx.AsyncClient | None = None,
) -> tuple[dict[str, object], int]:
    """Create a preview-quality media capability for a camera wall.

    Args:
        payload: Validated multi-camera playback request.
        principal: Authenticated caller that will own the session.
        redis: Redis connection used to create media-session state.

    Returns:
        Signed wall-playback body and upstream HTTP status code.

    Raises:
        HTTPException: If no cameras are authorised, upstream validation fails,
            or media-session creation fails.
    """
    profile = _normalise_profile(payload.profile)
    upstream, status_code = await _post_streaming_playback(
        '/stream-playback/batch',
        principal=principal,
        payload=_wall_upstream_payload(payload, profile),
        http_client=http_client,
    )
    try:
        batch = StreamingPlaybackBatchResponse.model_validate(upstream)
    except ValidationError as exc:
        raise HTTPException(
            status_code=502,
            detail='invalid_streaming_upstream_response',
        ) from exc
    streams = batch.items
    cameras = [stream.key for stream in streams]
    if not cameras:
        raise HTTPException(status_code=404, detail='cameras_not_found')
    token, session = await _create_scoped_media_session(
        redis,
        principal=principal,
        site=payload.site,
        cameras=cameras,
        profile=profile,
        quality='preview',
        language=next(
            (
                stream.language
                for stream in streams
                if stream.language is not None
            ),
            payload.language,
        ),
        playback_sessions=_streaming_session_descriptors(streams),
    )
    return (
        _wall_response_body(
            media_session=session,
            stream_items=[stream.model_dump() for stream in streams],
            media_token=token,
            site=payload.site,
            profile=profile,
            max_streams=batch.max_streams,
        ),
        status_code,
    )


async def playback_session_response(
    payload: PlaybackSessionRequest,
    request: Request,
    redis: Redis,
    db: AsyncSession,
) -> JSONResponse:
    """Create a signed single-camera playback response.

    Args:
        payload: Validated single-camera playback request.
        request: HTTP request used to authenticate the caller.
        redis: Redis connection used for authentication and media state.

    Returns:
        Non-cacheable signed single-camera playback response.
    """
    principal = await _resolve_playback_principal(request, redis, db)
    body, status_code = await _create_single_playback(
        payload,
        principal,
        redis,
        await _request_http_client(request),
    )
    return JSONResponse(
        body,
        status_code=status_code,
        headers={'Cache-Control': 'no-store'},
    )


async def playback_wall_response(
    payload: PlaybackWallRequest,
    request: Request,
    redis: Redis,
    db: AsyncSession,
) -> JSONResponse:
    """Create signed playback responses for a camera wall.

    Args:
        payload: Validated multi-camera playback request.
        request: HTTP request used to authenticate the caller.
        redis: Redis connection used for authentication and media state.

    Returns:
        Non-cacheable signed multi-camera playback response.
    """
    principal = await _resolve_playback_principal(request, redis, db)
    body, status_code = await _create_wall_playback(
        payload,
        principal,
        redis,
        await _request_http_client(request),
    )
    return JSONResponse(
        body,
        status_code=status_code,
        headers={'Cache-Control': 'no-store'},
    )


async def renew_playback_response(
    payload: PlaybackRenewRequest,
    request: Request,
    redis: Redis,
    db: AsyncSession,
) -> JSONResponse:
    """Renew an existing signed playback session.

    Args:
        payload: Public identifier of the media session to renew.
        request: HTTP request used to authenticate the caller.
        redis: Redis connection holding authentication and media state.

    Returns:
        Non-cacheable response confirming renewal without changing HLS URLs.

    Raises:
        HTTPException: If the session expired or is not owned by the caller.
    """
    principal = await _resolve_playback_principal(request, redis, db)
    current = await renew_media_session(
        redis, payload.id, owner=principal.parent,
    )
    if current is None:
        raise HTTPException(status_code=401, detail='expired_media_session')
    return JSONResponse(
        {
            'id': str(current['id']),
            'mode': (
                'multi_stream' if current.get('scope') == 'batch' else 'single'
            ),
            'renew_endpoint': _playback_endpoints()['renew_endpoint'],
            'expires_in': MEDIA_SESSION_TTL_SECONDS,
            'renewed': True,
            'hls_urls_changed': False,
        },
        headers={'Cache-Control': 'no-store'},
    )


async def delete_playback_response(
    session_id: str,
    request: Request,
    redis: Redis,
    db: AsyncSession,
) -> Response:
    """Delete a signed playback session owned by the caller.

    Args:
        session_id: Public media-session identifier to delete.
        request: HTTP request used to authenticate the caller.
        redis: Redis connection holding authentication and media state.

    Returns:
        Empty non-cacheable successful deletion response.

    Raises:
        HTTPException: If the session is unavailable or not owned by the
            caller.
    """
    principal = await _resolve_playback_principal(request, redis, db)
    if not await delete_media_session(
        redis, session_id, owner=principal.parent,
    ):
        raise HTTPException(status_code=404, detail='session_not_found')
    return Response(status_code=204, headers={'Cache-Control': 'no-store'})
