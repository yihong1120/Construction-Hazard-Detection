from __future__ import annotations

import redis.asyncio as redis
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.session_store import get_media_session
from examples.streaming_web import playback_service
from examples.streaming_web import stream_catalog_service
from examples.streaming_web.media_paths import build_clean_demand_key
from examples.streaming_web.media_paths import build_media_path
from examples.streaming_web.media_paths import build_media_webrtc_url
from examples.streaming_web.media_paths import build_overlay_demand_key
from examples.streaming_web.media_paths import build_preview_media_path
from examples.streaming_web.overlay_renderer import normalise_label_language
from examples.streaming_web.overlay_renderer import normalise_overlay_mode
from examples.streaming_web.playback_demand import active_overlay_languages
from examples.streaming_web.playback_hls import authorise_label_access
from examples.streaming_web.playback_hls import extract_media_path_from_uri
from examples.streaming_web.playback_hls import extract_opaque_media_token
from examples.streaming_web.playback_hls import fetch_internal_hls_playlist
from examples.streaming_web.playback_hls import media_auth_401
from examples.streaming_web.playback_hls import (
    MEDIA_INTERNAL_HLS_TIMEOUT_SECONDS,
)
from examples.streaming_web.playback_hls import media_session_demand_ttl
from examples.streaming_web.playback_hls import (
    opaque_media_session_allows_path,
)
from examples.streaming_web.playback_hls import (
    rewrite_hls_playlist_media_urls,
)
from examples.streaming_web.playback_hls import split_hls_playlist_query
from examples.streaming_web.schemas import MAX_STREAM_PLAYBACK_BATCH_STREAMS
from examples.streaming_web.schemas import OverlayLanguageListResponse
from examples.streaming_web.schemas import PlaybackProfile
from examples.streaming_web.schemas import PlaybackRendition
from examples.streaming_web.schemas import PlaybackSessionResponse
from examples.streaming_web.schemas import StreamPlaybackBatchRequest
from examples.streaming_web.schemas import StreamPlaybackRequest
from examples.streaming_web.webrtc_service import get_public_ice_servers
from src.http_client_pool import HttpClientPool


def _username(credentials: JwtAuthorizationCredentials) -> str:
    """Extract the authenticated username from verified credentials.

    Args:
        credentials: Verified JWT credentials containing the subject claims.

    Returns:
        Non-empty authenticated username.

    Raises:
        HTTPException: If the verified token has no username claim.
    """
    username = credentials.subject.get('username')
    if not username:
        raise HTTPException(status_code=401, detail='Invalid token')
    return username


def get_overlay_languages(
    credentials: JwtAuthorizationCredentials,
) -> OverlayLanguageListResponse:
    """Return overlay-language options for an authenticated client.

    Args:
        credentials: Verified JWT credentials for the caller.

    Returns:
        Supported language and translation capability response.
    """
    _username(credentials)
    return playback_service._build_overlay_language_response()


async def authorise_media_request(
    request: Request,
    rds: redis.Redis,
) -> Response:
    """Authorise one MediaMTX request from an opaque capability.

    Args:
        request: Incoming media-proxy authorisation request.
        rds: Redis connection used to resolve the opaque capability.

    Returns:
        Empty no-store response when the request is within the capability
            scope.

    Raises:
        HTTPException: If the path, capability, or its scope is invalid.
    """
    original_uri = (
        request.headers.get('x-original-uri')
        or request.headers.get('x-forwarded-uri')
        or str(request.url.path)
    )
    media_path = extract_media_path_from_uri(original_uri)
    if not media_path.startswith('hazard_'):
        raise HTTPException(status_code=403, detail='Invalid media path')

    opaque_token = extract_opaque_media_token(request)
    if not opaque_token:
        raise media_auth_401('missing_media_token')
    opaque_session = await get_media_session(rds, opaque_token)
    if opaque_session is None:
        raise media_auth_401('expired_media_session')
    if opaque_session.get('user_active') is False:
        raise media_auth_401('inactive_user')
    if not opaque_media_session_allows_path(
        opaque_session,
        media_path,
    ):
        raise HTTPException(status_code=403, detail='media_scope_denied')
    # A successful fragment read is proof of active viewing, so renew both
    # producer demand and the corresponding browser playback session.
    await playback_service._touch_media_demand_from_media_path(
        rds,
        media_path,
        ttl_seconds=media_session_demand_ttl(opaque_session),
    )
    await playback_service._refresh_playback_sessions_for_media_path(
        rds,
        media_path,
    )
    return Response(
        status_code=204,
        headers={
            'Cache-Control': 'no-store',
            'X-Media-Auth-Mode': 'opaque_media_session',
        },
    )


async def stream_playback_session_playlist(
    session_id: str,
    request: Request,
    rds: redis.Redis,
) -> Response:
    """Serve a stable playlist whose child requests retain media authorisation.

    Args:
        session_id: Opaque stable playback session identifier.
        request: Client request that supplies an opaque media capability.
        rds: Redis connection used to load and refresh the session.

    Returns:
        Rewritten HLS playlist response with the required child-request state.

    Raises:
        HTTPException: If the session, capability, or upstream playlist fails.
    """
    session = await playback_service._load_playback_session(rds, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='session_not_found')

    auth_query, media_query = split_hls_playlist_query(
        request.url.query,
    )
    if not auth_query:
        raise media_auth_401('missing_media_token')

    await playback_service._refresh_playback_session_ttl(rds, session_id)
    state = await playback_service._select_session_playback(rds, session)
    await playback_service._wait_for_session_startup(session)
    media_path = extract_media_path_from_uri(
        str(state['hls_url']),
    )
    if not media_path.startswith('hazard_'):
        raise HTTPException(status_code=502, detail='invalid_media_playlist')
    http_clients = getattr(request.app.state, 'http_clients', None)
    http_client = None
    if isinstance(http_clients, HttpClientPool):
        http_client = await http_clients.get(
            'mediamtx-hls',
            timeout=MEDIA_INTERNAL_HLS_TIMEOUT_SECONDS,
            follow_redirects=True,
        )
    if http_client is None:
        playlist, hls_session_cookie = await fetch_internal_hls_playlist(
            media_path,
            media_query=media_query,
        )
    else:
        playlist, hls_session_cookie = await fetch_internal_hls_playlist(
            media_path,
            media_query=media_query,
            http_client=http_client,
        )
    # MediaMTX checks each child playlist and segment independently.  Preserve
    # the opaque capability on every URI emitted from the parent playlist.
    rewritten = rewrite_hls_playlist_media_urls(
        playlist,
        media_path=media_path,
        auth_query=auth_query,
    )
    has_media_uri = (
        any(
            line and not line.startswith('#')
            for line in rewritten.splitlines()
        )
        or 'URI="/hazard/media/' in rewritten
    )
    response = Response(
        content=rewritten,
        media_type='application/vnd.apple.mpegurl',
        headers={
            'Cache-Control': 'no-store',
            'X-Playback-Session': session_id,
            'X-Playback-Profile': session['profile'],
            'X-HLS-Media-Path': media_path,
            'X-HLS-Playlist-Lines': str(len(rewritten.splitlines())),
            'X-HLS-Playlist-Has-Media-URI': (
                'true' if has_media_uri else 'false'
            ),
        },
    )
    if hls_session_cookie:
        response.headers.append('Set-Cookie', hls_session_cookie)
    return response


async def negotiate_stream_playback(
    request_body: StreamPlaybackRequest,
    username: str,
    credentials: JwtAuthorizationCredentials,
    db: AsyncSession,
    rds: redis.Redis,
) -> tuple[PlaybackSessionResponse, int]:
    """Create or update a playback session and build its API payload.

    Args:
        request_body: Validated request for one camera stream.
        username: Authenticated owner of the resulting session.
        credentials: Verified JWT credentials used for site authorisation.
        db: Database session used to resolve site and stream access.
        rds: Redis connection used for demand and session state.

    Returns:
        Playback response body and HTTP status code, including ``202`` while
        an on-demand overlay publisher is starting.

    Raises:
        HTTPException: If the request, access, profile, or language is invalid.
    """
    if not request_body.label:
        raise HTTPException(status_code=422, detail='label_required')

    await authorise_label_access(
        credentials=credentials,
        db=db,
        label=request_body.label,
    )
    stream_name = await stream_catalog_service._resolve_configured_stream_name(
        db,
        request_body.label,
        stream_id=request_body.stream_id,
        key=request_body.key,
    )
    return await _negotiate_validated_stream_playback(
        request_body,
        username=username,
        stream_name=stream_name,
        rds=rds,
    )


async def _prepare_validated_playback_request(
    request_body: StreamPlaybackRequest,
    stream_name: str,
    rds: redis.Redis,
) -> tuple[PlaybackProfile, PlaybackRendition, str | None]:
    """Normalise one already-authorised, configured playback request."""
    label = request_body.label
    if not label:
        raise HTTPException(status_code=422, detail='label_required')
    profile = playback_service._normalise_playback_profile(
        request_body.profile,
    )
    rendition = playback_service._normalise_playback_rendition(
        request_body.rendition,
    )
    language: str | None = None
    if profile == 'overlay':
        language = normalise_label_language(request_body.language)
        allowed_languages = playback_service._allowed_overlay_languages()
        if language not in allowed_languages:
            raise HTTPException(status_code=422, detail='unsupported_language')

        media_path = build_media_path(label, stream_name)
        if rendition == 'preview':
            media_path = build_preview_media_path(media_path)
        active_languages = await active_overlay_languages(
            rds,
            media_path,
            allowed_languages,
        )
        if (
            language not in active_languages
            and len(active_languages)
            >= playback_service.OVERLAY_MAX_ACTIVE_LANGUAGES
        ):
            raise HTTPException(
                status_code=429,
                detail='overlay_language_limit_reached',
            )

    return profile, rendition, language


async def _negotiate_validated_stream_playback(
    request_body: StreamPlaybackRequest,
    *,
    username: str,
    stream_name: str,
    rds: redis.Redis,
) -> tuple[PlaybackSessionResponse, int]:
    """Negotiate after the caller has already checked access and stream ID."""
    label = request_body.label
    if not label:
        raise HTTPException(status_code=422, detail='label_required')
    profile, rendition, language = await _prepare_validated_playback_request(
        request_body,
        stream_name,
        rds,
    )
    session = await playback_service._create_or_update_playback_session(
        rds,
        session_id=request_body.session_id,
        username=username,
        label=label,
        stream_name=stream_name,
        profile=profile,
        rendition=rendition,
        language=language,
    )
    response_body = (
        await playback_service._build_playback_session_response_body(
            rds,
            session,
        )
    )
    response_body['webrtc_url'] = build_media_webrtc_url(
        session['base_media_path'],
    )
    return response_body, 202 if response_body['status'] == 'starting' else 200


def _model_field_was_set(
    model: StreamPlaybackRequest,
    field_name: str,
) -> bool:
    """Determine whether a Pydantic v2 field was explicitly supplied.

    Args:
        model: Stream request whose provided fields are inspected.
        field_name: Model field name to test.

    Returns:
        ``True`` when the client supplied the field rather than its default.
    """
    return field_name in model.model_fields_set


def _inherit_batch_playback_defaults(
    item: StreamPlaybackRequest,
    batch: StreamPlaybackBatchRequest,
) -> StreamPlaybackRequest:
    """Apply batch-level defaults to one explicit stream request.

    Args:
        item: Explicit stream request from the batch.
        batch: Parent batch containing fallback playback options.

    Returns:
        New request with inherited values only where the item omitted them.
    """
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
    """Expand explicit or site-level batch playback requests.

    Args:
        request_body: Validated batch playback request.
        db: Database session used to enumerate site streams when needed.

    Returns:
        Concrete one-stream playback requests.

    Raises:
        HTTPException: If no stream list or site label is provided.
    """
    if request_body.streams:
        return [
            _inherit_batch_playback_defaults(item, request_body)
            for item in request_body.streams
        ]
    if not request_body.label:
        raise HTTPException(status_code=422, detail='label_required')

    result = await db.execute(
        stream_catalog_service._visible_stream_names_query(request_body.label),
    )
    return [
        StreamPlaybackRequest(
            label=request_body.label,
            key=stream_name,
            profile=request_body.profile,
            rendition=request_body.rendition,
            language=request_body.language,
            transport=request_body.transport,
        )
        for stream_name in result.scalars().all()
    ]


def _enforce_stream_playback_batch_limit(
    requests: list[StreamPlaybackRequest],
) -> None:
    """Reject an oversized playback batch before allocating sessions.

    Args:
        requests: Concrete playback requests to validate.

    Raises:
        HTTPException: If the batch exceeds the configured stream limit.
    """
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


def _build_stream_playback_batch_response(
    items: list[PlaybackSessionResponse],
    status_code: int,
) -> JSONResponse:
    """Build a batch response using the single-playback payload contract.

    Args:
        items: Negotiated playback responses in request order.
        status_code: Overall response status, including a possible ``202``.

    Returns:
        JSON response with items, endpoints, and batch limit metadata.
    """
    base_path = playback_service.STREAM_PLAYBACK_PUBLIC_BASE_PATH
    return JSONResponse(
        {
            'items': items,
            'count': len(items),
            'stream_playback_endpoint': base_path,
            'batch_endpoint': f"{base_path}/batch",
            'release_endpoint': f"{base_path}/release",
            'max_streams': MAX_STREAM_PLAYBACK_BATCH_STREAMS,
        },
        status_code=status_code,
    )


async def request_stream_playback(
    request_body: StreamPlaybackRequest,
    credentials: JwtAuthorizationCredentials,
    db: AsyncSession,
    rds: redis.Redis,
) -> JSONResponse:
    """Negotiate a single camera playback session.

    Args:
        request_body: Validated one-stream playback request.
        credentials: Verified JWT credentials for the caller.
        db: Database session used for access checks.
        rds: Redis connection used for sessions and producer demand.

    Returns:
        JSON response for the negotiated session.
    """
    response_body, status_code = await negotiate_stream_playback(
        request_body,
        username=_username(credentials),
        credentials=credentials,
        db=db,
        rds=rds,
    )
    return JSONResponse(response_body, status_code=status_code)


async def request_stream_playback_batch(
    request_body: StreamPlaybackBatchRequest,
    credentials: JwtAuthorizationCredentials,
    db: AsyncSession,
    rds: redis.Redis,
) -> JSONResponse:
    """Negotiate playback sessions for a site or explicit stream list.

    Args:
        request_body: Validated batch playback request.
        credentials: Verified JWT credentials for the caller.
        db: Database session used for stream lookup and access checks.
        rds: Redis connection used for sessions and producer demand.

    Returns:
        JSON response containing negotiated sessions.
    """
    username = _username(credentials)
    authorised_labels: set[str] = set()
    # A site-wall request is expanded from the database.  Authorise its label
    # before enumerating streams, then reuse that check below.
    if not request_body.streams and request_body.label:
        await authorise_label_access(credentials, db, request_body.label)
        authorised_labels.add(request_body.label)
    requests = await _build_batch_playback_requests(request_body, db)
    _enforce_stream_playback_batch_limit(requests)
    labels = {request.label for request in requests if request.label}
    for label in sorted(labels.difference(authorised_labels)):
        await authorise_label_access(credentials, db, label)

    stream_names = (
        await stream_catalog_service.resolve_configured_stream_names(
            db,
            [
                (request.label or '', request.stream_id, request.key)
                for request in requests
            ],
        )
    )
    prepared = [
        await _prepare_validated_playback_request(request, stream_name, rds)
        for request, stream_name in zip(requests, stream_names, strict=True)
    ]

    # Camera walls create fresh sessions.  Store every session and activate all
    # demand leases with the existing pipeline/MGET batch helpers.  Explicit
    # session refreshes retain their stricter ownership/update path below.
    items: list[PlaybackSessionResponse | None] = [None] * len(requests)
    new_indexes = [
        index
        for index, request in enumerate(requests)
        if request.session_id is None
    ]
    if new_indexes:
        sessions = await playback_service.create_playback_sessions(
            rds,
            username=username,
            requests=[
                (
                    requests[index].label or '',
                    stream_names[index],
                    prepared[index][0],
                    prepared[index][1],
                    prepared[index][2],
                )
                for index in new_indexes
            ],
        )
        response_bodies = (
            await playback_service.build_playback_session_response_bodies(
                rds,
                sessions,
            )
        )
        for index, session, body in zip(
            new_indexes,
            sessions,
            response_bodies,
            strict=True,
        ):
            body['webrtc_url'] = build_media_webrtc_url(
                session['base_media_path'],
            )
            items[index] = body

    for index, request in enumerate(requests):
        if request.session_id is None:
            continue
        body, _item_status = await _negotiate_validated_stream_playback(
            request,
            username=username,
            stream_name=stream_names[index],
            rds=rds,
        )
        items[index] = body

    response_items = [item for item in items if item is not None]
    status_code = (
        202
        if any(item['status'] == 'starting' for item in response_items)
        else 200
    )
    return _build_stream_playback_batch_response(
        items=response_items,
        status_code=status_code,
    )


async def release_stream_playback(
    request_body: StreamPlaybackRequest,
    credentials: JwtAuthorizationCredentials,
    rds: redis.Redis,
) -> JSONResponse:
    """Release a playback session and unused producer demand.

    Args:
        request_body: Request identifying the session to release.
        credentials: Verified JWT credentials for the caller.
        rds: Redis connection used for sessions and producer demand.

    Returns:
        JSON status response for the released session.

    Raises:
        HTTPException: If the session identifier, session, or owner is invalid.
    """
    username = _username(credentials)
    session_id = request_body.session_id
    if not session_id:
        raise HTTPException(status_code=422, detail='session_id_required')
    session = await playback_service._load_playback_session(rds, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='session_not_found')
    if session['username'] != username:
        raise HTTPException(status_code=403, detail='session_forbidden')

    await playback_service._delete_playback_session_media_indexes(rds, session)
    await rds.delete(playback_service._playback_session_key(session_id))
    base_media_path = session['base_media_path']
    # Demand keys are shared by matching sessions, so remove one only after
    # establishing that no other live session still requires the producer.
    if session['profile'] == 'overlay':
        has_other_session = await playback_service._has_other_playback_session(
            rds,
            base_media_path=base_media_path,
            profile='overlay',
            language=session['language'],
        )
        if not has_other_session:
            await rds.delete(
                build_overlay_demand_key(base_media_path, session['language']),
            )
    else:
        has_other_session = await playback_service._has_other_playback_session(
            rds,
            base_media_path=base_media_path,
            profile='clean',
        )
        if not has_other_session:
            await rds.delete(build_clean_demand_key(base_media_path))
    return JSONResponse(
        {
            'status': 'released',
            'session_id': session_id,
            'profile': session['profile'],
        },
    )


async def get_streams_for_label(
    label: str,
    overlay: str | None,
    language: str | None,
    credentials: JwtAuthorizationCredentials,
    db: AsyncSession,
    rds: redis.Redis,
) -> JSONResponse:
    """Build stable playback records for every accessible site stream.

    Args:
        label: Site label whose streams are requested.
        overlay: Optional requested overlay mode.
        language: Optional requested overlay label language.
        credentials: Verified JWT credentials for the caller.
        db: Database session used for authorisation and stream lookup.
        rds: Redis connection used for sessions and producer demand.

    Returns:
        JSON response containing a session-backed descriptor for every stream.

    Raises:
        HTTPException: If the caller lacks access or the language is invalid.
    """
    username = _username(credentials)
    await authorise_label_access(credentials, db, label)
    overlay_mode = normalise_overlay_mode(overlay)
    overlay_language = normalise_label_language(language)
    if (
        overlay_mode == 'backend'
        and overlay_language
        not in playback_service._allowed_overlay_languages()
    ):
        raise HTTPException(status_code=422, detail='unsupported_language')

    profile: PlaybackProfile = (
        'overlay' if overlay_mode == 'backend' else 'clean'
    )
    selected_language = overlay_language if profile == 'overlay' else None
    result = await db.execute(
        stream_catalog_service._visible_stream_names_query(label),
    )
    stream_names = list(result.scalars().all())

    sessions = await playback_service.create_playback_sessions_for_streams(
        rds,
        username=username,
        label=label,
        stream_names=stream_names,
        profile=profile,
        rendition='detail',
        language=selected_language,
    )
    streams = await playback_service.build_playback_session_response_bodies(
        rds,
        sessions,
    )
    for session, body in zip(sessions, streams, strict=True):
        body['webrtc_url'] = build_media_webrtc_url(session['base_media_path'])
    return JSONResponse({'streams': streams})


def get_webrtc_ice_servers(
    credentials: JwtAuthorizationCredentials,
) -> dict[str, list[dict[str, object]]]:
    """Return authenticated ICE configuration for supported clients.

    Args:
        credentials: Verified JWT credentials for the caller.

    Returns:
        Browser-compatible ``iceServers`` configuration.
    """
    return {'iceServers': get_public_ice_servers(_username(credentials))}
