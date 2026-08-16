from __future__ import annotations

import redis.asyncio as redis
from fastapi import Request
from fastapi import Response
from fastapi import WebSocket
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.streaming_web import stream_catalog_service
from examples.streaming_web import streaming_api_service
from examples.streaming_web import streaming_metadata_service
from examples.streaming_web.schemas import LabelListResponse
from examples.streaming_web.schemas import OverlayLanguageListResponse
from examples.streaming_web.schemas import StreamPlaybackBatchRequest
from examples.streaming_web.schemas import StreamPlaybackRequest


class StreamCatalogueRequestService:
    """Coordinate catalogue queries for one authenticated database request.

    Attributes:
        credentials: Verified credentials for the requesting user.
        db: Request-scoped database session used for access and stream queries.
    """

    def __init__(
        self,
        credentials: JwtAuthorizationCredentials,
        db: AsyncSession,
    ) -> None:
        """Bind authenticated catalogue dependencies for one request.

        Args:
            credentials: Verified credentials for the requesting user.
            db: Request-scoped database session used for access and stream queries.
        """
        self.credentials = credentials
        self.db = db

    async def visible_labels(self) -> LabelListResponse:
        """List site labels visible to the bound authenticated user.

        Returns:
            Validated labels available to the authenticated user.
        """
        return await stream_catalog_service.get_visible_labels(
            self.credentials,
            self.db,
        )


class StreamingCapabilityService:
    """Provide streaming capabilities for one authenticated user.

    Attributes:
        credentials: Verified credentials for the requesting user.
    """

    def __init__(self, credentials: JwtAuthorizationCredentials) -> None:
        """Bind verified credentials to user-specific capability responses.

        Args:
            credentials: Verified credentials for the requesting user.
        """
        self.credentials = credentials

    def overlay_languages(self) -> OverlayLanguageListResponse:
        """Return overlay-language capabilities for the bound user.

        Returns:
            Supported overlay languages, aliases, labels, and lease limits.
        """
        return streaming_api_service.get_overlay_languages(self.credentials)

    def ice_servers(self) -> dict[str, list[dict[str, object]]]:
        """Return ICE configuration for the bound authenticated user.

        Returns:
            Browser-compatible STUN and optional TURN configuration.
        """
        return streaming_api_service.get_webrtc_ice_servers(self.credentials)


class MediaPlaybackProxyService:
    """Coordinate MediaMTX proxy operations for one Redis connection.

    Attributes:
        rds: Request-scoped Redis connection used for capability and session state.
    """

    def __init__(self, rds: redis.Redis) -> None:
        """Bind Redis playback state to media-proxy operations.

        Args:
            rds: Request-scoped Redis connection used for media state.
        """
        self.rds = rds

    async def authorise_media(self, request: Request) -> Response:
        """Authorise one MediaMTX proxy request.

        Args:
            request: Incoming media-proxy authorisation request.

        Returns:
            Empty success response or a raised authorisation failure.
        """
        return await streaming_api_service.authorise_media_request(
            request,
            self.rds,
        )

    async def session_playlist(
        self,
        session_id: str,
        request: Request,
    ) -> Response:
        """Serve an authorised stable HLS playlist for one session.

        Args:
            session_id: Opaque playback session identifier.
            request: Incoming client playlist request.

        Returns:
            Rewritten HLS playlist response.
        """
        return await streaming_api_service.stream_playback_session_playlist(
            session_id,
            request,
            self.rds,
        )


class PlaybackReleaseService:
    """Coordinate session release for one authenticated Redis request.

    Attributes:
        credentials: Verified credentials for the requesting user.
        rds: Request-scoped Redis connection used for playback state.
    """

    def __init__(
        self,
        credentials: JwtAuthorizationCredentials,
        rds: redis.Redis,
    ) -> None:
        """Bind release-operation dependencies for one request.

        Args:
            credentials: Verified credentials for the requesting user.
            rds: Request-scoped Redis connection used for playback state.
        """
        self.credentials = credentials
        self.rds = rds

    async def release(self, request_body: StreamPlaybackRequest) -> JSONResponse:
        """Release one playback session owned by the bound user.

        Args:
            request_body: Request identifying the playback session.

        Returns:
            JSON status response for the released session.
        """
        return await streaming_api_service.release_stream_playback(
            request_body,
            self.credentials,
            self.rds,
        )


class StreamingRequestService:
    """Coordinate authorised HTTP streaming operations for one request.

    Attributes:
        credentials: Verified credentials for the requesting user.
        db: Request-scoped database session used for access checks.
        rds: Request-scoped Redis connection used for playback state.
    """

    def __init__(
        self,
        credentials: JwtAuthorizationCredentials,
        db: AsyncSession,
        rds: redis.Redis,
    ) -> None:
        """Bind request-scoped dependencies to streaming use cases.

        Args:
            credentials: Verified credentials for the requesting user.
            db: Request-scoped database session used for access checks.
            rds: Request-scoped Redis connection used for playback state.
        """
        self.credentials = credentials
        self.db = db
        self.rds = rds

    async def request_playback(
        self,
        request_body: StreamPlaybackRequest,
    ) -> JSONResponse:
        """Create or update one playback session for the bound user.

        Args:
            request_body: Validated one-stream playback request.

        Returns:
            JSON response describing the stable playback session.
        """
        return await streaming_api_service.request_stream_playback(
            request_body,
            self.credentials,
            self.db,
            self.rds,
        )

    async def request_playback_batch(
        self,
        request_body: StreamPlaybackBatchRequest,
    ) -> JSONResponse:
        """Create or update batch playback sessions for the bound user.

        Args:
            request_body: Validated batch playback request.

        Returns:
            JSON response containing ordered playback sessions.
        """
        return await streaming_api_service.request_stream_playback_batch(
            request_body,
            self.credentials,
            self.db,
            self.rds,
        )

    async def streams_for_label(
        self,
        label: str,
        overlay: str | None,
        language: str | None,
    ) -> JSONResponse:
        """Build playback descriptors for accessible streams in one site.

        Args:
            label: Site label whose streams are requested.
            overlay: Optional requested overlay mode.
            language: Optional requested overlay label language.

        Returns:
            JSON response containing session-backed stream descriptors.
        """
        return await streaming_api_service.get_streams_for_label(
            label,
            overlay,
            language,
            self.credentials,
            self.db,
            self.rds,
        )

    async def metadata_event_stream(
        self,
        request: Request,
        label: str,
        stream_id: str,
        overlay: str | None,
        language: str | None,
    ) -> StreamingResponse:
        """Create an authorised metadata SSE response for one stream.

        Args:
            request: Incoming SSE request used for disconnect detection.
            label: Site label containing the stream.
            stream_id: Encoded configured stream identifier.
            overlay: Optional requested overlay mode.
            language: Optional requested overlay label language.

        Returns:
            Non-buffered server-sent-event response for live metadata.
        """
        return await streaming_metadata_service.metadata_stream_response(
            request,
            label,
            stream_id,
            overlay,
            language,
            self.credentials,
            self.db,
            self.rds,
        )


class StreamingMetadataSocketService:
    """Coordinate metadata WebSocket delivery for one connection.

    Attributes:
        db: Request-scoped database session used to enforce site access.
        rds: WebSocket-scoped Redis connection used to consume frame metadata.
    """

    def __init__(self, db: AsyncSession, rds: redis.Redis) -> None:
        """Bind connection-scoped dependencies to metadata WebSocket handling.

        Args:
            db: Database session used to enforce site access.
            rds: Redis connection used to consume metadata records.
        """
        self.db = db
        self.rds = rds

    async def serve(
        self,
        websocket: WebSocket,
        label: str,
        stream_id: str,
    ) -> None:
        """Serve one metadata WebSocket using the bound connection resources.

        Args:
            websocket: Client WebSocket to authenticate and serve.
            label: Site label containing the stream.
            stream_id: Encoded configured stream identifier.
        """
        await streaming_metadata_service.metadata_stream_websocket(
            websocket=websocket,
            label=label,
            stream_id=stream_id,
            rds=self.rds,
            db=self.db,
        )
