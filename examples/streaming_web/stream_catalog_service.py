from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy import Select
from sqlalchemy import select
from sqlalchemy import tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.auth.models import Site
from examples.auth.models import StreamConfig as StreamConfigModel
from examples.auth.user_service import load_user_access_context
from examples.streaming_web.media_paths import decode_media_segment
from examples.streaming_web.schemas import LabelListResponse


async def get_visible_labels(
    credentials: JwtAuthorizationCredentials,
    db: AsyncSession,
) -> LabelListResponse:
    """Return site labels visible to the authenticated user.

    Args:
        credentials: Verified JWT credentials for the caller.
        db: Database session used to resolve access and sites.

    Returns:
        Validated labels available to the caller.

    Raises:
        HTTPException: If the token does not identify a username.
    """
    username = credentials.subject.get('username')
    if not username:
        raise HTTPException(
            status_code=401,
            detail='Invalid token: no subject',
        )

    _, user_site_names, user_role = await load_user_access_context(
        db,
        username,
    )
    statement = select(Site.name).order_by(Site.name)
    if user_role != 'super_admin':
        if not user_site_names:
            return LabelListResponse(labels=[])
        statement = statement.where(Site.name.in_(user_site_names))
    result = await db.execute(statement)
    return LabelListResponse(
        labels=list(result.scalars().all()),
    )


def _visible_stream_names_query(label: str) -> Select[tuple[str]]:
    """Build the query for streams enabled for live playback.

    Args:
        label: Site label whose configured streams are selected.

    Returns:
        SQLAlchemy select statement yielding configured stream names.
    """
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
    """Resolve and validate an enabled configured stream name.

    Args:
        db: Database session used to query configured streams.
        label: Site label containing the requested stream.
        stream_id: Optional encoded configured stream identifier.
        key: Optional decoded configured stream name.

    Returns:
        Validated decoded configured stream name.

    Raises:
        HTTPException: If the stream identifier is absent or not enabled.
    """
    requested_name = key or (
        decode_media_segment(stream_id) if stream_id else ''
    )
    if not requested_name:
        raise HTTPException(
            status_code=422,
            detail='stream_id_or_key_required',
        )

    result = await db.scalar(
        select(StreamConfigModel.stream_name)
        .join(Site)
        .where(
            Site.name == label,
            StreamConfigModel.stream_name == requested_name,
            StreamConfigModel.recognition_enabled.is_(True),
        )
        .limit(1),
    )
    if result is None:
        raise HTTPException(status_code=404, detail='stream_not_found')
    return str(result)


async def resolve_configured_stream_names(
    db: AsyncSession,
    requests: list[tuple[str, str | None, str | None]],
) -> list[str]:
    """Resolve many requested streams with one indexed database query.

    Args:
        db: Database session used to validate configured streams.
        requests: ``(site_label, stream_id, stream_name)`` request triples.

    Returns:
        Canonical stream names in the same order as ``requests``.

    Raises:
        HTTPException: If a requested stream is missing or disabled.
    """
    requested_pairs: list[tuple[str, str]] = []
    for label, stream_id, key in requests:
        stream_name = key or (
            decode_media_segment(stream_id) if stream_id else ''
        )
        if not stream_name:
            raise HTTPException(
                status_code=422,
                detail='stream_id_or_key_required',
            )
        requested_pairs.append((label, stream_name))

    if not requested_pairs:
        return []
    available_rows = await db.execute(
        select(Site.name, StreamConfigModel.stream_name)
        .join(Site)
        .where(
            StreamConfigModel.recognition_enabled.is_(True),
            tuple_(Site.name, StreamConfigModel.stream_name).in_(
                set(requested_pairs),
            ),
        ),
    )
    available = {
        (str(label), str(stream_name))
        for label, stream_name in available_rows.all()
    }
    for pair in requested_pairs:
        if pair not in available:
            raise HTTPException(status_code=404, detail='stream_not_found')
    return [stream_name for _label, stream_name in requested_pairs]
