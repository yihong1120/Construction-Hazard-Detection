from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Response
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.database import get_db
from examples.auth.models import User
from examples.db_management.deps import get_current_user
from examples.db_management.deps import require_admin
from examples.db_management.schemas.resource_options import ResourceOptions
from examples.db_management.schemas.stream_config import SiteStreamConfigUpsert
from examples.db_management.schemas.stream_config import StreamConfigCreate
from examples.db_management.schemas.stream_config import StreamConfigRead
from examples.db_management.schemas.stream_config import StreamConfigUpdate
from examples.db_management.services import (
    stream_management_services as service,
)

router = APIRouter(tags=['stream-config'])


@router.get('/sites/{site_id}/stream-models', response_model=ResourceOptions)
async def get_stream_models(
    site_id: int,
    response: Response,
    me: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ResourceOptions:
    options = await service.get_stream_model_options(site_id, me, db)
    response.headers['Cache-Control'] = 'private, no-store'
    return options


@router.get(
    '/sites/{site_id}/stream-config',
    response_model=list[StreamConfigRead],
)
async def endpoint_get_site_stream_config(
    site_id: int,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> list[StreamConfigRead]:
    """List stream settings for a site without accepting a group selection.

    Args:
        site_id: Identifier of the site whose streams are requested.
        db: Database session used to load stream configurations.
        me: Authenticated user whose site scope is enforced.

    Returns:
        Stream configurations visible to the authenticated user.

    Raises:
        HTTPException: If the site does not exist or is outside the user's
            management scope.
    """
    return await service.list_site_stream_configs(site_id, db, me)


@router.post(
    '/create_stream_config',
    dependencies=[Depends(require_admin)],
)
async def endpoint_create_stream_config(
    payload: StreamConfigCreate,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> dict[str, str | int]:
    """Create a new stream configuration for a site.

    Args:
        payload (StreamConfigCreate): The stream configuration data.
        db (AsyncSession): The database session.
        me (User): The currently authenticated user.

    Returns:
        Dict[str, str | int]: Confirmation message and the created config ID.

    Raises:
        HTTPException: If the site does not exist
            or group stream limit is exceeded.
    """
    return await service.create_stream_config_for_operator(payload, db, me)


@router.put(
    '/sites/{site_id}/stream-config',
    response_model=list[StreamConfigRead],
    dependencies=[Depends(require_admin)],
)
async def endpoint_put_site_stream_config(
    site_id: int,
    payload: SiteStreamConfigUpsert,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> list[StreamConfigRead]:
    """Replace or create stream settings for a site-owned group.

    Args:
        site_id: Identifier of the site whose streams are being updated.
        payload: Desired site-scoped stream configuration items.
        db: Database session used to persist configurations.
        me: Authenticated administrator whose scope determines the group.

    Returns:
        Current stream configurations visible to the administrator.

    Raises:
        HTTPException: If the site or an existing configuration is unavailable,
            a name is duplicated, or the group's stream limit is reached.
    """
    return await service.upsert_site_stream_configs(site_id, payload, db, me)


@router.put(
    '/stream_config/update/{cfg_id}',
    dependencies=[Depends(require_admin)],
)
async def endpoint_update_stream_config(
    cfg_id: int,
    payload: StreamConfigUpdate,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> dict[str, str]:
    """Update an existing stream configuration.

    Args:
        cfg_id (int): The stream configuration identifier.
        payload (StreamConfigUpdate): Updated stream configuration details.
        db (AsyncSession): The database session.
        me (User): The currently authenticated user.

    Returns:
        Dict[str, str]: Confirmation message of successful update.

    Raises:
        HTTPException: If config does not exist, permission denied,
            or name conflict.
    """
    return await service.update_stream_config_for_operator(
        cfg_id, payload, db, me,
    )


@router.delete(
    '/delete_stream_config/{cfg_id}',
    dependencies=[Depends(require_admin)],
)
async def endpoint_delete_stream_config(
    cfg_id: int,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> dict[str, str]:
    """Delete an existing stream configuration.

    Args:
        cfg_id (int): The identifier of the stream configuration.
        db (AsyncSession): The database session.
        me (User): The currently authenticated user.

    Returns:
        Dict[str, str]: Confirmation message.

    Raises:
        HTTPException: If configuration not found or permission denied.
    """
    return await service.delete_stream_config_for_operator(cfg_id, db, me)


@router.get(
    '/group_stream_limit',
    dependencies=[Depends(get_current_user)],
)
async def endpoint_group_stream_limit(
    group_id: int,
    db: AsyncSession = Depends(get_db),
    me: User = Depends(get_current_user),
) -> dict[str, int]:
    """Retrieve the stream limit and current usage for a group.

    Args:
        group_id (int): The identifier of the group.
        db (AsyncSession): The database session.
        me (User): The currently authenticated user.

    Returns:
        Dict[str, int]: Details of stream usage and limits.

    Raises:
        HTTPException: If the user lacks permission to view the group's limits.
    """
    # Super admin has unlimited access, admins restricted to their group
    return await service.get_group_stream_usage(group_id, db, me)
