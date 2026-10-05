from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy import select

from examples.auth.models import Site
from examples.auth.models import user_sites_table


async def require_site(db, user, site_id: int):
    site = await db.get(Site, site_id)
    if site is None:
        raise HTTPException(404, detail='Site not found')
    # Explicit membership remains required, including for platform operators.
    member = await db.scalar(
        select(user_sites_table.c.user_id).where(
            user_sites_table.c.site_id == site_id,
            user_sites_table.c.user_id == user.id,
        ),
    )
    if member is None:
        raise HTTPException(403, detail='No access to site')
    return site
