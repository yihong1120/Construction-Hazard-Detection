"""Add missing catalog colors without changing historical record snapshots.

Preview: python -m scripts.publish_class_colors
Apply:   python -m scripts.publish_class_colors --apply
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy

from sqlalchemy import select

from examples.auth.database import AsyncSessionLocal
from examples.auth.database import engine
from examples.auth.models import DetectionModelCatalog
from examples.shared.class_colors import normalize_color
from examples.shared.class_colors import published_color
from examples.YOLO_server_api.model_registry import ModelEntry


async def publish(apply=False):
    try:
        async with AsyncSessionLocal() as db:
            rows = (
                await db.scalars(
                    select(DetectionModelCatalog).with_for_update(),
                )
            ).all()
            changed = []
            for row in rows:
                definition = deepcopy(row.definition)
                for item in definition.get('classes', []):
                    if not normalize_color(item.get('color')):
                        item['color'] = published_color(item['code'])
                ModelEntry.model_validate({**definition, 'id': row.model_key})
                if definition != row.definition:
                    row.definition = definition
                    changed.append(row.model_key)
            if apply:
                await db.commit()
            else:
                await db.rollback()
            print({'applied': apply, 'updated_models': changed})
    finally:
        await engine.dispose()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    asyncio.run(publish(parser.parse_args().apply))
