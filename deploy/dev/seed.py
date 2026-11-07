from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import UUID

from deploy.dev.run import CONFIG
from deploy.dev.run import load_environment
from deploy.dev.run import STATE

TENANT = UUID('d8c43372-0609-4afb-b903-39cae0f2e4eb')
DEPLOYMENT = UUID('b3094d94-1b94-405e-bfa1-4aad6e58fbaf')


async def seed():
    load_environment()
    from deploy.dev.schema import prepare_schema
    from sqlalchemy import select
    from examples.auth.database import engine, AsyncSessionLocal
    from examples.auth.models import (
        Tenant,
        Deployment,
        Group,
        User,
        UserProfile,
        UserIdentity,
        Feature,
        LegalDocument,
    )
    from examples.auth.models import group_features_table

    await prepare_schema(engine)
    operator = json.loads((CONFIG / 'operator.json').read_text())
    reference = json.loads(
        (CONFIG / 'authorization-baseline.json').read_text(),
    )
    async with AsyncSessionLocal() as db:
        if await db.get(Tenant, TENANT) is None:
            db.add(Tenant(id=TENANT, name='Visionnaire DEV', status='active'))
            await db.flush()
        if await db.get(Deployment, DEPLOYMENT) is None:
            db.add(
                Deployment(
                    id=DEPLOYMENT,
                    tenant_id=TENANT,
                    api_base_url=(
                        'https://dev.changdar-server.mooo.com/hazard/api'
                    ),
                    config_revision=1,
                    status='active',
                ),
            )
        group = await db.scalar(
            select(Group).where(Group.uniform_number == '00000000'),
        )
        if group is None:
            group = Group(
                name='DEV 測試群組',
                uniform_number='00000000',
                max_allowed_streams=reference['operator'][
                    'max_allowed_streams'
                ],
            )
            db.add(group)
            await db.flush()
        user = await db.scalar(
            select(User).where(User.username == operator['username']),
        )
        if user is None:
            user = User(
                username=operator['username'],
                role='admin',
                status='active',
                tenant_id=TENANT,
                group_id=group.id,
            )
            db.add(user)
            await db.flush()
            db.add(
                UserProfile(
                    user_id=user.id,
                    family_name='DEV',
                    given_name='ChangDar',
                    email=operator['email'],
                ),
            )
            db.add(
                UserIdentity(
                    user_id=user.id,
                    provider='keycloak',
                    provider_user_id=operator['id'],
                    email=operator['email'],
                    email_verified=True,
                ),
            )
        for entry in reference['catalog']:
            name = entry['feature_name']
            feature = await db.scalar(
                select(Feature).where(Feature.feature_name == name),
            )
            if feature is None:
                feature = Feature(
                    feature_name=name,
                    description=entry['description'],
                )
                db.add(feature)
                await db.flush()
            granted = await db.scalar(
                select(group_features_table.c.feature_id).where(
                    group_features_table.c.group_id == group.id,
                    group_features_table.c.feature_id == feature.id,
                ),
            )
            if name in reference['operator']['features'] and granted is None:
                await db.execute(
                    group_features_table.insert().values(
                        group_id=group.id,
                        feature_id=feature.id,
                    ),
                )
        from datetime import datetime, timezone

        for kind in ('terms', 'privacy', 'ai_terms'):
            if (
                await db.scalar(
                    select(LegalDocument.id).where(LegalDocument.type == kind),
                )
                is None
            ):
                db.add(
                    LegalDocument(
                        type=kind,
                        version='dev-1',
                        locale='zh-TW',
                        title='DEV 測試文件',
                        content='此為獨立開發環境，請使用測試資料。',
                        effective_at=datetime.now(timezone.utc),
                        is_active=True,
                    ),
                )
        catalog_path = CONFIG / 'detection-catalog.json'
        if catalog_path.exists():
            from examples.auth.models import DetectionModelCatalog

            for row in json.loads(catalog_path.read_text()):
                definition = row['definition']
                if definition['tenant_ids'] != [str(TENANT)] or definition[
                    'capabilities'
                ] != ['image']:
                    raise RuntimeError(
                        'Refusing non-dev detection model grants.',
                    )
                if (
                    not Path(definition['artifact'])
                    .resolve()
                    .is_relative_to(
                        (STATE / 'models/pt').resolve(),
                    )
                ):
                    raise RuntimeError(
                        'Refusing shared detection artifact paths.',
                    )
                if (
                    await db.get(DetectionModelCatalog, row['model_key'])
                    is None
                ):
                    db.add(DetectionModelCatalog(**row))
        await db.commit()
    await engine.dispose()
    print(
        'Deployed schema baseline and dev operator grants verified; '
        'production business data is not copied.',
    )


if __name__ == '__main__':
    asyncio.run(seed())
