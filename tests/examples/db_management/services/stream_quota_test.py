from __future__ import annotations

import asyncio
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import HTTPException

from examples.auth.models import Group
from examples.auth.models import Site
from examples.db_management.schemas.stream_config import SiteStreamConfigItem
from examples.db_management.schemas.stream_config import SiteStreamConfigUpsert
from examples.db_management.schemas.stream_config import StreamConfigCreate
from examples.db_management.services import (
    stream_management_services as service,
)
from tests.database_fixtures_test import OrmDatabase


def _return_none(*a, **kw):
    return None


async def _single_and_batch_share_the_batch(testcase, /):
    async with testcase.sessions() as db:
        return await service.upsert_site_stream_configs(
            testcase.site_id,
            SiteStreamConfigUpsert(
                streams=[
                    SiteStreamConfigItem(
                        stream_name='batch',
                        video_url='rtsp://batch',
                    ),
                ],
            ),
            db,
            testcase.me,
        )


class StreamQuotaTest(OrmDatabase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        async with self.sessions() as db:
            group = Group(
                name='quota',
                uniform_number='12345678',
                max_allowed_streams=1,
            )
            site = Site(name='quota')
            db.add_all([group, site])
            await db.commit()
            self.group_id, self.site_id = group.id, site.id
        self.me = SimpleNamespace(
            role='admin',
            username='manager',
            group_id=self.group_id,
            tenant_id='tenant',
        )
        site = SimpleNamespace(
            id=self.site_id,
            groups=[SimpleNamespace(id=self.group_id)],
        )
        for name, value in [
            ('_get_site_or_404', AsyncMock(return_value=site)),
            ('_site_permission', _return_none),
            ('require_site', AsyncMock()),
            ('require_model_async', AsyncMock()),
            ('_list_site_stream_config_reads', AsyncMock(return_value=[])),
        ]:
            operation = patch.object(service, name, value)
            operation.start()
            self.addCleanup(operation.stop)

    async def single(self, name):
        async with self.sessions() as db:
            return await service.create_stream_config_for_operator(
                StreamConfigCreate(
                    site_id=self.site_id,
                    stream_name=name,
                    video_url='rtsp://' + name,
                ),
                db,
                self.me,
            )

    async def test_two_single_creates_cannot_exceed_one_remaining_slot(self):
        result = await asyncio.gather(
            self.single('first'),
            self.single('second'),
            return_exceptions=True,
        )
        self.assertEqual(sum(isinstance(value, dict) for value in result), 1)
        errors = [
            value for value in result if isinstance(value, HTTPException)
        ]
        self.assertEqual([error.status_code for error in errors], [403])
        self.assertEqual(
            await self.db.fetchval('SELECT count(*) FROM stream_configs'),
            1,
        )

    async def test_single_and_batch_share_the_same_quota_lock(self):
        batch = partial(_single_and_batch_share_the_batch, self)

        result = await asyncio.gather(
            self.single('single'),
            batch(),
            return_exceptions=True,
        )
        self.assertEqual(
            sum(isinstance(value, HTTPException) for value in result),
            1,
        )
        self.assertEqual(
            await self.db.fetchval('SELECT count(*) FROM stream_configs'),
            1,
        )

    async def test_failed_flush_rolls_back_and_releases_capacity(self):
        async with self.sessions() as db:
            with patch.object(
                db,
                'flush',
                AsyncMock(side_effect=RuntimeError('failed insert')),
            ):
                with self.assertRaises(HTTPException):
                    await service.create_stream_config_for_operator(
                        StreamConfigCreate(
                            site_id=self.site_id,
                            stream_name='failed',
                            video_url='rtsp://failed',
                        ),
                        db,
                        self.me,
                    )
        self.assertEqual(
            await self.db.fetchval('SELECT count(*) FROM stream_configs'),
            0,
        )
        self.assertIsInstance(await self.single('retry'), dict)
