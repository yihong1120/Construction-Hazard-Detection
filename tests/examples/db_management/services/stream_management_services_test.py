from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError

from examples.auth.models import User
from examples.db_management.schemas.stream_config import SiteStreamConfigItem
from examples.db_management.schemas.stream_config import SiteStreamConfigUpsert
from examples.db_management.services import (
    stream_management_services as service,
)


class TestStreamManagementTransactions(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.admin = User(username='manager', role='admin', group_id=1)
        self.site = SimpleNamespace(id=1, groups=[SimpleNamespace(id=1)])
        self.db = AsyncMock()
        self.db.get.return_value = self.site
        self.db.add = MagicMock()
        self.payload = SiteStreamConfigUpsert(
            streams=[
                SiteStreamConfigItem(
                    stream_name='New',
                    video_url='rtsp://new',
                ),
            ],
        )
        self.models = AsyncMock()
        self.list_configs = AsyncMock(return_value=[])
        self.quota = AsyncMock(return_value=(0, 5))
        for name, replacement in (
            ('require_site', AsyncMock()),
            ('require_model_async', self.models),
            ('list_stream_configs', self.list_configs),
            ('require_stream_capacity', self.quota),
        ):
            patched = patch.object(service, name, replacement)
            patched.start()
            self.addCleanup(patched.stop)

    async def test_failed_flush_or_commit_rolls_back_batch(self) -> None:
        for phase in ('flush', 'commit'):
            for error, status in (
                (IntegrityError('insert', {}, Exception('conflict')), 409),
                (RuntimeError('database unavailable'), 500),
            ):
                with self.subTest(phase=phase, status=status):
                    self.db.flush.reset_mock(side_effect=True)
                    self.db.commit.reset_mock(side_effect=True)
                    self.db.rollback.reset_mock()
                    getattr(self.db, phase).side_effect = error
                    with self.assertRaises(HTTPException) as raised:
                        await service.upsert_site_stream_configs(
                            1,
                            self.payload,
                            self.db,
                            self.admin,
                        )
                    self.assertEqual(raised.exception.status_code, status)
                    self.db.rollback.assert_awaited_once()
                    if phase == 'flush':
                        self.db.commit.assert_not_awaited()
                    else:
                        self.db.commit.assert_awaited_once()

    async def test_model_denial_stops_before_any_batch_mutation(self) -> None:
        self.models.side_effect = HTTPException(403, 'Model access denied')
        with self.assertRaises(HTTPException) as raised:
            await service.upsert_site_stream_configs(
                1,
                self.payload,
                self.db,
                self.admin,
            )
        self.assertEqual(raised.exception.status_code, 403)
        self.list_configs.assert_not_awaited()
        self.db.add.assert_not_called()
        self.db.flush.assert_not_awaited()
        self.db.commit.assert_not_awaited()

    async def test_invalid_later_item_does_not_create_earlier_item(self):
        self.payload.streams.append(
            SiteStreamConfigItem(
                id=999,
                stream_name='Missing',
                video_url='rtsp://missing',
            ),
        )
        with self.assertRaises(HTTPException) as raised:
            await service.upsert_site_stream_configs(
                1,
                self.payload,
                self.db,
                self.admin,
            )
        self.assertEqual(raised.exception.status_code, 404)
        self.quota.assert_not_awaited()
        self.db.add.assert_not_called()
        self.db.flush.assert_not_awaited()
        self.db.commit.assert_not_awaited()
