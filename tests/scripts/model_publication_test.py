from __future__ import annotations

import asyncio
import json
import os
from functools import partial
from functools import partialmethod
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from redis.asyncio import Redis
from sqlalchemy import text

from examples.auth.models import DetectionModelCatalog
from examples.auth.models import Group
from examples.auth.models import Site
from examples.auth.models import StreamConfig
from examples.YOLO_server_api.model_registry import ModelEntry
from examples.YOLO_server_api.model_registry import revision_for
from scripts import model_publication as module
from tests.database_fixtures_test import OrmDatabase


async def _wait_for_published_revision(
    instance,
    revisions,
    timeout,
    *,
    _scope,
):
    if revisions['detector'] == revision_for(
        _scope.testcase.new.model_dump(exclude={'id'}),
    ):
        _scope.ready.set()
        await asyncio.Event().wait()
    else:
        await _scope.verify(instance, revisions, timeout)


async def _weight_rollback_preserves_an_unrelated_change_default(
    apply,
    _capture_testcase,
    /,
    prepared,
    default,
    **kwargs,
):
    await apply(prepared, default, **kwargs)
    async with _capture_testcase.sessions() as db:
        own = await db.get(DetectionModelCatalog, 'detector')
        own.is_default = False
        await db.flush()
        other = await db.get(DetectionModelCatalog, 'other')
        other.is_default = True
        await db.commit()


async def _preflight_cannot_overwrite_concurrent_catalog_prepare(
    _capture_testcase,
    /,
    args,
):
    async with _capture_testcase.sessions() as db:
        row = await db.get(DetectionModelCatalog, 'detector')
        row.definition = {
            **row.definition,
            'display_name': 'newer edit',
        }
        await db.commit()
    return [(_capture_testcase.new, 0, True)]


async def _asyncSetUp_acknowledge_restored(
    instance,
    *,
    restore,
    _capture_testcase,
):
    await restore(instance)
    await _capture_testcase.acknowledge(_capture_testcase.old)


async def _publication_reads_back_actual_loaded_report_after_commit(
    apply,
    _capture_testcase,
    /,
    prepared,
    default,
    **kwargs,
):
    await apply(prepared, default, **kwargs)
    await _capture_testcase.acknowledge(_capture_testcase.new)


async def _serialization_lock_is_released_after_prepare(
    ready,
    /,
    args,
):
    ready.set()
    await asyncio.Event().wait()


async def _concurrent_catalog_change_is_not_concurrent_change(
    apply,
    _capture_testcase,
    /,
    prepared,
    default,
    **kwargs,
):
    await apply(prepared, default, **kwargs)
    async with _capture_testcase.sessions() as db:
        row = await db.get(DetectionModelCatalog, 'detector')
        row.definition = {
            **row.definition,
            'display_name': 'intervening edit',
        }
        await db.commit()


async def _applied_recovery_verifies_loaded_version_acknowledge_new(
    instance,
    *args,
    _capture_testcase,
    verify,
):
    await _capture_testcase.acknowledge(_capture_testcase.new)
    await verify(instance, *args)


class _CancellationAfterCommitCompletesVerifiedRestorationCallbacks:
    ready: Any
    testcase: ModelPublicationTest
    verify: Any


class ModelPublicationTest(OrmDatabase):
    async def test_weight_rollback_preserves_an_unrelated_default_edit(self):
        async with self.sessions() as db:
            db.add(
                DetectionModelCatalog(
                    model_key='other',
                    definition=self.old.model_dump(exclude={'id'}),
                    display_order=1,
                    is_default=False,
                ),
            )
            await db.commit()
        apply = module.catalog.apply_models

        change_default = partial(
            _weight_rollback_preserves_an_unrelated_change_default,
            apply,
            self,
        )

        with patch.object(module.catalog, 'apply_models', change_default):
            with self.assertRaisesRegex(RuntimeError, 'restored and verified'):
                await module.ModelPublication().publish(self.args)
        async with self.sessions() as db:
            own = await db.get(DetectionModelCatalog, 'detector')
            other = await db.get(DetectionModelCatalog, 'other')
            self.assertEqual(own.definition['version'], 'v1')
            self.assertFalse(own.is_default)
            self.assertTrue(other.is_default)

    async def test_preflight_cannot_overwrite_concurrent_catalog_edit(self):
        prepare = partial(
            _preflight_cannot_overwrite_concurrent_catalog_prepare,
            self,
        )

        with patch.object(module.catalog, 'prepare_models', prepare):
            with self.assertRaises(module.catalog.PublicationConflict):
                await module.ModelPublication().publish(self.args)
        self.assertEqual((await self.current())['display_name'], 'newer edit')
        self.assertEqual((await self.current())['version'], 'v1')

    @pytest.mark.ml
    async def test_supervisor_reads_catalog_revision_from_real_database(self):
        import asyncpg
        from main import MainApp

        app = MainApp.__new__(MainApp)
        app.db_pool = await asyncpg.create_pool(
            self.url,
            min_size=1,
            max_size=1,
        )
        try:
            configs = await app.fetch_stream_configs()
        finally:
            await app.db_pool.close()
        self.assertEqual(configs[0]['stream_id'], self.stream_id)
        self.assertEqual(
            configs[0]['model_revision'],
            revision_for(
                self.old.model_dump(exclude={'id'}),
            ),
        )

    async def test_cached_ack_cannot_verify_an_identical_republication(self):
        await self.acknowledge(self.old)
        with patch.object(
            module.catalog,
            'prepare_models',
            new=AsyncMock(return_value=[(self.old, 0, True)]),
        ):
            with self.assertRaisesRegex(RuntimeError, 'restored and verified'):
                await module.ModelPublication().publish(self.args)

    async def test_precision_alias_cannot_bypass_int8_manifest_preflight(self):
        with patch.object(
            module,
            'WorkerSettings',
            return_value=SimpleNamespace(
                precision='ENGINE',
                artifact_manifest=None,
            ),
        ):
            with self.assertRaisesRegex(ValueError, 'engine manifest'):
                await module.ModelPublication().publish(self.args)
        self.assertEqual((await self.current())['version'], 'v1')

    async def test_failed_publication_restores_previous_default(self):
        async with self.sessions() as db:
            row = await db.get(DetectionModelCatalog, 'detector')
            row.is_default = False
            await db.flush()
            db.add(
                DetectionModelCatalog(
                    model_key='other',
                    definition=self.old.model_dump(exclude={'id'}),
                    display_order=1,
                    is_default=True,
                ),
            )
            await db.commit()
        self.args.default = 'detector'
        with patch.object(
            module.catalog,
            'prepare_models',
            new=AsyncMock(return_value=[(self.new, 0, False)]),
        ):
            with self.assertRaisesRegex(RuntimeError, 'restored and verified'):
                await module.ModelPublication().publish(self.args)
        async with self.sessions() as db:
            own = await db.get(DetectionModelCatalog, 'detector')
            other = await db.get(DetectionModelCatalog, 'other')
            self.assertFalse(own.is_default)
            self.assertTrue(other.is_default)

    async def test_restore_without_loaded_ack_is_reported_as_incomplete(self):
        with patch.object(
            module.ModelPublication,
            '_verify',
            new=AsyncMock(side_effect=TimeoutError()),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                'restoration needs attention',
            ):
                await module.ModelPublication().publish(self.args)
        self.assertEqual((await self.current())['version'], 'v1')

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.redis = Redis.from_url(os.environ['NOTIFICATION_REDIS_TEST_URL'])
        self.old = ModelEntry(
            id='detector',
            display_name='Detector',
            version='v1',
            artifact='weights.pt',
            sha256='a' * 64,
            capabilities=['stream'],
            tenant_ids=['tenant'],
            site_ids=[1],
            classes=[{'id': 0, 'code': 'helmet', 'display_name': 'Helmet'}],
        )
        self.new = self.old.model_copy(
            update={'version': 'v2', 'sha256': 'b' * 64},
        )
        async with self.sessions() as db:
            group = Group(name='publication', uniform_number='12345678')
            site = Site(name='publication')
            db.add_all([group, site])
            await db.flush()
            stream = StreamConfig(
                group_id=group.id,
                site_id=site.id,
                stream_name='camera',
                video_url='rtsp://test-camera',
                model_key='detector',
                recognition_enabled=True,
                work_start_hour=0,
                work_end_hour=24,
            )
            db.add_all(
                [
                    stream,
                    DetectionModelCatalog(
                        model_key='detector',
                        definition=self.old.model_dump(
                            exclude={'id'},
                        ),
                        display_order=0,
                        is_default=True,
                    ),
                ],
            )
            await db.commit()
            self.stream_id = stream.id
        self.key = f'test:model-status:{self.database}:{self.stream_id}'
        self.challenge_key = f'test:model-challenge:{self.database}:detector'
        await self.redis.delete(self.key)
        self.args = SimpleNamespace(
            default=None,
            catalog_only=False,
            verify_timeout=0.1,
        )
        preflight = patch.object(
            module,
            'verify_artifact',
            return_value='weights.pt',
        )
        self.preflight = preflight.start()
        self.addCleanup(preflight.stop)
        restore = module.ModelPublication._restore_catalog

        acknowledge_restored = partialmethod(
            _asyncSetUp_acknowledge_restored,
            restore=restore,
            _capture_testcase=self,
        )
        for operation in (
            patch.object(module, 'engine', self.engine),
            patch.object(module, 'model_status_key', return_value=self.key),
            patch.object(
                module,
                'model_challenge_key',
                return_value=self.challenge_key,
            ),
            patch.object(module, 'AsyncSessionLocal', self.sessions),
            patch.object(module.catalog, 'AsyncSessionLocal', self.sessions),
            patch.object(
                module.catalog,
                'prepare_models',
                new=AsyncMock(return_value=[(self.new, 0, True)]),
            ),
            patch.object(
                module,
                'RedisManager',
                return_value=SimpleNamespace(redis=self.redis),
            ),
            patch.object(
                module,
                'WorkerSettings',
                return_value=SimpleNamespace(precision=None),
            ),
            patch.object(
                module.ModelPublication,
                '_restore_catalog',
                acknowledge_restored,
            ),
        ):
            operation.start()
            self.addCleanup(operation.stop)

    async def asyncTearDown(self):
        # ModelPublication owns and closes its dedicated Redis client; create a
        # temporary cleanup connection for this test's one acknowledgement key.
        cleanup = Redis.from_url(os.environ['NOTIFICATION_REDIS_TEST_URL'])
        await cleanup.delete(self.key, self.challenge_key)
        await cleanup.aclose()
        await self.redis.aclose()
        await super().asyncTearDown()

    async def acknowledge(self, entry):
        challenge = await self.redis.get(self.challenge_key)
        await self.redis.set(
            self.key,
            json.dumps(
                {
                    'model_id': entry.id,
                    'model_version': entry.version,
                    'model_sha256': entry.sha256,
                    'catalog_revision': revision_for(
                        entry.model_dump(exclude={'id'}),
                    ),
                    'publication': challenge.decode() if challenge else None,
                },
            ),
            ex=45,
        )

    async def current(self):
        async with self.sessions() as db:
            row = await db.get(DetectionModelCatalog, 'detector')
            return row.definition

    async def test_publication_reads_back_actual_loaded_revision(self):
        apply = module.catalog.apply_models

        report_after_commit = partial(
            _publication_reads_back_actual_loaded_report_after_commit,
            apply,
            self,
        )

        with patch.object(module.catalog, 'apply_models', report_after_commit):
            await module.ModelPublication().publish(self.args)
        self.assertEqual((await self.current())['version'], 'v2')
        self.assertEqual(self.preflight.call_count, 2)

    async def test_timeout_restores_catalog_and_verifies_previous_worker(self):
        await self.acknowledge(self.old)
        with self.assertRaisesRegex(RuntimeError, 'restored and verified'):
            await module.ModelPublication().publish(self.args)
        self.assertEqual(
            await self.current(),
            self.old.model_dump(exclude={'id'}),
        )
        async with self.sessions() as db:
            row = await db.get(DetectionModelCatalog, 'detector')
            self.assertTrue(row.is_default)

    async def test_artifact_failure_happens_before_catalog_mutation(self):
        self.preflight.side_effect = ValueError('checksum mismatch')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            await module.ModelPublication().publish(self.args)
        self.assertEqual((await self.current())['version'], 'v1')

    async def test_catalog_only_cannot_bypass_running_workers(self):
        self.args.catalog_only = True
        with self.assertRaisesRegex(ValueError, 'cannot bypass'):
            await module.ModelPublication().publish(self.args)
        self.assertEqual((await self.current())['version'], 'v1')

    async def test_serialization_lock_is_released_after_cancellation(self):
        ready = asyncio.Event()

        prepare = partial(_serialization_lock_is_released_after_prepare, ready)

        with patch.object(module.catalog, 'prepare_models', prepare):
            task = asyncio.create_task(
                module.ModelPublication().publish(self.args),
            )
            await asyncio.wait_for(ready.wait(), 5)
            with self.assertRaisesRegex(
                RuntimeError,
                'Another model publication',
            ):
                await module.ModelPublication().publish(self.args)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        async with self.engine.begin() as connection:
            self.assertTrue(
                await connection.scalar(
                    text(
                        'SELECT pg_try_advisory_xact_lock(675839201)',
                    ),
                ),
            )

    async def test_concurrent_catalog_change_is_not_overwritten_by_rollback(
        self,
    ):
        apply = module.catalog.apply_models

        concurrent_change = partial(
            _concurrent_catalog_change_is_not_concurrent_change,
            apply,
            self,
        )

        with patch.object(module.catalog, 'apply_models', concurrent_change):
            with self.assertRaisesRegex(
                RuntimeError,
                'restoration needs attention',
            ):
                await module.ModelPublication().publish(self.args)
        self.assertEqual(
            (await self.current())['display_name'],
            'intervening edit',
        )

    async def test_cancellation_after_commit_completes_verified_restoration(
        self,
    ):
        _callbacks = (
            _CancellationAfterCommitCompletesVerifiedRestorationCallbacks()
        )
        _callbacks.testcase = self
        await self.acknowledge(self.old)
        _callbacks.ready = asyncio.Event()
        _callbacks.verify = module.ModelPublication._verify

        wait_for_new = partialmethod(
            _wait_for_published_revision,
            _scope=_callbacks,
        )

        with patch.object(module.ModelPublication, '_verify', wait_for_new):
            task = asyncio.create_task(
                module.ModelPublication().publish(self.args),
            )
            await asyncio.wait_for(_callbacks.ready.wait(), 5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual((await self.current())['version'], 'v1')

    async def test_failure_before_commit_preserves_previous_publication(self):
        await self.acknowledge(self.old)
        with patch.object(
            module.catalog,
            'apply_models',
            new=AsyncMock(side_effect=RuntimeError('write failed')),
        ):
            with self.assertRaisesRegex(RuntimeError, 'restored and verified'):
                await module.ModelPublication().publish(self.args)
        self.assertEqual((await self.current())['version'], 'v1')

    async def interrupted_publication(self):
        publication = module.ModelPublication()
        publication.previous = {
            'detector': {
                'definition': self.old.model_dump(exclude={'id'}),
                'display_order': 0,
                'is_default': True,
            },
        }
        publication.prepared = [(self.new, 0, True)]
        await publication._record({self.stream_id: 'detector'}, 0.1)
        return publication

    async def test_killed_publisher_is_restored_from_durable_journal(self):
        import signal
        import subprocess
        import sys
        from examples.auth.models import ModelPublicationRecord

        publication = await self.interrupted_publication()
        # A separate interpreter commits using the real publication code and
        # then dies without running Python cleanup or compensation.
        child = '''
import asyncio, os, signal
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.engine import make_url
from examples.auth.models import ModelPublicationRecord
from examples.YOLO_server_api.model_registry import ModelEntry
from scripts import sync_huggingface_models as catalog
async def run():
    url = os.environ['CHD_TEST_PUBLICATION_URL']
    assert make_url(url).database.startswith('chd_test_')
    engine = create_async_engine(url)
    catalog.AsyncSessionLocal = async_sessionmaker(
        engine, expire_on_commit=False,
    )
    async with catalog.AsyncSessionLocal() as db:
        record = await db.get(
            ModelPublicationRecord, os.environ['CHD_TEST_PUBLICATION_ID'],
        )
        snapshot = record.snapshot
    prepared = [(ModelEntry.model_validate(entry), order, default)
                for entry, order, default in snapshot['prepared']]
    await catalog.apply_models(prepared, snapshot['selected_default'],
        expected=snapshot['previous'],
        publication_id=os.environ['CHD_TEST_PUBLICATION_ID'],
    )
    os.kill(os.getpid(), signal.SIGKILL)
asyncio.run(run())
'''
        environment = {
            **os.environ,
            'CHD_TEST_PUBLICATION_URL': self.url.replace(
                'postgresql://',
                'postgresql+asyncpg://',
                1,
            ),
            'CHD_TEST_PUBLICATION_ID': publication.publication_id,
        }
        process = subprocess.Popen(
            [sys.executable, '-c', child],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        _, error = await asyncio.to_thread(process.communicate, timeout=15)
        self.assertEqual(process.returncode, -signal.SIGKILL, error.decode())
        self.assertEqual((await self.current())['version'], 'v2')
        async with self.sessions() as db:
            record = await db.get(
                ModelPublicationRecord,
                publication.publication_id,
            )
            self.assertEqual(record.phase, 'applied')
        with self.assertRaisesRegex(RuntimeError, publication.publication_id):
            await module.ModelPublication().publish(self.args)
        recover = SimpleNamespace(
            recover=publication.publication_id,
            restore=True,
        )
        await module.ModelPublication().publish(recover)
        self.assertEqual((await self.current())['version'], 'v1')
        async with self.sessions() as db:
            record = await db.get(
                ModelPublicationRecord,
                publication.publication_id,
            )
            self.assertEqual(record.phase, 'restored')
        await module.ModelPublication().publish(recover)

    async def test_prepared_recovery_preserves_old_model(self):
        from examples.auth.models import ModelPublicationRecord

        publication = await self.interrupted_publication()
        await module.ModelPublication().publish(
            SimpleNamespace(recover=publication.publication_id, restore=False),
        )
        self.assertEqual((await self.current())['version'], 'v1')
        async with self.sessions() as db:
            self.assertEqual(
                (
                    await db.get(
                        ModelPublicationRecord,
                        publication.publication_id,
                    )
                ).phase,
                'restored',
            )

    async def test_applied_recovery_verifies_loaded_version(self):
        from examples.auth.models import ModelPublicationRecord

        publication = await self.interrupted_publication()
        await module.catalog.apply_models(
            publication.prepared,
            publication_id=publication.publication_id,
            expected=publication.previous,
        )
        verify = module.ModelPublication._verify

        acknowledge_new = partialmethod(
            _applied_recovery_verifies_loaded_version_acknowledge_new,
            _capture_testcase=self,
            verify=verify,
        )

        with patch.object(module.ModelPublication, '_verify', acknowledge_new):
            await module.ModelPublication().publish(
                SimpleNamespace(
                    recover=publication.publication_id,
                    restore=False,
                ),
            )
        self.assertEqual((await self.current())['version'], 'v2')
        async with self.sessions() as db:
            self.assertEqual(
                (
                    await db.get(
                        ModelPublicationRecord,
                        publication.publication_id,
                    )
                ).phase,
                'verified',
            )

    async def test_failed_recovery_remains_pending(self):
        from examples.auth.models import ModelPublicationRecord

        publication = await self.interrupted_publication()
        await module.catalog.apply_models(
            publication.prepared,
            publication_id=publication.publication_id,
            expected=publication.previous,
        )
        recover = SimpleNamespace(
            recover=publication.publication_id,
            restore=True,
        )
        with patch.object(
            module.ModelPublication,
            '_verify',
            new=AsyncMock(side_effect=TimeoutError),
        ):
            with self.assertRaises(TimeoutError):
                await module.ModelPublication().publish(recover)
        async with self.sessions() as db:
            self.assertEqual(
                (
                    await db.get(
                        ModelPublicationRecord,
                        publication.publication_id,
                    )
                ).phase,
                'catalog_restored',
            )
        await module.ModelPublication().publish(recover)
        self.assertEqual((await self.current())['version'], 'v1')
