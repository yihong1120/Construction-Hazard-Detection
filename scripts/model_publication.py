from __future__ import annotations

import argparse
import asyncio
import json
import math
import secrets
from copy import deepcopy
from datetime import datetime
from datetime import timezone
from pathlib import Path

from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy import text
from sqlalchemy import update

from examples.auth.database import AsyncSessionLocal
from examples.auth.database import engine
from examples.auth.models import DetectionModelCatalog
from examples.auth.models import ModelPublicationRecord
from examples.auth.models import StreamConfig
from examples.shared.model_status import model_challenge_key
from examples.shared.model_status import model_status_key
from examples.YOLO_server_api.model_registry import ModelEntry
from examples.YOLO_server_api.model_registry import revision_for
from examples.YOLO_server_api.model_registry import verify_artifact
from scripts import sync_huggingface_models as catalog
from src.redis_client import RedisManager
from src.settings.inference import WorkerSettings
from src.settings.precision import _parse_worker_precision
from src.yolo_runtime_artifacts import stream_engine_path


class ModelPublication:
    """Serialize publication, verify loaded models and compensate failures."""

    def __init__(self) -> None:
        self.previous: dict[str, dict] = {}
        self.prepared: list[tuple[ModelEntry, int, bool]] = []
        self.redis: Redis | None = None
        self.selected_default: str | None = None
        self.challenge = ''
        self.publication_id: str | None = None
        self.verified_streams: tuple[int, ...] = ()

    async def publish(self, args: argparse.Namespace) -> None:
        async with engine.begin() as connection:
            acquired = await connection.scalar(
                text(
                    'SELECT pg_try_advisory_xact_lock(675839201)',
                ),
            )
            if not acquired:
                raise RuntimeError('Another model publication is running')
            try:
                if getattr(args, 'recover', None):
                    await self._recover(args)
                else:
                    await self._assert_no_pending()
                    await self._publish_locked(args)
            finally:
                if self.redis is not None:
                    await self.redis.aclose()

    async def _assert_no_pending(self) -> None:
        async with AsyncSessionLocal() as db:
            pending = await db.scalar(
                select(ModelPublicationRecord).where(
                    ModelPublicationRecord.phase.not_in(
                        ('verified', 'restored', 'aborted'),
                    ),
                ),
            )
            if pending is not None:
                raise RuntimeError(
                    f'Pending publication: use --recover {pending.id}',
                )

    async def _phase(self, phase: str, db=None) -> None:
        if self.publication_id is None:
            return
        if db is None:
            async with AsyncSessionLocal() as session:
                await self._phase(phase, session)
                await session.commit()
            return
        record = await db.get(ModelPublicationRecord, self.publication_id)
        if record is None:
            raise RuntimeError('Publication journal is missing')
        if phase in ('verified', 'restored'):
            record.snapshot = {
                **record.snapshot,
                'verified_streams': list(self.verified_streams),
            }
        record.phase = phase
        record.updated_at = datetime.now(timezone.utc).isoformat()

    async def _record(self, affected: dict[int, str], timeout: float) -> None:
        self.publication_id = secrets.token_hex(16)
        async with AsyncSessionLocal() as db:
            db.add(
                ModelPublicationRecord(
                    id=self.publication_id,
                    phase='prepared',
                    updated_at=datetime.now(timezone.utc).isoformat(),
                    snapshot={
                        'format_version': 1,
                        'previous': self.previous,
                        'prepared': [
                            [entry.model_dump(), order, default]
                            for entry, order, default in self.prepared
                        ],
                        'selected_default': self.selected_default,
                        'affected': {
                            str(key): value for key, value in affected.items()
                        },
                        'timeout': timeout,
                    },
                ),
            )
            await db.commit()
        print(f'Publication {self.publication_id}', flush=True)

    async def _recover(self, args: argparse.Namespace) -> None:
        async with AsyncSessionLocal() as db:
            record = await db.get(ModelPublicationRecord, args.recover)
            if record is None:
                raise ValueError('Unknown publication')
            phase, snapshot = record.phase, record.snapshot
        if phase in ('verified', 'restored', 'aborted'):
            print(f'Publication {args.recover}: {phase}')
            return
        if snapshot.get('format_version') != 1:
            raise ValueError('Unsupported publication journal version')
        self.publication_id = args.recover
        self.previous = snapshot['previous']
        self.prepared = [
            (ModelEntry.model_validate(entry), order, default)
            for entry, order, default in snapshot['prepared']
        ]
        self.selected_default = snapshot['selected_default']
        timeout = snapshot['timeout']
        if await self._active_streams():
            self.redis = RedisManager().redis
            await self.redis.ping()
        if args.restore or phase in (
            'prepared',
            'restoring',
            'catalog_restored',
        ):
            await self._restore_and_verify(timeout)
        else:
            await self._assert_applied()
            await self._prepare_challenge(timeout)
            await self._verify(self._selected_revisions(), timeout)
            await self._phase('verified')
        print(f'Recovered publication {args.recover}')

    async def _assert_applied(self) -> None:
        async with AsyncSessionLocal() as db:
            rows = (await db.scalars(select(DetectionModelCatalog))).all()
            current = {row.model_key: row for row in rows}
            for entry, order, _ in self.prepared:
                row = current.get(entry.id)
                if (
                    row is None
                    or row.definition != entry.model_dump(exclude={'id'})
                    or row.display_order != order
                ):
                    raise catalog.PublicationConflict(
                        'Catalog changed after publication',
                    )
            if self.selected_default and {
                row.model_key for row in rows if row.is_default
            } != {self.selected_default}:
                raise catalog.PublicationConflict(
                    'Catalog default changed after publication',
                )

    async def _publish_locked(self, args: argparse.Namespace) -> None:
        async with AsyncSessionLocal() as db:
            rows = (await db.scalars(select(DetectionModelCatalog))).all()
            self.previous = {
                row.model_key: {
                    'definition': deepcopy(row.definition),
                    'display_order': row.display_order,
                    'is_default': row.is_default,
                }
                for row in rows
            }
        self.prepared = await catalog.prepare_models(args)
        self.selected_default = args.default
        settings = WorkerSettings()
        precision = _parse_worker_precision(settings.precision)
        manifest = (
            Path(settings.artifact_manifest)
            if precision == 'int8' and settings.artifact_manifest
            else None
        )
        if precision == 'int8' and manifest is None:
            raise ValueError('INT8 publication requires an engine manifest')
        for entry, _, _ in self.prepared:
            await asyncio.to_thread(verify_artifact, entry)
            if manifest is not None:
                await asyncio.to_thread(
                    stream_engine_path,
                    manifest,
                    entry.id,
                    entry.version,
                    entry.sha256,
                )
            previous = self.previous.get(entry.id)
            if previous is not None:
                old_entry = ModelEntry(id=entry.id, **previous['definition'])
                await asyncio.to_thread(verify_artifact, old_entry)
                if manifest is not None:
                    await asyncio.to_thread(
                        stream_engine_path,
                        manifest,
                        old_entry.id,
                        old_entry.version,
                        old_entry.sha256,
                    )
        affected = await self._active_streams()
        if args.catalog_only and affected:
            raise ValueError(
                '--catalog-only cannot bypass active stream verification',
            )
        if affected:
            self.redis = RedisManager().redis
            await self.redis.ping()
            await self._prepare_challenge(args.verify_timeout)
        await self._record(affected, args.verify_timeout)
        try:
            await catalog.apply_models(
                self.prepared,
                args.default,
                expected=self.previous,
                publication_id=self.publication_id,
            )
            await self._verify(
                self._selected_revisions(),
                args.verify_timeout,
            )
            await self._phase('verified')
            print(
                f'Verified publication for {len(self.verified_streams)} '
                'active streams',
            )
        except catalog.PublicationConflict:
            await self._phase('aborted')
            raise
        except BaseException as error:
            # Finish compensation even if the CLI receives cancellation after
            # the publication commit. A competing edit is never overwritten.
            restoration = asyncio.create_task(
                self._restore_and_verify(
                    args.verify_timeout,
                ),
            )
            try:
                await asyncio.shield(restoration)
            except asyncio.CancelledError:
                await restoration
            except Exception as rollback_error:
                raise RuntimeError(
                    'Publication failed; automatic restoration '
                    'needs attention',
                ) from rollback_error
            if isinstance(error, asyncio.CancelledError):
                raise
            raise RuntimeError(
                'Publication failed; previous models were '
                'restored and verified',
            ) from error

    def _selected_revisions(self) -> dict[str, str]:
        return {
            entry.id: revision_for(entry.model_dump(exclude={'id'}))
            for entry, _, _ in self.prepared
        }

    async def _prepare_challenge(self, timeout: float) -> None:
        self.challenge = secrets.token_hex(16)
        if self.redis is not None:
            async with self.redis.pipeline(transaction=True) as pipe:
                for entry, _, _ in self.prepared:
                    pipe.set(
                        model_challenge_key(entry.id),
                        self.challenge,
                        ex=math.ceil(timeout) + 60,
                    )
                await pipe.execute()

    async def _active_streams(self) -> dict[int, str]:
        async with AsyncSessionLocal() as db:
            rows = await db.execute(
                select(
                    StreamConfig.id,
                    StreamConfig.model_key,
                ).where(
                    StreamConfig.model_key.in_(
                        [e.id for e, _, _ in self.prepared],
                    ),
                    StreamConfig.recognition_enabled.is_(True),
                    (StreamConfig.expire_date.is_(None))
                    | (StreamConfig.expire_date > datetime.now(timezone.utc)),
                ),
            )
            return dict(rows.all())

    async def _verify(self, revisions: dict[str, str], timeout: float) -> None:
        verification = _PublicationVerification(self, revisions, timeout)
        await asyncio.wait_for(verification.wait(), timeout=timeout)

    async def _restore_catalog(self) -> None:
        async with AsyncSessionLocal() as db:
            rows = (
                await db.scalars(
                    select(DetectionModelCatalog).with_for_update(),
                )
            ).all()
            current = {row.model_key: row for row in rows}
            current_defaults = {
                row.model_key for row in rows if row.is_default
            }
            previous_defaults = {
                key for key, row in self.previous.items() if row['is_default']
            }
            unchanged = True
            for entry, _, _ in self.prepared:
                before = self.previous.get(entry.id)
                actual = current.get(entry.id)
                if before is None:
                    unchanged &= actual is None
                else:
                    unchanged &= actual is not None and (
                        actual.definition == before['definition']
                        and actual.display_order == before['display_order']
                    )
            if unchanged and (
                self.selected_default is None
                or current_defaults == previous_defaults
            ):
                await self._phase('catalog_restored', db)
                await db.commit()
                return
            if self.selected_default and current_defaults != {
                self.selected_default,
            }:
                raise RuntimeError(
                    'Catalog default changed during publication',
                )
            for entry, order, _ in self.prepared:
                row = current.get(entry.id)
                if row is None or (
                    row.definition != entry.model_dump(exclude={'id'})
                    or row.display_order != order
                ):
                    raise RuntimeError('Catalog changed during publication')
            if self.selected_default:
                await db.execute(
                    update(DetectionModelCatalog).values(is_default=False),
                )
            for entry, _, _ in self.prepared:
                row = current[entry.id]
                previous = self.previous.get(entry.id)
                if previous is None:
                    await db.delete(row)
                else:
                    row.definition = deepcopy(previous['definition'])
                    row.display_order = previous['display_order']
            await db.flush()
            if self.selected_default:
                for key in previous_defaults:
                    if key in current:
                        current[key].is_default = True
            await db.execute(
                text(
                    "SELECT pg_notify('stream_config_changed', 'models')",
                ),
            )
            await self._phase('catalog_restored', db)
            await db.commit()

    async def _restore_and_verify(
        self,
        timeout: float,
    ) -> None:
        await self._phase('restoring')
        await self._prepare_challenge(timeout)
        await self._restore_catalog()
        revisions = {
            key: revision_for(
                ModelEntry(
                    id=key,
                    **previous['definition'],
                ).model_dump(exclude={'id'}),
            )
            for key, previous in self.previous.items()
        }
        await self._verify(revisions, timeout)
        await self._phase('restored')


class _PublicationVerification:
    def __init__(
        self,
        publication: ModelPublication,
        revisions: dict[str, str],
        timeout: float,
    ):
        self.publication = publication
        self.revisions = revisions
        self.timeout = timeout

    async def wait(self) -> None:

        while True:
            active = await self.publication._active_streams()
            pending = {
                key: model
                for key, model in active.items()
                if model in self.revisions
            }
            if pending and self.publication.redis is None:
                self.publication.redis = RedisManager().redis
                await self.publication.redis.ping()
                await self.publication._prepare_challenge(self.timeout)
            raw = (
                await self.publication.redis.mget(
                    [model_status_key(key) for key in pending],
                )
                if pending and self.publication.redis is not None
                else []
            )
            if all(
                self.acknowledged(value, model)
                for model, value in zip(pending.values(), raw, strict=True)
            ):
                confirmed = await self.publication._active_streams()
                if pending == {
                    key: model
                    for key, model in confirmed.items()
                    if model in self.revisions
                }:
                    self.publication.verified_streams = tuple(sorted(pending))
                    return
            await asyncio.sleep(1)

    def acknowledged(self, value, model):
        try:
            status = json.loads(value) if value is not None else None
        except (TypeError, ValueError):
            return False
        return isinstance(status, dict) and (
            status.get('model_id') == model
            and status.get('catalog_revision') == self.revisions[model]
            and status.get('publication') == self.publication.challenge
        )
