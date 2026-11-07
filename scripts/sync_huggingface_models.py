from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from datetime import datetime
from datetime import timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy import text
from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert

from examples.auth.database import AsyncSessionLocal
from examples.auth.database import engine
from examples.auth.models import DetectionModelCatalog
from examples.auth.models import ModelPublicationRecord
from examples.YOLO_server_api.model_registry import ModelEntry

LABELS = {
    'Hardhat': ('helmet', '安全帽'),
    'Mask': ('mask', '口罩'),
    'NO-Hardhat': ('no_helmet', '未戴安全帽'),
    'NO-Mask': ('no_mask', '未戴口罩'),
    'NO-Safety Vest': ('no_safety_vest', '未穿安全背心'),
    'Person': ('person', '人員'),
    'Safety Cone': ('safety_cone', '交通錐'),
    'Safety Vest': ('safety_vest', '安全背心'),
    'machinery': ('machinery', '機具'),
    'utility pole': ('utility_pole', '電桿'),
    'vehicle': ('vehicle', '車輛'),
}
ROOT = Path(__file__).resolve().parents[1]


class PublicationConflict(RuntimeError):
    pass


def resolve_model(key, repo, revision, filename, previous):
    from huggingface_hub import HfApi, hf_hub_download
    from ultralytics import YOLO

    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,119}', key):
        raise ValueError('Invalid model key')
    if filename is None:
        family = re.fullmatch(r'(yolo[0-9]+)[nslmx]', key)
        if not family:
            raise ValueError('Custom model keys require --filename')
        filename = f'models/{family[1]}/pt/{key}.pt'
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    remote = next((f for f in info.siblings if f.rfilename == filename), None)
    if remote is None or remote.lfs is None:
        raise ValueError(
            'The selected model must have Hugging Face LFS SHA-256 metadata',
        )
    digest = remote.lfs.sha256
    local = ROOT / 'models' / 'pt' / f'best_{key}.pt'

    if _matches_digest(local, digest):
        source = local
    else:
        source = Path(hf_hub_download(repo, filename, revision=info.sha))
        if not _matches_digest(source, digest):
            raise ValueError('Hugging Face model hash mismatch')
    # Only inspect the trusted, hash-verified repository artifact.
    names = YOLO(str(source), task='detect').names
    from examples.shared.class_colors import normalize_color, published_color

    previous_colors = {
        c['code']: normalize_color(c.get('color'))
        for c in previous.get('classes', [])
    }
    classes = []
    for number, name in names.items():
        code, label = LABELS.get(
            name,
            (re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_'), name),
        )
        if not code or code.isdecimal():
            raise ValueError(f'Class {name!r} requires a stable semantic code')
        classes.append(
            {
                'id': int(number),
                'code': code,
                'display_name': label,
                'color': previous_colors.get(code) or published_color(code),
            },
        )
    entry = ModelEntry.model_validate(
        {
            **previous,
            'id': key,
            'display_name': previous.get('display_name', key),
            'version': info.sha,
            'hf_repo': repo,
            'hf_filename': filename,
            'hf_revision': info.sha,
            'sha256': digest,
            'classes': classes,
            # Versioned cache avoids loading yesterday's local best_key.pt.
            'artifact': str(source),
            'capabilities': previous.get('capabilities', ['image', 'stream']),
        },
    )
    return entry


async def prepare_models(
    args: argparse.Namespace,
) -> list[tuple[ModelEntry, int, bool]]:
    bootstrap = {}
    if args.import_registry:
        bootstrap = {
            m['id']: m
            for m in json.loads(Path(args.import_registry).read_text())[
                'models'
            ]
        }
    prepared: list[tuple[ModelEntry, int, bool]] = []
    async with AsyncSessionLocal() as db:
        for key in args.model_key:
            current = await db.get(DetectionModelCatalog, key)
            definition = dict(
                current.definition if current else bootstrap.get(key, {}),
            )
            if args.tenant_id:
                definition['tenant_ids'] = args.tenant_id
            if args.site_id is not None:
                definition['site_ids'] = args.site_id
            if not definition.get('tenant_ids'):
                raise ValueError(
                    'New model requires --tenant-id; access is never inferred',
                )
            entry = await asyncio.to_thread(
                resolve_model,
                key,
                args.repo,
                args.revision,
                args.filename,
                definition,
            )
            prepared.append(
                (
                    entry,
                    current.display_order if current else len(prepared),
                    current.is_default if current else False,
                ),
            )
    return prepared


async def apply_models(
    prepared: list[tuple[ModelEntry, int, bool]],
    default_id: str | None = None,
    expected: dict[str, dict] | None = None,
    publication_id: str | None = None,
) -> None:
    async with AsyncSessionLocal() as db:
        # All downloads/metadata validation finish before changing publication.
        if expected is not None:
            rows = (
                await db.scalars(
                    select(DetectionModelCatalog).with_for_update(),
                )
            ).all()
            current = {row.model_key: row for row in rows}
            for entry, _, _ in prepared:
                before = expected.get(entry.id)
                actual = current.get(entry.id)
                if before is None:
                    unchanged = actual is None
                else:
                    unchanged = actual is not None and (
                        actual.definition == before['definition']
                        and actual.display_order == before['display_order']
                        and actual.is_default == before['is_default']
                    )
                if not unchanged:
                    raise PublicationConflict(
                        'Catalog changed before publication',
                    )
            if default_id and (
                {key for key, row in current.items() if row.is_default}
                != {key for key, row in expected.items() if row['is_default']}
            ):
                raise PublicationConflict(
                    'Catalog default changed before publication',
                )
        if default_id:
            if default_id not in {entry.id for entry, _, _ in prepared}:
                raise ValueError('--default must be a selected key')
            await db.execute(
                update(DetectionModelCatalog).values(is_default=False),
            )
        for entry, order, default in prepared:
            values = dict(
                model_key=entry.id,
                definition=entry.model_dump(exclude={'id'}),
                display_order=order,
                is_default=(entry.id == default_id) if default_id else default,
            )
            statement = insert(DetectionModelCatalog).values(**values)
            await db.execute(
                statement.on_conflict_do_update(
                    index_elements=['model_key'],
                    set_=values,
                ),
            )
        await db.execute(
            text(
                "SELECT pg_notify('stream_config_changed', 'models')",
            ),
        )
        if publication_id is not None:
            record = await db.get(ModelPublicationRecord, publication_id)
            if record is None or record.phase != 'prepared':
                raise PublicationConflict('Publication journal changed')
            record.phase = 'applied'
            record.updated_at = datetime.now(timezone.utc).isoformat()
        await db.commit()
    for entry, _, _ in prepared:
        print(
            f'{entry.id}: published {entry.version} '
            f'({len(entry.classes)} classes)',
        )


async def synchronize(args: argparse.Namespace) -> None:
    await apply_models(await prepare_models(args), args.default)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-key', action='append', default=[])
    parser.add_argument('--recover', metavar='PUBLICATION_ID')
    parser.add_argument(
        '--restore',
        action='store_true',
        help='Restore a pending publication instead of verifying it',
    )
    parser.add_argument(
        '--repo',
        default='yihong1120/Construction-Hazard-Detection',
    )
    parser.add_argument('--revision', default='main')
    parser.add_argument(
        '--filename',
        help='Required for a custom key; use with one key only',
    )
    parser.add_argument('--tenant-id', action='append')
    parser.add_argument('--site-id', action='append', type=int)
    parser.add_argument('--default')
    parser.add_argument(
        '--import-registry',
        help='One-time migration of existing permissions; not used at runtime',
    )
    parser.add_argument('--verify-timeout', type=float, default=180)
    parser.add_argument(
        '--catalog-only',
        action='store_true',
        help='Bootstrap a catalog only when no affected stream is active',
    )
    args = parser.parse_args()
    if not args.recover and not args.model_key:
        parser.error('--model-key or --recover is required')
    if args.restore and not args.recover:
        parser.error('--restore requires --recover')
    if args.recover and args.model_key:
        parser.error('--recover cannot be combined with --model-key')
    if args.verify_timeout <= 0:
        parser.error('--verify-timeout must be positive')
    if args.filename and len(args.model_key) != 1:
        parser.error('--filename requires exactly one key')

    asyncio.run(_publish_models(args))


def _matches_digest(path, digest):
    if not path.is_file():
        return False
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest() == digest


async def _publish_models(args):
    try:
        from scripts.model_publication import ModelPublication

        await ModelPublication().publish(args)
    finally:
        await engine.dispose()


if __name__ == '__main__':
    main()
