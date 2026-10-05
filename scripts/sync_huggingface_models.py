"""Publish model keys from Hugging Face into the database catalog.

Run from the repository: python -m scripts.sync_huggingface_models.
Weights are never uploaded by this command. Existing grants are preserved.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path

from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert

from examples.auth.database import AsyncSessionLocal
from examples.auth.database import engine
from examples.auth.models import DetectionModelCatalog
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

    def matches(path):
        if not path.is_file():
            return False
        with path.open('rb') as f:
            return hashlib.file_digest(f, 'sha256').hexdigest() == digest

    if matches(local):
        source = local
    else:
        source = Path(hf_hub_download(repo, filename, revision=info.sha))
        if not matches(source):
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
            name, (re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_'), name),
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


async def synchronize(args):
    bootstrap = {}
    if args.import_registry:
        bootstrap = {
            m['id']: m
            for m in json.loads(Path(args.import_registry).read_text())[
                'models'
            ]
        }
    prepared = []
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
        # All downloads/metadata validation finish before changing publication.
        if args.default:
            if args.default not in args.model_key:
                raise ValueError('--default must be a selected key')
            await db.execute(
                update(DetectionModelCatalog).values(is_default=False),
            )
        for entry, order, default in prepared:
            values = dict(
                model_key=entry.id,
                definition=entry.model_dump(exclude={'id'}),
                display_order=order,
                is_default=(entry.id == args.default)
                if args.default
                else default,
            )
            statement = insert(DetectionModelCatalog).values(**values)
            await db.execute(
                statement.on_conflict_do_update(
                    index_elements=['model_key'], set_=values,
                ),
            )
        await db.commit()
    for entry, _, _ in prepared:
        print(
            f'{entry.id}: published {entry.hf_revision} '
            f'({len(entry.classes)} classes)',
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-key', action='append', required=True)
    parser.add_argument(
        '--repo', default='yihong1120/Construction-Hazard-Detection',
    )
    parser.add_argument('--revision', default='main')
    parser.add_argument(
        '--filename', help='Required for a custom key; use with one key only',
    )
    parser.add_argument('--tenant-id', action='append')
    parser.add_argument('--site-id', action='append', type=int)
    parser.add_argument('--default')
    parser.add_argument(
        '--import-registry',
        help='One-time migration of existing permissions; not used at runtime',
    )
    args = parser.parse_args()
    if args.filename and len(args.model_key) != 1:
        parser.error('--filename requires exactly one key')

    async def run():
        try:
            await synchronize(args)
        finally:
            await engine.dispose()

    asyncio.run(run())


if __name__ == '__main__':
    main()
