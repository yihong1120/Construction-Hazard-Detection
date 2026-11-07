from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from deploy.dev.database_parity import production_options
from deploy.dev.run import CONFIG
from deploy.dev.run import ROOT
from deploy.dev.run import STATE
from deploy.dev.run import validate


def dev_entry(row):
    definition = dict(row['definition'])
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', row['model_key']):
        raise RuntimeError('Invalid model key.')
    if not re.fullmatch(r'[0-9a-f]{64}', definition['sha256']):
        raise RuntimeError('Invalid model checksum.')
    source = Path(definition['artifact'])
    if not source.is_absolute():
        source = ROOT / source
    if (
        not source.resolve().is_relative_to((ROOT / 'models/pt').resolve())
        or source.suffix != '.pt'
    ):
        raise RuntimeError(
            'Only existing local PyTorch model artifacts may be copied.',
        )
    target = (
        STATE
        / 'models/pt'
        / (row['model_key'] + '-' + definition['sha256'] + '.pt')
    )
    definition.update(
        id=row['model_key'],
        artifact=str(target),
        tenant_ids=[str(TENANT)],
        site_ids=[],
        capabilities=['image'],
        hf_repo=None,
        hf_filename=None,
        hf_revision=None,
    )
    return source, target, definition


TENANT = 'd8c43372-0609-4afb-b903-39cae0f2e4eb'


def main():
    if os.geteuid() == 0 or Path.home() != Path('/home/changdar'):
        raise RuntimeError('Run as changdar on A without sudo.')
    import psycopg
    from psycopg.rows import dict_row

    env = json.loads((CONFIG / 'environment.json').read_text())
    env.update(
        {
            'BFF_DETECTION_API_URL': 'http://127.0.0.1:19000',
            'YOLO_INFERENCE_DEVICE': 'cuda:0',
            'USE_TENSORRT': 'false',
            'USE_SAHI': 'false',
            'LAZY_LOAD_MODELS': 'true',
            'PRELOAD_SMALLEST': 'false',
            'MAX_LOADED_MODELS': '1',
            'DETECT_INGRESS_CONCURRENCY': '1',
            'OMP_NUM_THREADS': '2',
            'MKL_NUM_THREADS': '2',
        },
    )
    validate(env)
    with psycopg.connect(**production_options(), row_factory=dict_row) as db:
        db.read_only = True
        rows = db.execute(
            """SELECT model_key,definition,is_default,display_order
            FROM construction_hazard_detection.detection_model_catalog
            ORDER BY display_order,model_key""",
        ).fetchall()
    catalog = []
    for row in rows:
        if (
            not row['definition'].get('enabled', True)
            or 'image' not in row['definition']['capabilities']
        ):
            continue
        source, target, definition = dev_entry(row)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not target.is_file():
            with tempfile.NamedTemporaryFile(
                dir=target.parent,
                delete=False,
            ) as output:
                temporary = Path(output.name)
                try:
                    with source.open('rb') as stream:
                        shutil.copyfileobj(stream, output)
                    output.flush()
                    with temporary.open('rb') as stream:
                        if (
                            hashlib.file_digest(stream, 'sha256').hexdigest()
                            != definition['sha256']
                        ):
                            raise RuntimeError(
                                'Model artifact checksum mismatch.',
                            )
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
        with target.open('rb') as stream:
            if (
                hashlib.file_digest(stream, 'sha256').hexdigest()
                != definition['sha256']
            ):
                raise RuntimeError('Dev model artifact checksum mismatch.')
        catalog.append(
            dict(
                model_key=row['model_key'],
                definition=definition,
                display_order=row['display_order'],
                is_default=row['is_default'],
            ),
        )
    if not catalog or sum(r['is_default'] for r in catalog) != 1:
        raise RuntimeError('An image catalog with one default is required.')
    from deploy.dev.setup import PYTHON, private, unit, nginx_configuration

    backup = STATE / 'detection-deploy' / str(time.time_ns())
    backup.mkdir(parents=True, mode=0o700)
    for path in (
        CONFIG / 'environment.json',
        CONFIG / 'detection-catalog.json',
        STATE / 'gateway/nginx.conf',
    ):
        if path.exists():
            shutil.copy2(path, backup / path.name)
    private(CONFIG / 'detection-catalog.json', json.dumps(catalog, indent=2))
    private(CONFIG / 'environment.json', json.dumps(env, indent=2))
    subprocess.run(
        [str(PYTHON), '-m', 'deploy.dev.seed'],
        cwd=ROOT,
        check=True,
    )
    unit(
        'detection',
        (
            'Type=simple\nExecStart='
            f"{PYTHON}"
            ' '
            f"{ROOT}"
            '/deploy/dev/run.py detection\nRestart=on-failu'
            're\nRestartSec=5\nUMask=0077\nMemoryMax=4G\nCP'
            'UQuota=200%'
        ),
    )
    candidate = STATE / 'gateway/nginx-detection-candidate.conf'
    candidate.write_text(nginx_configuration())
    subprocess.run(
        [
            '/usr/sbin/nginx',
            '-t',
            '-p',
            str(
                STATE / 'gateway',
            ),
            '-c',
            str(candidate),
        ],
        check=True,
    )
    candidate.replace(STATE / 'gateway/nginx.conf')
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    subprocess.run(
        [
            'systemctl',
            '--user',
            'enable',
            '--now',
            'construction-dev-detection.service',
        ],
        check=True,
    )
    subprocess.run(
        [
            'systemctl',
            '--user',
            'restart',
            'construction-dev-management.service',
            'construction-dev-gateway.service',
        ],
        check=True,
    )
    print(
        f"Dev YOLO enabled on 127.0.0.1:19000 with {
            len(catalog)
        } verified model artifacts; concurrency one.",
    )


if __name__ == '__main__':
    main()
