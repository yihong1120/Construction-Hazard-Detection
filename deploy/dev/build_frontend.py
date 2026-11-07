from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / 'frontend/web'
STATE = Path.home() / '.local/share/construction-dev'


def manifest():
    required = (
        'index.html',
        'main.dart.js',
        'flutter_bootstrap.js',
        'version.json',
    )
    for name in required:
        if not (FRONTEND / name).is_file():
            raise RuntimeError('Missing frontend/web artifact: ' + name)
    if not (FRONTEND / 'assets').is_dir():
        raise RuntimeError('Missing frontend/web/assets.')
    hashes = {
        name: hashlib.sha256(
            (FRONTEND / name).read_bytes(),
        ).hexdigest()
        for name in required
    }
    build_id = FRONTEND / '.last_build_id'
    value = {
        'environment': 'dev',
        'artifact_source': 'frontend/web',
        'artifact_build_id': (
            build_id.read_text().strip() if build_id.exists() else None
        ),
        'app_version': json.loads((FRONTEND / 'version.json').read_text()),
        'artifact_sha256': hashes,
        'backend_revision': subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
            text=True,
        ).strip(),
        'main_dart_sha256': hashes['main.dart.js'],
        'index_sha256': hashes['index.html'],
    }
    destination = STATE / 'gateway/frontend-manifest.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(destination)
    return value


def main():
    manifest()
    print(
        (
            'Registered existing frontend/web checksums; artifact files are '
            'unchanged.'
        ),
    )


if __name__ == '__main__':
    main()
