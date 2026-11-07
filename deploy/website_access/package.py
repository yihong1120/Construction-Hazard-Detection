from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from deploy.website_access.deploy import NEW_FILES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in NEW_FILES:
        target = destination / 'payload' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
        paths.append(target)
    for name in (
        'deploy.py', 'agent.py', 'snapshot.py', 'migrate.py',
        'README.md',
    ):
        target = destination / name
        shutil.copyfile(Path(__file__).parent / name, target)
        paths.append(target)
    target = destination / 'migration.sql'
    shutil.copyfile(
        root / 'deploy/website_access/migration.sql',
        target,
    )
    paths.append(target)
    checksums = {
        str(p.relative_to(destination)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
    }
    (destination / 'checksums.json').write_text(
        json.dumps(checksums, indent=2) + '\n',
    )
    print('Prepared ' + str(len(paths)) + ' files: ' + str(destination))


if __name__ == '__main__':
    main()
