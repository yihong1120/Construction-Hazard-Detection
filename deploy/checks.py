from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

from deploy.keycloak.session_policy import generated_files
from src.settings.sessions import SESSION_POLICY


def changed_files(base: str) -> list[str]:
    if not base or set(base) == {'0'}:
        try:
            base = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD^'],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except subprocess.CalledProcessError:
            return (
                subprocess.check_output(
                    ['git', 'ls-tree', '-rz', '--name-only', 'HEAD'],
                )
                .decode()
                .split('\0')
            )
    return (
        subprocess.check_output(
            [
                'git',
                'diff',
                '--name-only',
                '--diff-filter=ACMR',
                '-z',
                base + '...HEAD',
            ],
        )
        .decode()
        .split('\0')
    )


def check_paths(paths: list[str]) -> None:
    python = []
    for name in paths:
        path = Path(name)
        if not name.startswith('deploy/') or not path.is_file():
            continue
        if path.suffix == '.py':
            compile(path.read_text(), name, 'exec', ast.PyCF_ONLY_AST)
            python.append(name)
        elif path.suffix == '.json':
            json.loads(path.read_text())
        elif (
            path.suffix == '.sh'
            or name == 'deploy/keycloak/session-policy.env'
        ):
            subprocess.run(['bash', '-n', name], check=True)
    if python:
        subprocess.run(
            [
                sys.executable,
                '-m',
                'flake8',
                '--select=E9,F',
                *python,
            ],
            check=True,
        )
    stale = [
        str(p)
        for p, value in generated_files(
            SESSION_POLICY,
        ).items()
        if not p.exists() or p.read_text() != value
    ]
    if stale:
        raise ValueError('Regenerate session policy: ' + ', '.join(stale))


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Validate changed deployment tools without running them',
    )
    parser.add_argument('--base')
    parser.add_argument('--files', nargs='*')
    args = parser.parse_args()
    if args.files is None and args.base is None:
        parser.error('Provide --base or --files')
    check_paths(
        args.files if args.files is not None else changed_files(args.base),
    )


if __name__ == '__main__':
    main()
