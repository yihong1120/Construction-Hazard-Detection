from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_spec = importlib.util.spec_from_file_location(
    'python315_bootstrap',
    Path(__file__).resolve().parents[2] / 'deploy/python315/bootstrap.py',
)
assert _spec is not None and _spec.loader is not None
bootstrap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bootstrap)


class SourceCheckoutTest(unittest.TestCase):
    def test_failed_fetch_can_retry_pinned_commit_from_mirror(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            cache = Path(root)
            mirror = cache / 'mirror'
            git = ('git', '-C', str(mirror))
            subprocess.run(
                ['git', 'init', '--quiet', mirror], check=True,
            )
            subprocess.run(
                [
                    *git, '-c', 'user.name=Fixture',
                    '-c', 'user.email=fixture@example.com',
                    'commit', '--allow-empty', '--no-gpg-sign',
                    '--quiet', '-m', 'Pinned source',
                ],
                check=True,
            )
            commit = subprocess.check_output(
                [*git, 'rev-parse', 'HEAD'], text=True,
            ).strip()
            unavailable = str(cache / 'unavailable')
            checkout = cache / 'source'
            with self.assertRaises(subprocess.CalledProcessError):
                bootstrap.checkout(unavailable, checkout, commit)

            bootstrap.checkout(
                unavailable, checkout, commit, mirror=mirror.as_uri(),
            )
            revision = subprocess.check_output(
                ['git', '-C', checkout, 'rev-parse', 'HEAD'], text=True,
            ).strip()
            self.assertEqual(revision, commit)
            with patch.object(bootstrap, 'run') as run:
                bootstrap.checkout(unavailable, checkout, commit)
                run.assert_not_called()


class NativeWheelCacheTest(unittest.TestCase):
    def test_cached_cpu_wheel_skips_native_toolchain_setup(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            repo = Path(root)
            wheels = repo / 'wheels/python315'
            wheels.mkdir(parents=True)
            wheel = wheels / (
                f'onnxruntime-{bootstrap.ORT_VERSION}'
                '-cp315-cp315-linux_x86_64.whl'
            )
            wheel.write_bytes(b'cached artifact')
            args = [
                'bootstrap.py', '--cpu-only', '--build-only',
                '--cache-dir', str(repo / 'cache'),
            ]
            with (
                patch.object(bootstrap, 'REPO', repo),
                patch.object(bootstrap.sys, 'argv', args),
                patch.object(bootstrap, 'run') as run,
            ):
                bootstrap.main()
            run.assert_not_called()
            manifest = json.loads((wheels / 'manifest.json').read_text())
            self.assertEqual(
                manifest['sha256'][wheel.name],
                hashlib.sha256(wheel.read_bytes()).hexdigest(),
            )
