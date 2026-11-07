from __future__ import annotations

import hashlib
import importlib.util
import json
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
