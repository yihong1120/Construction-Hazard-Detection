from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deploy.checks import check_paths


class DeploymentChecksTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        original = Path.cwd()
        os.chdir(self.directory.name)
        self.addCleanup(os.chdir, original)
        Path('deploy').mkdir()
        operation = patch('deploy.checks.generated_files', return_value={})
        operation.start()
        self.addCleanup(operation.stop)

    def test_shell_checks_parse_without_executing_deployment_commands(
        self,
    ) -> None:
        path = Path('deploy/with spaces.sh')
        path.write_text('touch must-not-exist\n')
        check_paths([str(path)])
        self.assertFalse(Path('must-not-exist').exists())
        path.write_text('if then\n')
        with self.assertRaises(subprocess.CalledProcessError):
            check_paths([str(path)])

    def test_python_checks_lint_and_never_execute_script(self) -> None:
        path = Path('deploy/test.py')
        path.write_text("raise RuntimeError('must not execute')\n")
        with patch('deploy.checks.subprocess.run') as lint:
            check_paths([str(path)])
        self.assertEqual(
            lint.call_args.args[0][1:4],
            [
                '-m',
                'flake8',
                '--select=E9,F',
            ],
        )
        path.write_text('if )\n')
        with self.assertRaises(SyntaxError):
            check_paths([str(path)])

    def test_json_checks_reject_invalid_content(self) -> None:
        path = Path('deploy/settings.json')
        path.write_text('{invalid')
        with self.assertRaises(ValueError):
            check_paths([str(path)])
