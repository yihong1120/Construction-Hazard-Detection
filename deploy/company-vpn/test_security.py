from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import enroll_b as e

import deploy as d


class CredentialOutputTest(unittest.TestCase):
    def test_recovery_secret_is_never_written_to_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            credentials = root / 'bootstrap.json'
            credentials.write_text(
                json.dumps({
                    'email': 'recovery@example.test',
                    'password': 'test-only-recovery-secret',
                }),
            )
            output = StringIO()
            with patch.object(d, 'ROOT', root), redirect_stdout(output):
                d.show_credentials_path()
            self.assertIn(str(credentials), output.getvalue())
            self.assertNotIn('test-only-recovery-secret', output.getvalue())
            self.assertNotIn('recovery@example.test', output.getvalue())

    def test_missing_recovery_file_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(d, 'ROOT', Path(directory)):
                with self.assertRaisesRegex(RuntimeError, 'missing'):
                    d.show_credentials_path()


class ClientPortPreflightTest(unittest.TestCase):
    def test_existing_udp_listener_is_rejected_without_binding(self):
        with patch.object(e.d, 'run', return_value='occupied') as command:
            with self.assertRaisesRegex(RuntimeError, '51820'):
                e.ensure_client_port_available()
            command.assert_called_once_with(
                'ss', '-H', '-lun', 'sport = :51820',
            )

    def test_free_udp_port_is_accepted_without_binding(self):
        with patch.object(e.d, 'run', return_value=''):
            e.ensure_client_port_available()
