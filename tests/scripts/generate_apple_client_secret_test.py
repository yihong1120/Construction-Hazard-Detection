"""Exercise credential generation through the real command-line entrypoint."""
from __future__ import annotations

import os
import stat
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / 'scripts/generate_apple_client_secret.py'
)


class AppleClientSecretCliTest(unittest.TestCase):
    """Secrets reach only a private file, including on CLI failures."""

    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.private_key = ec.generate_private_key(ec.SECP256R1())
        self.pem = self.private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        self.key_file = self.root / 'apple.p8'
        self.key_file.write_bytes(self.pem)
        self.output = self.root / 'client-secret.jwt'

    def run_cli(self, *extra: str, output: bool = True):
        """Run the script with a disposable real ES256 key."""
        command = [
            sys.executable, str(SCRIPT),
            '--team-id', 'TESTTEAM', '--key-id', 'TESTKEY',
            '--client-id', 'com.example.signin',
            '--private-key-file', str(self.key_file),
        ]
        if output:
            command.extend(['--output-file', str(self.output)])
        command.extend(extra)
        return subprocess.run(
            command, capture_output=True, text=True, timeout=10,
        )

    def test_signed_secret_is_private_and_absent_from_terminal_output(self):
        """A permissive umask cannot make the generated JWT public."""
        previous_umask = os.umask(0)
        try:
            result = self.run_cli()
        finally:
            os.umask(previous_umask)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, '')
        secret = self.output.read_text().strip()
        self.assertNotIn(secret, result.stdout + result.stderr)
        self.assertNotIn(self.pem.decode(), result.stdout + result.stderr)
        self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o600)
        claims = jwt.decode(
            secret, self.private_key.public_key(), algorithms=['ES256'],
            audience='https://appleid.apple.com', issuer='TESTTEAM',
        )
        self.assertEqual(claims['sub'], 'com.example.signin')
        self.assertEqual(claims['exp'] - claims['iat'], 150 * 86400)
        self.assertEqual(jwt.get_unverified_header(secret)['kid'], 'TESTKEY')
        self.assertEqual(self.key_file.read_bytes(), self.pem)

    def test_output_path_is_required(self):
        """Omitting a destination must never fall back to stdout."""
        result = self.run_cli(output=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '')
        self.assertIn('--output-file', result.stderr)
        self.assertFalse(self.output.exists())

    def test_existing_secret_is_not_overwritten(self):
        """Rotation requires an explicit new path and preserves prior files."""
        self.output.write_text('existing-credential')
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.output.read_text(), 'existing-credential')
        self.assertEqual(result.stdout, '')
        self.assertNotIn('existing-credential', result.stderr)

    def test_symlink_cannot_redirect_the_secret_write(self):
        """A pre-existing symlink cannot overwrite another destination."""
        target = self.root / 'target'
        target.write_text('unchanged')
        self.output.symlink_to(target)
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(self.output.is_symlink())
        self.assertEqual(target.read_text(), 'unchanged')
        self.assertEqual(result.stdout, '')

    def test_invalid_key_is_not_echoed_by_error_reporting(self):
        """Signing errors do not expose the input key or leave a file."""
        invalid_key = 'private-key-material-that-must-never-be-logged'
        self.key_file.write_text(invalid_key)
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '')
        self.assertNotIn(invalid_key, result.stderr)
        self.assertNotIn('Traceback', result.stderr)
        self.assertFalse(self.output.exists())

    def test_invalid_lifetime_does_not_create_a_secret(self):
        """Apple's maximum lifetime and positive duration remain enforced."""
        for days in ('0', '181'):
            with self.subTest(days=days):
                result = self.run_cli('--expires-in-days', days)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, '')
                self.assertFalse(self.output.exists())

    def test_unwritable_destination_reports_no_credential(self):
        """Invalid destination directories produce only a safe error."""
        self.output = self.root / 'missing-directory' / 'secret.jwt'
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '')
        self.assertNotIn(self.pem.decode(), result.stderr)
        self.assertFalse(self.output.exists())
