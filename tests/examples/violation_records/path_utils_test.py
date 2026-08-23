from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import HTTPException

from examples.violation_records.path_utils import _determine_media_type
from examples.violation_records.path_utils import normalise_safe_relative_path
from examples.violation_records.path_utils import resolve_authorised_media_path


class TestNormalizeSafeRelPath(unittest.TestCase):
    """Tests for normalising and sanitising user-supplied relative paths."""

    def test_reject_absolute_path(self) -> None:
        """Absolute paths must be rejected with HTTP 400.

        Given an absolute file system path, when validated, the function should
        raise an HTTPException with a 400 status code.
        """
        abs_path: str = str(Path('/etc/passwd'))
        with self.assertRaises(HTTPException) as cm:
            normalise_safe_relative_path(abs_path)
        self.assertEqual(cm.exception.status_code, 400)
        self.assertIn('Invalid path', cm.exception.detail)

    def test_reject_traversal_component(self) -> None:
        """Traversal tokens ('..') must be rejected with HTTP 400."""
        with self.assertRaises(HTTPException) as cm:
            normalise_safe_relative_path('a/../b.png')
        self.assertEqual(cm.exception.status_code, 400)
        self.assertIn('Invalid path', cm.exception.detail)

    def test_reject_leading_static(self) -> None:
        """The API accepts only paths relative to the static directory."""
        with self.assertRaises(HTTPException) as cm:
            normalise_safe_relative_path('static/2025-01-01/img.png')
        self.assertEqual(cm.exception.status_code, 400)

    def test_segment_sanitization_empty(self) -> None:
        """Empty result from sanitising must raise a 400 segment error.

        sanitize_filename is patched to return an empty string to exercise the
        branch where a segment is considered invalid after sanitisation.
        """
        with patch(
            'examples.violation_records.path_utils.sanitize_filename',
            return_value='',
        ):
            with self.assertRaises(HTTPException) as cm:
                normalise_safe_relative_path('bad/segment.png')
        self.assertEqual(cm.exception.status_code, 400)
        self.assertIn('Invalid path segment', cm.exception.detail)

    def test_valid_sanitization_preserves_structure(self) -> None:
        """Sanitisation should keep directory structure intact."""
        raw: str = 'valid My/img 1.PNG'
        out: Path = normalise_safe_relative_path(raw)
        # Compute expected using the same sanitiser per segment.
        from examples.shared.filename_utils import sanitize_filename

        exp: Path = Path(sanitize_filename('valid My')) / sanitize_filename(
            'img 1.PNG',
        )
        self.assertEqual(out, exp)


class TestResolveAndAuthorize(unittest.TestCase):
    """Tests for resolving and authorising paths under a base directory."""

    def test_inside_base_dir_ok(self) -> None:
        """A path under the base directory should resolve successfully."""
        with TemporaryDirectory() as td:
            base: Path = Path(td)
            rel: Path = Path('a') / 'b.png'
            full: Path = resolve_authorised_media_path(base, rel, username='u')
            self.assertTrue(str(full).startswith(str(base.resolve())))
            self.assertEqual(full, (base / rel).resolve())

    def test_escape_outside_denied(self) -> None:
        """Paths that escape the base directory must be denied with 403."""
        with TemporaryDirectory() as td:
            base: Path = Path(td)
            # Intentionally attempt to go outside the base directory.
            rel: Path = Path('..') / 'x.png'
            with self.assertRaises(HTTPException) as cm:
                resolve_authorised_media_path(base, rel, username='u')
            self.assertEqual(cm.exception.status_code, 403)
            self.assertIn('Access denied', cm.exception.detail)


class TestDetermineMediaType(unittest.TestCase):
    """Tests for mapping file suffixes to HTTP media types."""

    def test_png_ok(self) -> None:
        """'.png' should map to image/png."""
        self.assertEqual(
            _determine_media_type(Path('a.png')),
            'image/png',
        )

    def test_jpg_ok(self) -> None:
        """'.jpg' should map to image/jpeg."""
        self.assertEqual(
            _determine_media_type(Path('a.jpg')),
            'image/jpeg',
        )

    def test_jpeg_ok(self) -> None:
        """'.jpeg' (case-insensitive) should map to image/jpeg."""
        self.assertEqual(
            _determine_media_type(Path('a.JPEG')),
            'image/jpeg',
        )

    def test_unsupported_gif(self) -> None:
        """Unsupported suffixes should raise HTTP 400."""
        with self.assertRaises(HTTPException) as cm:
            _determine_media_type(Path('a.gif'))
        self.assertEqual(cm.exception.status_code, 400)
        self.assertIn('Unsupported file type', cm.exception.detail)


if __name__ == '__main__':
    unittest.main()
