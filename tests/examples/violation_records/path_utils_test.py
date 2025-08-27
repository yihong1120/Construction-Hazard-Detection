from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import HTTPException

from examples.violation_records.path_utils import _determine_media_type
from examples.violation_records.path_utils import _normalize_safe_rel_path
from examples.violation_records.path_utils import _resolve_and_authorize
from examples.violation_records.settings import STATIC_DIR


class TestNormalizeSafeRelPath(unittest.TestCase):
    """Tests for normalising and sanitising user-supplied relative paths."""

    def test_reject_absolute_path(self) -> None:
        """Absolute paths must be rejected with HTTP 400.

        Given an absolute file system path, when validated, the function
        should raise an HTTPException with a 400 status code.
        """
        abs_path: str = str(Path('/etc/passwd'))
        with self.assertRaises(HTTPException) as cm:
            _normalize_safe_rel_path(abs_path)
        self.assertEqual(cm.exception.status_code, 400)
        self.assertIn('Invalid path', cm.exception.detail)

    def test_reject_traversal_component(self) -> None:
        """Traversal tokens ('..') must be rejected with HTTP 400."""
        with self.assertRaises(HTTPException) as cm:
            _normalize_safe_rel_path('a/../b.png')
        self.assertEqual(cm.exception.status_code, 400)
        self.assertIn('Invalid path', cm.exception.detail)

    def test_strip_leading_static(self) -> None:
        """A leading 'static/' segment should be normalised away."""
        p: str = f"{STATIC_DIR.name}/2025-01-01/img.png"
        out: Path = _normalize_safe_rel_path(p)
        self.assertEqual(out, Path('2025-01-01') / 'img.png')

    def test_dot_segment_invalid(self) -> None:
        """A '.' segment must trigger a 400 Invalid path error.

        A simple FakePath is used to ensure the '.' segment survives any
        implicit normalisation that might be performed by the real Path.
        """

        class FakePath:
            """Minimal Path-like type to simulate dotted segments.

            Attributes:
                parts: The path segments.
            """

            def __init__(self, *parts: str) -> None:
                # If initialised with a single string, split by '/'.
                if len(parts) == 1 and isinstance(parts[0], str):
                    self.parts: tuple[str, ...] = tuple(parts[0].split('/'))
                else:
                    self.parts = tuple(parts)

            def is_absolute(self) -> bool:
                """Pretend the path is always relative."""
                return False

        with self.assertRaises(HTTPException) as cm:
            _normalize_safe_rel_path('valid/./img.png', path_cls=FakePath)
        self.assertEqual(cm.exception.status_code, 400)
        self.assertIn('Invalid path', cm.exception.detail)

    def test_segment_sanitization_empty(self) -> None:
        """Empty result from sanitising must raise a 400 segment error.

        secure_filename is patched to return an empty string to exercise the
        branch where a segment is considered invalid after sanitisation.
        """
        with patch(
            'examples.violation_records.path_utils.secure_filename',
            return_value='',
        ):
            with self.assertRaises(HTTPException) as cm:
                _normalize_safe_rel_path('bad/segment.png')
        self.assertEqual(cm.exception.status_code, 400)
        self.assertIn('Invalid path segment', cm.exception.detail)

    def test_valid_sanitization_preserves_structure(self) -> None:
        """Sanitisation should keep directory structure intact."""
        raw: str = 'valid My/img 1.PNG'
        out: Path = _normalize_safe_rel_path(raw)
        # Compute expected using the same sanitiser per segment.
        from examples.violation_records.path_utils import secure_filename

        exp: Path = (
            Path(secure_filename('valid My')) / secure_filename('img 1.PNG')
        )
        self.assertEqual(out, exp)


class TestResolveAndAuthorize(unittest.TestCase):
    """Tests for resolving and authorising paths under a base directory."""

    def test_inside_base_dir_ok(self) -> None:
        """A path under the base directory should resolve successfully."""
        with TemporaryDirectory() as td:
            base: Path = Path(td)
            rel: Path = Path('a') / 'b.png'
            full: Path = _resolve_and_authorize(base, rel, username='u')
            self.assertTrue(str(full).startswith(str(base.resolve())))
            self.assertEqual(full, (base / rel).resolve())

    def test_escape_outside_denied(self) -> None:
        """Paths that escape the base directory must be denied with 403."""
        with TemporaryDirectory() as td:
            base: Path = Path(td)
            # Intentionally attempt to go outside the base directory.
            rel: Path = Path('..') / 'x.png'
            with self.assertRaises(HTTPException) as cm:
                _resolve_and_authorize(base, rel, username='u')
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

"""
pytest \
  --cov=examples.violation_records.path_utils \
  --cov-report=term-missing \
  tests/examples/violation_records/path_utils_test.py
"""
