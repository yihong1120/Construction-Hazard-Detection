from __future__ import annotations

import unittest

from examples.shared.filename_utils import sanitize_filename


class FilenameUtilsTests(unittest.TestCase):
    """Tests for filesystem-safe filename normalisation."""

    def test_windows_reserved_device_name_is_prefixed(self) -> None:
        """Windows device filenames remain safe when files are persisted."""
        self.assertEqual(sanitize_filename('CON.jpg'), '_CON.jpg')


if __name__ == '__main__':
    unittest.main()
