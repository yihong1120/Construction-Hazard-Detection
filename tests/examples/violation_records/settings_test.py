from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path


class SettingsStaticDirTest(unittest.TestCase):
    def test_static_dir_is_path_and_default(self) -> None:
        from examples.violation_records import settings

        # Should exist and be a Path
        self.assertTrue(hasattr(settings, 'STATIC_DIR'))
        self.assertIsInstance(settings.STATIC_DIR, Path)

        # Default should be a relative path named 'static'
        self.assertEqual(settings.STATIC_DIR.name, 'static')
        self.assertFalse(settings.STATIC_DIR.is_absolute())

    def test_violation_manager_uses_settings_static_dir_by_default(
        self,
    ) -> None:
        # Prepare a temporary directory to serve as STATIC_DIR
        from examples.violation_records import settings

        original_static = settings.STATIC_DIR
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp_path = Path(td)
                # Point settings to temp dir, then reload violation_manager
                settings.STATIC_DIR = tmp_path

                # If module is already imported, reload it to pick up
                # new STATIC_DIR
                if (
                    'examples.violation_records.violation_manager'
                    in sys.modules
                ):
                    import examples.violation_records.violation_manager as vm
                    importlib.reload(vm)
                else:
                    import examples.violation_records.violation_manager as vm

                # Create instance without passing base_dir,
                # expect it to use STATIC_DIR
                mgr = vm.ViolationManager()
                self.assertEqual(mgr.base_dir, tmp_path)
                # __init__ ensures mkdir
                self.assertTrue(mgr.base_dir.exists())
        finally:
            # Restore and reload to not leak state to other tests
            settings.STATIC_DIR = original_static
            if 'examples.violation_records.violation_manager' in sys.modules:
                import examples.violation_records.violation_manager as vm
                importlib.reload(vm)


if __name__ == '__main__':
    unittest.main()

'''
pytest \
    --cov=examples.violation_records.search_utils \
    --cov-report=term-missing \
        tests/examples/violation_records/search_utils_test.py
'''
