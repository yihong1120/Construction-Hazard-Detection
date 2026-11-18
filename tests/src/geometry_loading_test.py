"""Keep camera startup and PPE-only operation free of geometry runtimes."""
from __future__ import annotations

import subprocess
import sys


def test_ppe_and_empty_default_checks_defer_geometry_dependencies():
    result = subprocess.run(
        [
            sys.executable, '-c', '''
import sys
import src.stream_processor
from src.danger_detector import DangerDetector

assert not any(name in sys.modules for name in ('shapely', 'sklearn', 'scipy'))
detector = DangerDetector()
assert detector.detect_danger([]) == ({}, [], [])
warnings, cones, poles = detector.detect_danger([
    [0, 0, 10, 20, .9, 2, 0, 0],
    [0, 0, 10, 20, .9, 5, 1, 0],
])
assert warnings['warning_no_hardhat']['count'] == 1
assert not cones and not poles
assert detector._clusterer is None
assert not any(name in sys.modules for name in ('shapely', 'sklearn', 'scipy'))
detector.detect_danger([
    [0, 0, 10, 10, .9, 6, 0, 0],
    [100, 0, 110, 10, .9, 6, 1, 0],
    [0, 100, 10, 110, .9, 6, 2, 0],
])
assert detector._clusterer is not None
assert 'sklearn' in sys.modules
''',
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
