from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_isolated(source: str) -> None:
    result = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(source)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ('module', 'deferred'),
    [
        (
            'src.local_yolo_detector',
            [
                'cv2',
                'numpy',
                'torch',
                'sahi',
                'ultralytics',
            ],
        ),
        ('src.yolo_detector', ['scipy', 'torch', 'ultralytics']),
        ('src.yolo_worker', ['torch', 'ultralytics']),
        ('examples.YOLO_server_api.config', ['torch']),
        ('examples.YOLO_server_api.models', ['torch', 'sahi', 'ultralytics']),
        (
            'examples.YOLO_server_api.detection',
            ['torch', 'sahi', 'ultralytics'],
        ),
        (
            'examples.YOLO_server_api.overlay_response',
            ['sahi', 'sklearn', 'scipy'],
        ),
        (
            'examples.YOLO_server_api.model_registry',
            ['sqlalchemy', 'huggingface_hub'],
        ),
        (
            'examples.mcp_server.tools.inference',
            [
                'cv2',
                'numpy',
                'torch',
                'ultralytics',
            ],
        ),
        ('examples.mcp_server.tools.hazard', ['cv2', 'numpy', 'sklearn']),
        ('examples.mcp_server.tools.notify', ['cv2', 'numpy', 'telegram']),
        ('examples.mcp_server.tools.streaming', ['cv2']),
    ],
)
def test_optional_dependencies_are_not_imported(
    module: str,
    deferred: list[str],
) -> None:
    """Configuration and unused tools must work without inference extras."""
    run_isolated(
        (
            '\n        import importlib\n        import sys\n  '
            '      blocked = '
            f"{deferred!r}"
            '\n\n        class RejectHeavyImports:\n           '
            ' def find_spec(self, fullname, path=None, '
            'target=None):\n                if '
            "fullname.split('.')[0] in blocked:\n              "
            "      raise AssertionError('Unexpected eager "
            "import: ' + fullname)\n\n        "
            'sys.meta_path.insert(0, RejectHeavyImports())\n   '
            '     importlib.import_module('
            f"{module!r}"
            ')\n        assert not any(name in sys.modules for '
            'name in blocked)\n    '
        ),
    )


def test_local_backends_resolve_only_when_selected() -> None:
    (
        'Both YOLO and SAHI resolve at first use without custom loader '
        'wrappers.'
    )
    run_isolated(
        (
            '\n        import asyncio\n        import '
            'sys\n        from types import '
            'ModuleType, SimpleNamespace\n        from '
            'src.local_yolo_detector import '
            'LocalYoloDetector\n\n        assert '
            "'ultralytics' not in sys.modules and "
            "'sahi' not in sys.modules\n        calls "
            "= []\n        yolo = ModuleType('ultralyti"
            "cs')\n        yolo.YOLO = lambda path: "
            "calls.append(('yolo', path)) or "
            "'yolo-model'\n        sys.modules['ultraly"
            "tics'] = yolo\n        detector = "
            "LocalYoloDetector(model_key='yolo26n')\n  "
            '      assert detector.ultralytics_model '
            "== 'yolo-model'\n        assert 'sahi' "
            'not in sys.modules\n\n        class '
            'Factory:\n            @staticmethod\n     '
            '       def from_pretrained(*args, '
            '**kwargs):\n                '
            "calls.append(('sahi', args, kwargs))\n    "
            "            return 'sahi-model'\n\n       "
            " sahi = ModuleType('sahi')\n        "
            'sahi.AutoDetectionModel = Factory\n       '
            " prediction = ModuleType('sahi.predict')\n"
            '        prediction.get_sliced_prediction '
            '= lambda *a, **kw: SimpleNamespace(object_'
            'prediction_list=[])\n        '
            "sys.modules.update({'sahi': sahi, "
            "'sahi.predict': prediction})\n        "
            "detector = LocalYoloDetector(model_key='yo"
            "lo26n', use_ultralytics=False)\n        "
            "assert detector.model == 'sahi-model'\n   "
            '     assert asyncio.run(detector._detect_l'
            'ocal(None)) == []\n        assert '
            "[entry[0] for entry in calls] == ['yolo', "
            "'sahi']\n    "
        ),
    )
