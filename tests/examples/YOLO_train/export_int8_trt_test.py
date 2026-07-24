from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch


def _install_optional_onnx_stubs() -> None:
    """Provide only the import-time APIs needed when CI lacks ONNX wheels."""
    try:
        importlib.import_module('onnx')
    except ModuleNotFoundError:
        sys.modules['onnx'] = ModuleType('onnx')

    try:
        importlib.import_module('onnxruntime.quantization')
    except ModuleNotFoundError:
        onnxruntime = ModuleType('onnxruntime')
        quantization = ModuleType('onnxruntime.quantization')
        quantization.CalibrationDataReader = type(
            'CalibrationDataReader',
            (),
            {},
        )
        onnxruntime.quantization = quantization
        sys.modules['onnxruntime'] = onnxruntime
        sys.modules['onnxruntime.quantization'] = quantization


_install_optional_onnx_stubs()
subject = importlib.import_module('examples.YOLO_train.export_int8_trt')


class ResettableDataLoader:
    """Small dataloader stand-in that exposes the exporter contract."""

    def __init__(self, batches: list[dict[str, torch.Tensor]]) -> None:
        self.batches = batches
        self.dataset = list(
            range(sum(batch['img'].shape[0] for batch in batches)),
        )
        self.batch_size = batches[0]['img'].shape[0]
        self.reset_count = 0

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)

    def reset(self) -> None:
        self.reset_count += 1


def _install_modelopt(
    monkeypatch: pytest.MonkeyPatch,
    quantize: MagicMock,
) -> None:
    """Install the minimal nested module structure imported by the exporter."""
    modelopt = ModuleType('modelopt')
    modelopt_onnx = ModuleType('modelopt.onnx')
    modelopt_quantization = ModuleType('modelopt.onnx.quantization')
    modelopt_quantization.quantize = quantize
    monkeypatch.setitem(sys.modules, 'modelopt', modelopt)
    monkeypatch.setitem(sys.modules, 'modelopt.onnx', modelopt_onnx)
    monkeypatch.setitem(
        sys.modules,
        'modelopt.onnx.quantization',
        modelopt_quantization,
    )


def _export_args(
    tmp_path: Path,
    *,
    static: bool = False,
    batch: int = 4,
    calib_batch: int = 2,
    workspace: int | None = None,
) -> SimpleNamespace:
    """Build the parsed-argument shape consumed by the script entry point."""
    return SimpleNamespace(
        model=tmp_path / 'model.pt',
        output=tmp_path / 'output' / 'model.engine',
        data=tmp_path / 'data.yaml',
        static=static,
        workspace=workspace,
        batch=batch,
        calib_batch=calib_batch,
        device=0,
        imgsz=640,
        fraction=1.0,
    )


def test_calibration_reader_reads_all_batches_and_rewinds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The calibration reader normalizes every batch and resets correctly."""
    dataloader = ResettableDataLoader([
        {'img': torch.full((2, 3, 2, 2), 255, dtype=torch.uint8)},
        {'img': torch.full((1, 3, 2, 2), 128, dtype=torch.uint8)},
    ])
    progress = MagicMock()
    monkeypatch.setattr(subject, 'tqdm', lambda **_kwargs: progress)

    reader = subject.AllImagesCalibrationReader('images', dataloader)

    assert len(reader) == 2
    assert reader.get_first()['images'].shape == (2, 3, 2, 2)
    first = reader.get_next()
    second = reader.get_next()
    assert first is not None
    assert second is not None
    assert first['images'].max() == pytest.approx(1.0)
    assert second['images'].max() == pytest.approx(128 / 255)
    assert reader.get_next() is None
    assert reader.seen_images == 3
    assert dataloader.reset_count >= 3
    assert progress.update.call_args_list[-1].args == (1,)
    assert progress.close.call_count >= 1

    with pytest.raises(NotImplementedError, match='Range slicing'):
        reader.set_range(0, 1)


def test_modelopt_quantize_delegates_non_int8_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-INT8 requests keep the upstream Ultralytics implementation."""
    upstream = MagicMock(return_value='upstream.onnx')
    monkeypatch.setattr(subject, 'original_modelopt_quantize_onnx', upstream)

    result = subject.modelopt_quantize_onnx_all_images(
        'model.onnx',
        quantize='fp16',
        dataset='dataset',
        shape=(1, 3, 320, 320),
        dynamic=True,
        prefix='export',
    )

    assert result == 'upstream.onnx'
    upstream.assert_called_once_with(
        'model.onnx',
        quantize='fp16',
        dataset='dataset',
        shape=(1, 3, 320, 320),
        dynamic=True,
        prefix='export',
    )


def test_modelopt_quantize_rejects_unsupported_or_missing_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper explains missing upstream support and calibration data."""
    monkeypatch.setattr(subject, 'original_modelopt_quantize_onnx', None)
    with pytest.raises(RuntimeError, match='does not expose'):
        subject.modelopt_quantize_onnx_all_images('model.onnx', quantize=4)

    with pytest.raises(ValueError, match='calibration dataset'):
        subject.modelopt_quantize_onnx_all_images('model.onnx', quantize=8)


@pytest.mark.parametrize('dynamic', [False, True])
def test_modelopt_quantize_int8_uses_all_calibration_images(
    monkeypatch: pytest.MonkeyPatch,
    dynamic: bool,
) -> None:
    """INT8 conversion uses all calibration images and an optional shape."""
    quantize = MagicMock()
    _install_modelopt(monkeypatch, quantize)
    reader = SimpleNamespace(batch_size=2, total_images=5)
    onnx_model = SimpleNamespace(
        graph=SimpleNamespace(input=[SimpleNamespace(name='images')]),
    )
    monkeypatch.setattr(
        subject, 'AllImagesCalibrationReader', lambda *_: reader,
    )
    monkeypatch.setattr(subject, 'check_requirements', MagicMock())
    monkeypatch.setattr(
        subject.onnx, 'load',
        MagicMock(return_value=onnx_model),
    )

    output = subject.modelopt_quantize_onnx_all_images(
        '/tmp/model.onnx',
        quantize=8,
        dataset=object(),
        shape=(1, 3, 640, 640),
        dynamic=dynamic,
        prefix='INT8',
    )

    assert output == '/tmp/model.int8.onnx'
    kwargs = quantize.call_args.kwargs
    assert kwargs['calibration_data_reader'] is reader
    assert kwargs['output_path'] == '/tmp/model.int8.onnx'
    if dynamic:
        assert kwargs['calibration_shapes'] == 'images:2x3x640x640'
    else:
        assert 'calibration_shapes' not in kwargs


def test_set_calibration_batch_size_restores_export_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calibration changes only the temporary dataloader batch value."""
    seen_batches: list[int] = []

    def upstream(exporter: Any, _prefix: str) -> str:
        seen_batches.append(exporter.args.batch)
        return 'dataloader'

    monkeypatch.setattr(
        subject,
        'original_get_int8_calibration_dataloader',
        upstream,
    )
    original = subject.Exporter.get_int8_calibration_dataloader
    subject.set_calibration_batch_size(2)
    exporter = SimpleNamespace(args=SimpleNamespace(batch=8))
    try:
        assert subject.Exporter.get_int8_calibration_dataloader(
            exporter,
            'INT8',
        ) == 'dataloader'
    finally:
        subject.Exporter.get_int8_calibration_dataloader = original

    assert seen_batches == [2]
    assert exporter.args.batch == 8


def test_parse_args_reads_script_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The CLI parser accepts every public export option."""
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'export_int8_trt.py',
            '--model', str(tmp_path / 'model.pt'),
            '--output', str(tmp_path / 'result.engine'),
            '--data', str(tmp_path / 'data.yaml'),
            '--device', 'cuda:0',
            '--imgsz', '320',
            '--batch', '8',
            '--calib-batch', '2',
            '--workspace', '4',
            '--static',
            '--fraction', '0.5',
        ],
    )

    args = subject.parse_args()

    assert args.model == tmp_path / 'model.pt'
    assert args.output == tmp_path / 'result.engine'
    assert args.device == 'cuda:0'
    assert args.imgsz == 320
    assert args.batch == 8
    assert args.calib_batch == 2
    assert args.workspace == 4
    assert args.static is True
    assert args.fraction == 0.5


@pytest.mark.parametrize(
    ('batch', 'calib_batch', 'message'),
    [
        (1, 1, 'batch > 1'),
        (4, 0, 'must be >= 1'),
        (4, 5, 'should be <= --batch'),
    ],
)
def test_main_rejects_invalid_batch_combinations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    batch: int,
    calib_batch: int,
    message: str,
) -> None:
    """Invalid dynamic and calibration batches stop before export."""
    args = _export_args(tmp_path, batch=batch, calib_batch=calib_batch)
    monkeypatch.setattr(subject, 'parse_args', lambda: args)

    with pytest.raises(SystemExit, match=message):
        subject.main()


@pytest.mark.parametrize(
    ('static', 'workspace', 'expected_workspace'),
    [(False, None, 2), (True, None, 16), (False, 6, 6)],
)
def test_main_exports_and_moves_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    static: bool,
    workspace: int | None,
    expected_workspace: int,
) -> None:
    """The command uses expected options and places the target engine."""
    args = _export_args(tmp_path, static=static, workspace=workspace)
    exported = tmp_path / 'intermediate.engine'
    exported.write_text('engine')
    calls: list[dict[str, object]] = []

    class FakeYolo:
        def __init__(self, model_path: str) -> None:
            self.model_path = model_path

        def export(self, **kwargs: object) -> str:
            calls.append(kwargs)
            return str(exported)

    move = MagicMock()
    monkeypatch.setattr(subject, 'parse_args', lambda: args)
    monkeypatch.setattr(subject, 'set_calibration_batch_size', MagicMock())
    monkeypatch.setattr(subject, 'YOLO', FakeYolo)
    monkeypatch.setattr(subject.shutil, 'move', move)

    subject.main()

    assert calls == [{
        'format': 'engine',
        'device': 0,
        'dynamic': not static,
        'batch': 4,
        'imgsz': 640,
        'workspace': expected_workspace,
        'quantize': 8,
        'data': str(args.data.resolve()),
        'fraction': 1.0,
    }]
    move.assert_called_once_with(
        str(exported.resolve()), args.output.resolve(),
    )


def test_script_main_block_invokes_main_before_model_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running the file as a script reaches the protected main block."""
    script = Path(subject.__file__).resolve()
    monkeypatch.setattr(sys, 'argv', [str(script), '--batch', '1'])

    with pytest.raises(SystemExit, match='batch > 1'):
        runpy.run_path(str(script), run_name='__main__')
