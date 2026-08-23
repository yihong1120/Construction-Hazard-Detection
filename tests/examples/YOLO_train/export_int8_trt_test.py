from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import yaml

from examples.YOLO_train import export_int8_trt as subject


class ResettableDataLoader:
    """Small dataloader stand-in that exposes the exporter contract."""

    def __init__(self, batches: list[dict[str, torch.Tensor]]) -> None:
        """Perform init.

        Args:
            batches: Value used by this callable.
        """
        self.batches = batches
        self.dataset = list(
            range(sum(batch['img'].shape[0] for batch in batches)),
        )
        self.batch_size = batches[0]['img'].shape[0]
        self.reset_count = 0

    def __iter__(self) -> Any:
        """Perform iter.

        Returns:
            The callable result.
        """
        return iter(self.batches)

    def __len__(self) -> int:
        """Perform len.

        Returns:
            The callable result.
        """
        return len(self.batches)

    def reset(self) -> None:
        """Perform reset.
        """
        self.reset_count += 1


def _install_modelopt(
    monkeypatch: pytest.MonkeyPatch,
    quantize: MagicMock,
) -> None:
    """Install the minimal nested module structure imported by the exporter."""
    modelopt = ModuleType('modelopt')
    modelopt_onnx = ModuleType('modelopt.onnx')
    modelopt_quantization = ModuleType('modelopt.onnx.quantization')
    setattr(modelopt_quantization, 'quantize', quantize)
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
        calib_images=4096,
        calib_seed=20260730,
        calib_split='val',
        calibration_method='entropy',
        exclude_node=[],
        exclude_detect_head=False,
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
    first_batch = reader.get_first()['images']
    assert isinstance(first_batch, np.ndarray)
    assert first_batch.shape == (2, 3, 2, 2)
    first = reader.get_next()
    second = reader.get_next()
    assert first is not None
    assert second is not None
    first_images = first['images']
    second_images = second['images']
    assert isinstance(first_images, np.ndarray)
    assert isinstance(second_images, np.ndarray)
    assert first_images.max() == pytest.approx(1.0)
    assert second_images.max() == pytest.approx(128 / 255)
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
    assert kwargs['calibration_method'] == 'entropy'
    assert kwargs['output_path'] == '/tmp/model.int8.onnx'
    if dynamic:
        assert kwargs['calibration_shapes'] == 'images:2x3x640x640'
    else:
        assert 'calibration_shapes' not in kwargs


def test_modelopt_quantize_int8_keeps_excluded_nodes_high_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mixed precision export forwards requested ONNX exclusion patterns."""
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
    monkeypatch.setattr(
        subject,
        'node_exclusion_patterns',
        (r'/model\\.23/.*',),
    )

    subject.modelopt_quantize_onnx_all_images(
        '/tmp/model.onnx',
        quantize=8,
        dataset=object(),
    )

    assert quantize.call_args.kwargs['nodes_to_exclude'] == [
        r'/model\\.23/.*',
    ]


def test_patched_onnx2engine_uses_batch_only_explicit_qdq() -> None:
    """TensorRT export patch keeps dynamic shape to batch 1..N only."""
    source = subject.original_onnx2engine_source
    assert source is not None
    patched = subject._patched_onnx2engine_source(
        source,
    )

    assert 'min_shape = (1, *shape[1:])' in patched
    assert 'opt_shape = shape' in patched
    assert 'max_shape = shape' in patched
    assert 'force_explicit_int8 = use_int8 and FORCE_EXPLICIT_INT8' in patched
    assert 'if force_explicit_int8:' in patched
    assert 'use_int8 = False' in patched
    assert 'use_fp16 = True' in patched


def test_modelopt_export_api_is_available() -> None:
    """The pinned Ultralytics version provides the required ModelOpt API."""
    if subject.ultralytics_version != subject.required_ultralytics_version:
        pytest.skip('requires the lockfile-pinned Ultralytics release')
    assert subject.ultralytics_version == subject.required_ultralytics_version
    assert subject.original_modelopt_quantize_onnx is not None


def test_modelopt_export_api_requires_pinned_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An incompatible environment gets a direct dependency repair message."""
    monkeypatch.setattr(subject, 'ultralytics_version', '0.0.0')

    with pytest.raises(RuntimeError, match='uv sync --locked'):
        subject._require_modelopt_export_support()


def test_patched_onnx2engine_accepts_compatible_modelopt_gate() -> None:
    """The exporter patch tolerates ModelOpt formatting and conditions."""
    source = subject.original_onnx2engine_source
    assert source is not None
    original_condition = '    if is_trt11 and (use_fp16 or use_int8):'
    original_call = (
        '        onnx_file = modelopt_quantize_onnx('
        'onnx_file, quantize, dataset, shape, dynamic, prefix)'
    )
    multiline_call = '\n'.join([
        '        onnx_file = modelopt_quantize_onnx(',
        '            onnx_file,',
        '            quantize,',
        '            dataset,',
        '            shape,',
        '            dynamic,',
        '            prefix,',
        '        )',
    ])
    assert original_condition in source
    assert original_call in source

    compatible_source = source.replace(
        original_condition,
        '    if use_fp16 or use_int8:',
    ).replace(
        original_call,
        multiline_call,
    )
    patched = subject._patched_onnx2engine_source(compatible_source)

    assert 'force_explicit_int8 = use_int8 and FORCE_EXPLICIT_INT8' in patched
    assert 'if (use_fp16 or use_int8) or force_explicit_int8:' in patched
    assert multiline_call in patched


def test_set_calibration_batch_size_restores_export_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calibration changes only the temporary dataloader batch value."""
    seen_batches: list[int] = []

    def upstream(exporter: Any, _prefix: str) -> str:
        """Perform upstream.

        Args:
            exporter: Value used by this callable.
            _prefix: Value used by this callable.

        Returns:
            The callable result.
        """
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
            '--calib-images', '1024',
            '--calib-seed', '99',
            '--calib-split', 'train',
            '--calibration-method', 'max',
            '--exclude-node', 'custom_node.*',
            '--exclude-detect-head',
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
    assert args.calib_images == 1024
    assert args.calib_seed == 99
    assert args.calib_split == 'train'
    assert args.calibration_method == 'max'
    assert args.exclude_node == ['custom_node.*']
    assert args.exclude_detect_head is True
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

        """Provide FakeYolo.
        """

        def __init__(self, model_path: str) -> None:
            """Perform init.

            Args:
                model_path: Value used by this callable.
            """
            self.model_path = model_path

        def export(self, **kwargs: object) -> str:
            """Perform export.

            Args:
                **kwargs: Value used by this callable.

            Returns:
                The callable result.
            """
            calls.append(kwargs)
            return str(exported)

    move = MagicMock()
    monkeypatch.setattr(subject, 'parse_args', lambda: args)
    monkeypatch.setattr(subject, 'set_calibration_batch_size', MagicMock())
    monkeypatch.setattr(subject, 'set_calibration_method', MagicMock())
    set_node_exclusions = MagicMock()
    monkeypatch.setattr(subject, 'set_node_exclusions', set_node_exclusions)
    monkeypatch.setattr(
        subject,
        'prepare_calibration_data',
        lambda source, _workdir, _limit, _seed, _split: (source, 4096),
    )
    monkeypatch.setattr(
        subject,
        'build_data_yaml',
        lambda value: Path(value),
    )
    monkeypatch.setattr(subject, 'YOLO', FakeYolo)
    monkeypatch.setattr(subject.shutil, 'move', move)
    monkeypatch.setattr(subject, 'patch_tensorrt_engine_exporter', MagicMock())

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
    set_node_exclusions.assert_called_once_with([])


def test_prepare_calibration_data_uses_requested_split(
    tmp_path: Path,
) -> None:
    """Calibration selection can use original validation images."""
    root = tmp_path / 'dataset'
    for split in ('train', 'val'):
        (root / split / 'images').mkdir(parents=True)
        (root / split / 'labels').mkdir()
    for name, class_id in [('first', 0), ('second', 1), ('third', 0)]:
        (root / 'val' / 'images' / f'{name}.jpg').write_bytes(b'image')
        (root / 'val' / 'labels' / f'{name}.txt').write_text(
            f'{class_id} 0.5 0.5 0.2 0.2\n',
        )
    data = tmp_path / 'data.yaml'
    data.write_text(
        yaml.safe_dump({
            'path': str(root),
            'train': 'train/images',
            'val': 'val/images',
            'names': ['first', 'second'],
        }),
    )

    calibration, count = subject.prepare_calibration_data(
        data,
        tmp_path / 'work',
        image_limit=2,
        seed=7,
        split='val',
    )

    selected = yaml.safe_load(calibration.read_text())
    paths = Path(selected['val']).read_text().splitlines()
    assert count == 2
    assert len(paths) == 2
    assert all('/val/images/' in path for path in paths)


def test_class_balanced_images_excludes_calibration_images(
    tmp_path: Path,
) -> None:
    """A holdout subset can be selected without calibration overlap."""
    images = tmp_path / 'images'
    labels = tmp_path / 'labels'
    images.mkdir()
    labels.mkdir()
    for index in range(4):
        (images / f'{index}.jpg').write_bytes(b'image')
        (labels / f'{index}.txt').write_text(
            f'{index % 2} 0.5 0.5 0.2 0.2\n',
        )

    calibration = subject.class_balanced_images(images, labels, 2, seed=1)
    holdout = subject.class_balanced_images(
        images,
        labels,
        2,
        seed=2,
        excluded_images=set(calibration),
    )

    assert len(holdout) == 2
    assert set(calibration).isdisjoint(holdout)


def test_script_main_block_invokes_main_before_model_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running the file as a script reaches the protected main block."""
    script = Path(subject.__file__).resolve()
    monkeypatch.setattr(sys, 'argv', [str(script), '--batch', '1'])

    with pytest.raises(SystemExit, match='batch > 1'):
        runpy.run_path(str(script), run_name='__main__')


def test_calibration_helpers_reject_invalid_input_and_skip_duplicates(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed labels and calibration layouts fail with clear outcomes."""
    missing = tmp_path / 'missing.txt'
    assert subject.label_classes(missing) == set()
    labels = tmp_path / 'labels'
    images = tmp_path / 'images'
    labels.mkdir()
    images.mkdir()
    (labels / 'item.txt').write_text('\nnot-a-class\n2 0.5 0.5 1 1\n')
    image = images / 'item.jpg'
    image.write_bytes(b'image')
    assert subject.label_classes(labels / 'item.txt') == {2}

    duplicate = images / 'duplicate.jpg'
    duplicate.write_bytes(b'image')
    (labels / 'duplicate.txt').write_text('0 0 0 0 0\n1 0 0 0 0\n')
    selected = subject.class_balanced_images(images, labels, 1, seed=1)
    assert len(selected) == 1

    invalid_yaml = tmp_path / 'invalid.yaml'
    invalid_yaml.write_text('- images\n')
    monkeypatch.setattr(
        subject, 'build_calibration_yaml',
        lambda *_args: invalid_yaml,
    )
    with pytest.raises(ValueError, match='must be a mapping'):
        subject.prepare_calibration_data(invalid_yaml, tmp_path / 'work', 1, 1)

    split_yaml = tmp_path / 'split.yaml'
    split_yaml.write_text('path: .\nval: [images]\n')
    monkeypatch.setattr(
        subject, 'build_calibration_yaml',
        lambda *_args: split_yaml,
    )
    with pytest.raises(ValueError, match='must point to one images directory'):
        subject.prepare_calibration_data(split_yaml, tmp_path / 'work', 1, 1)

    missing_layout = tmp_path / 'missing-layout.yaml'
    missing_layout.write_text('path: .\nval: absent/images\n')
    monkeypatch.setattr(
        subject,
        'build_calibration_yaml',
        lambda *_args: missing_layout,
    )
    assert subject.prepare_calibration_data(
        missing_layout,
        tmp_path / 'work',
        1,
        1,
    ) == (missing_layout, 0)

    empty_root = tmp_path / 'empty'
    (empty_root / 'val' / 'images').mkdir(parents=True)
    (empty_root / 'val' / 'labels').mkdir()
    empty_yaml = tmp_path / 'empty.yaml'
    empty_yaml.write_text(f'path: {empty_root}\nval: val/images\n')
    monkeypatch.setattr(
        subject, 'build_calibration_yaml',
        lambda *_args: empty_yaml,
    )
    with pytest.raises(FileNotFoundError, match='No calibration images'):
        subject.prepare_calibration_data(empty_yaml, tmp_path / 'work', 1, 1)


def test_export_configuration_setters_and_negative_image_limit(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
) -> None:
    """Export helpers retain unique exclusions and reject negative limits."""
    subject.set_calibration_method('max')
    subject.set_node_exclusions(['head', 'head', 'tail'])
    assert subject.calibration_method == 'max'
    assert subject.node_exclusion_patterns == ('head', 'tail')

    args = _export_args(tmp_path)
    args.calib_images = -1
    monkeypatch.setattr(subject, 'parse_args', lambda: args)
    with pytest.raises(SystemExit, match='calib-images'):
        subject.main()


def test_exporter_patch_reports_unexpected_upstream_shapes(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported Ultralytics exporter layouts fail before engine export."""
    with pytest.raises(RuntimeError, match='dynamic TensorRT profile'):
        subject._patched_onnx2engine_source('def onnx2engine():\n    pass\n')

    source = subject.original_onnx2engine_source
    assert source is not None
    without_gate = source.replace('modelopt_quantize_onnx', 'other_quantize')
    with pytest.raises(RuntimeError, match='ModelOpt INT8 gate'):
        subject._patched_onnx2engine_source(without_gate)

    with (
        monkeypatch.context() as context,
    ):
        context.setattr(subject.ast, 'get_source_segment', lambda *_args: None)
        with pytest.raises(RuntimeError, match='INT8 call'):
            subject._patched_onnx2engine_source(source)

    gate = subject.ast.If(
        test=subject.ast.Constant(value=True),
        body=[
            subject.ast.Assign(
                targets=[subject.ast.Name(id='onnx_file')],
                value=subject.ast.Call(
                    func=subject.ast.Name(id='modelopt_quantize_onnx'),
                    args=[],
                    keywords=[],
                ),
            ),
        ],
        orelse=[],
    )
    gate.lineno = 1
    gate.col_offset = 0
    call = gate.body[0]
    call.lineno = 1
    call.col_offset = 0
    call.end_lineno = None
    call.end_col_offset = None
    with monkeypatch.context() as context:
        context.setattr(subject.ast, 'walk', lambda _tree: [gate])
        context.setattr(
            subject.ast,
            'get_source_segment',
            lambda *_args: 'onnx_file = modelopt_quantize_onnx()',
        )
        with pytest.raises(RuntimeError, match='call end'):
            subject._patched_onnx2engine_source(source)

    monkeypatch.setattr(subject, 'original_onnx2engine_source', None)
    monkeypatch.setattr(
        subject, '_require_modelopt_export_support', MagicMock(),
    )
    with pytest.raises(RuntimeError, match='source is unavailable'):
        subject.patch_tensorrt_engine_exporter()


def test_main_adds_detect_head_exclusion_to_mixed_precision_export(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
) -> None:
    """The convenience flag is forwarded to the ModelOpt exclusion setter."""
    args = _export_args(tmp_path)
    args.exclude_detect_head = True
    monkeypatch.setattr(subject, 'parse_args', lambda: args)
    monkeypatch.setattr(subject, 'set_calibration_batch_size', MagicMock())
    monkeypatch.setattr(subject, 'set_calibration_method', MagicMock())
    exclusions = MagicMock()
    monkeypatch.setattr(subject, 'set_node_exclusions', exclusions)
    monkeypatch.setattr(subject, 'patch_tensorrt_engine_exporter', MagicMock())
    monkeypatch.setattr(
        subject,
        'prepare_calibration_data',
        lambda source, *_args: (source, 0),
    )
    monkeypatch.setattr(subject, 'build_data_yaml', lambda value: Path(value))

    class FakeYolo:

        """Provide FakeYolo.
        """

        def __init__(self, _path: str) -> None:
            """Perform init.

            Args:
                _path: Value used by this callable.
            """
            pass

        def export(self, **_kwargs: object) -> str:
            """Perform export.

            Args:
                **_kwargs: Value used by this callable.

            Returns:
                The callable result.
            """
            return str(args.output)

    monkeypatch.setattr(subject, 'YOLO', FakeYolo)
    subject.main()

    exclusions.assert_called_once_with([subject.detect_head_node_pattern])
