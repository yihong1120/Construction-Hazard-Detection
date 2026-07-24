from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from examples.YOLO_train import export_int8_engine


def test_checkpoint_accepts_model_key(tmp_path: Path) -> None:
    model_dir = tmp_path / 'models' / 'pt'
    ckpt = model_dir / 'best_yolo26n.pt'
    ckpt.parent.mkdir(parents=True)
    ckpt.write_text('model')

    assert export_int8_engine.checkpoint(
        'yolo26n', model_dir,
    ) == ckpt.resolve()


def test_checkpoint_and_data_yaml_report_missing_files(tmp_path: Path) -> None:
    """Invalid model and calibration paths fail with actionable errors."""
    with pytest.raises(FileNotFoundError, match='checkpoint'):
        export_int8_engine.checkpoint('missing-model', tmp_path / 'models')

    with pytest.raises(FileNotFoundError, match='calibration data yaml'):
        export_int8_engine.data_yaml(str(tmp_path / 'missing-data.yaml'))


def test_checkpoint_prefers_model_dir_for_bare_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / 'repo'
    model_dir = root / 'models' / 'pt'
    preferred = model_dir / 'best_yolo26n.pt'
    preferred.parent.mkdir(parents=True)
    preferred.write_text('preferred')
    cwd = root / 'examples' / 'YOLO_train'
    cwd.mkdir(parents=True)
    (cwd / 'yolo26n.pt').write_text('local')
    monkeypatch.setattr(export_int8_engine, 'ROOT', root)
    monkeypatch.chdir(cwd)

    assert export_int8_engine.checkpoint(
        'yolo26n', model_dir,
    ) == preferred.resolve()


def test_data_yaml_accepts_dataset_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / 'repo'
    dataset = root / 'examples' / 'YOLO_train' / 'cv_dataset'
    data_yaml = dataset / 'data.yaml'
    dataset.mkdir(parents=True)
    data_yaml.write_text('path: .')
    monkeypatch.setattr(export_int8_engine, 'ROOT', root)

    assert export_int8_engine.data_yaml(
        'examples/YOLO_train/cv_dataset',
    ) == data_yaml.resolve()


def test_calibration_yaml_keeps_valid_split_yaml(tmp_path: Path) -> None:
    dataset = tmp_path / 'dataset'
    (dataset / 'train' / 'images').mkdir(parents=True)
    (dataset / 'val' / 'images').mkdir(parents=True)
    data_yaml = dataset / 'data.yaml'
    data_yaml.write_text(
        f'path: {dataset}\ntrain: train/images\nval: val/images\n',
    )

    assert export_int8_engine.calibration_yaml(
        data_yaml, tmp_path / 'tmp',
    ) == data_yaml


def test_calibration_yaml_maps_flat_images_to_train_and_val(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / 'cv_dataset'
    images = dataset / 'images'
    images.mkdir(parents=True)
    (images / 'sample.jpg').write_text('image')
    data_yaml = dataset / 'data.yaml'
    data_yaml.write_text('train: train/images\nval: val/images\nnc: 1\n')

    generated = export_int8_engine.calibration_yaml(
        data_yaml, tmp_path / 'tmp',
    )

    assert generated == tmp_path / 'tmp' / 'data_int8_calibration.yaml'
    assert 'train: images' in generated.read_text()
    assert 'val: images' in generated.read_text()


def test_calibration_yaml_reports_missing_images(tmp_path: Path) -> None:
    dataset = tmp_path / 'cv_dataset'
    dataset.mkdir()
    data_yaml = dataset / 'data.yaml'
    data_yaml.write_text('train: train/images\nval: val/images\n')

    with pytest.raises(FileNotFoundError, match='Calibration images'):
        export_int8_engine.calibration_yaml(data_yaml, tmp_path / 'tmp')


def test_calibration_yaml_rejects_non_mapping_and_expands_list_entries(
    tmp_path: Path,
) -> None:
    """Calibration YAML requires a mapping and supports split path lists."""
    invalid_yaml = tmp_path / 'invalid.yaml'
    invalid_yaml.write_text('- images')
    with pytest.raises(ValueError, match='must be a mapping'):
        export_int8_engine.calibration_yaml(invalid_yaml, tmp_path / 'tmp')

    root = tmp_path / 'dataset'
    train = root / 'train'
    validation = root / 'val'
    train.mkdir(parents=True)
    validation.mkdir()
    data_yaml = root / 'data.yaml'
    data_yaml.write_text('train:\n  - train\nval:\n  - val\n')

    assert (
        export_int8_engine.calibration_yaml(data_yaml, tmp_path / 'tmp')
        == data_yaml
    )


def test_output_path_resolves_bare_and_project_relative_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / 'repo'
    output_dir = root / 'models' / 'int8_engine'
    monkeypatch.setattr(export_int8_engine, 'ROOT', root)

    assert export_int8_engine.output_path(Path('custom'), output_dir) == (
        output_dir / 'custom.engine'
    ).resolve()
    assert (
        export_int8_engine.output_path(
            Path('exports/custom.engine'),
            output_dir,
        )
        == (root / 'exports' / 'custom.engine').resolve()
    )
    assert export_int8_engine.output_path(None, output_dir) is None


def test_export_engine_passes_ultralytics_args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / 'models' / 'pt' / 'best_yolo26n.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text('model')
    data_yaml = tmp_path / 'data.yaml'
    data_yaml.write_text('path: .')
    output_dir = tmp_path / 'models' / 'int8_engine'
    calls: list[dict[str, Any]] = []

    class FakeYOLO:
        def __init__(self, model_path: str) -> None:
            self.model_path = Path(model_path)

        def export(self, **kwargs: Any) -> str:
            calls.append(kwargs)
            exported = self.model_path.with_suffix('.engine')
            exported.write_text('engine')
            return str(exported)

    monkeypatch.setattr(export_int8_engine, 'YOLO', FakeYOLO)
    args = SimpleNamespace(
        output_dir=output_dir,
        fraction=1.0,
        device='0',
        imgsz=640,
        batch=1,
        workspace=None,
        dynamic=False,
        overwrite=True,
    )

    exported = export_int8_engine.export_engine(
        checkpoint, data_yaml, None, args,
    )

    assert exported == output_dir / 'best_yolo26n.engine'
    assert calls == [{
        'format': 'engine',
        'quantize': 8,
        'data': str(data_yaml),
        'fraction': 1.0,
        'device': '0',
        'imgsz': 640,
        'batch': 1,
        'workspace': None,
        'dynamic': False,
    }]


def test_export_engine_respects_existing_output_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing engine outputs require explicit overwrite permission."""
    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_text('model')
    data_yaml = tmp_path / 'data.yaml'
    data_yaml.write_text('path: .')
    generated = tmp_path / 'model.engine'
    target = tmp_path / 'exports' / 'result.engine'
    target.parent.mkdir()
    target.write_text('old engine')

    class FakeYOLO:
        def __init__(self, _model_path: str) -> None:
            pass

        def export(self, **_kwargs: Any) -> str:
            generated.write_text('new engine')
            return str(generated)

    monkeypatch.setattr(export_int8_engine, 'YOLO', FakeYOLO)
    args = SimpleNamespace(
        output_dir=target.parent,
        fraction=1.0,
        device='0',
        imgsz=640,
        batch=1,
        workspace=None,
        dynamic=False,
        overwrite=False,
    )

    with pytest.raises(FileExistsError, match='already exists'):
        export_int8_engine.export_engine(checkpoint, data_yaml, target, args)

    args.overwrite = True
    assert export_int8_engine.export_engine(
        checkpoint,
        data_yaml,
        target,
        args,
    ) == target.resolve()
    assert target.read_text() == 'new engine'


def test_main_exports_every_requested_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI resolves paths once and exports each supplied model key."""
    model_dir = tmp_path / 'models'
    output_dir = tmp_path / 'engines'
    model_dir.mkdir()
    output_dir.mkdir()
    data = tmp_path / 'data.yaml'
    calibration = tmp_path / 'calibration.yaml'
    models = [model_dir / 'first.pt', model_dir / 'second.pt']
    exports: list[tuple[Path, Path, Path | None]] = []

    monkeypatch.setattr(export_int8_engine, 'data_yaml', lambda _value: data)
    monkeypatch.setattr(
        export_int8_engine,
        'calibration_yaml',
        lambda _data, _workdir: calibration,
    )
    monkeypatch.setattr(
        export_int8_engine,
        'checkpoint',
        lambda _name, _model_dir: models.pop(0),
    )

    def export(
        model: Path,
        exported_data: Path,
        target: Path | None,
        args: Any,
    ) -> Path:
        exports.append((model, exported_data, target))
        assert args.model_dir == model_dir.resolve()
        assert args.output_dir == output_dir.resolve()
        return output_dir / f'{model.stem}.engine'

    monkeypatch.setattr(export_int8_engine, 'export_engine', export)

    assert export_int8_engine.main([
        '--model-dir', str(model_dir),
        '--output-dir', str(output_dir),
        '--data', 'provided.yaml',
        'first',
        'second',
    ]) == 0
    assert exports == [
        (model_dir / 'first.pt', calibration, None),
        (model_dir / 'second.pt', calibration, None),
    ]


def test_main_rejects_output_with_multiple_models() -> None:
    with pytest.raises(SystemExit):
        export_int8_engine.main(['yolo26n', 'yolo26s', '--output', 'one'])
