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


def test_calibration_yaml_maps_flat_images_to_train_and_val(tmp_path: Path) -> None:
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
    assert export_int8_engine.output_path(Path('exports/custom.engine'), output_dir) == (
        root / 'exports' / 'custom.engine'
    ).resolve()


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


def test_main_rejects_output_with_multiple_models() -> None:
    with pytest.raises(SystemExit):
        export_int8_engine.main(['yolo26n', 'yolo26s', '--output', 'one'])
