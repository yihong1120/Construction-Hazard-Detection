#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO
ROOT = Path(__file__).resolve().parents[2]


def checkpoint(name: str, model_dir: Path) -> Path:
    raw = Path(name).expanduser()
    raw_pt = raw if raw.suffix else raw.with_suffix('.pt')
    stem = raw.stem if raw.suffix else raw.name
    best = model_dir / \
        f"{stem if stem.startswith('best_') else f'best_{stem}'}.pt"
    candidates = [raw] if raw.is_absolute() else (
        [best, ROOT / raw_pt, Path.cwd() / raw_pt]
        if raw.parent == Path('.') and not name.startswith(('.', '~'))
        else [ROOT / raw_pt, Path.cwd() / raw_pt, best]
    )
    for path in candidates:
        if path.exists() and path.suffix == '.pt':
            return path.resolve()
    raise FileNotFoundError(f"Cannot find .pt checkpoint for {name!r}")


def data_yaml(value: str) -> Path:
    raw = Path(value).expanduser()
    for path in ([raw] if raw.is_absolute() else [ROOT / raw, Path.cwd() / raw]):
        candidate = path / 'data.yaml' if path.is_dir() else path
        if candidate.exists() and candidate.suffix in {'.yaml', '.yml'}:
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot find calibration data yaml for {value!r}")


def yaml_entries(root: Path, value: object) -> list[Path]:
    if isinstance(value, str):
        path = Path(value).expanduser()
        return [path if path.is_absolute() else root / path]
    return [p for item in value for p in yaml_entries(root, item)] if isinstance(value, list) else []


def calibration_yaml(source: Path, tmp: Path) -> Path:
    data = yaml.safe_load(source.read_text(encoding='utf-8')) or {}
    if not isinstance(data, dict):
        raise ValueError(f'Calibration yaml must be a mapping: {source}')
    configured_root = Path(data.get('path') or '.').expanduser()
    root = configured_root if configured_root.is_absolute() else source.parent / \
        configured_root
    root = root.resolve() if root.exists() else source.parent.resolve()
    if all((paths := yaml_entries(root, data.get(k))) and all(p.exists() for p in paths) for k in ('train', 'val')):
        return source
    images = root / 'images'
    if not images.is_dir():
        raise FileNotFoundError(f'Calibration images not found under {images}')
    data.update({'path': str(root), 'train': 'images', 'val': 'images'})
    data.pop('test', None)
    tmp.mkdir(parents=True, exist_ok=True)
    out = tmp / f'{source.stem}_int8_calibration.yaml'
    out.write_text(
        yaml.safe_dump(
            data, allow_unicode=True,
            sort_keys=False,
        ), encoding='utf-8',
    )
    return out


def output_path(value: Path | None, output_dir: Path) -> Path | None:
    if value is None:
        return None
    value = value.expanduser()
    value = value if value.suffix else value.with_suffix('.engine')
    return value.resolve() if value.is_absolute() else (output_dir / value if value.parent == Path('.') else ROOT / value).resolve()


def export_engine(model: Path, data: Path, target: Path | None, args) -> Path:
    exported = Path(
        YOLO(str(model)).export(
            format='engine', quantize=8, data=str(data), fraction=args.fraction,
            device=args.device, imgsz=args.imgsz, batch=args.batch, workspace=args.workspace, dynamic=args.dynamic,
        ),
    ).resolve()
    target = target or args.output_dir / f'{model.stem}.engine'
    target.parent.mkdir(parents=True, exist_ok=True)
    if exported != target:
        if target.exists():
            if not args.overwrite:
                raise FileExistsError(f'Output already exists: {target}')
            target.unlink()
        shutil.move(str(exported), str(target))
    return target.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Export .pt checkpoints to INT8 TensorRT .engine files.',
    )
    parser.add_argument('models', nargs='+')
    parser.add_argument(
        '--model-dir', type=Path,
        default=ROOT / 'models' / 'pt',
    )
    parser.add_argument(
        '--output-dir', type=Path,
        default=ROOT / 'models' / 'int8_engine',
    )
    parser.add_argument('-o', '--output', type=Path)
    parser.add_argument(
        '--data', default=str(
            ROOT /
            'examples' / 'YOLO_train' / 'cv_dataset',
        ),
    )
    parser.add_argument('--device', default='0')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--workspace', type=float)
    parser.add_argument('--fraction', type=float, default=1.0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument(
        '--no-overwrite', dest='overwrite',
        action='store_false', default=True,
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.output and len(args.models) != 1:
        build_parser().error('--output can only be used with one model')
    args.model_dir, args.output_dir = args.model_dir.resolve(), args.output_dir.resolve()
    target = output_path(args.output, args.output_dir)
    with tempfile.TemporaryDirectory(prefix='yolo_int8_calibration_') as workdir:
        data = calibration_yaml(data_yaml(args.data), Path(workdir))
        print(f'[1/3] Calibration data: {data}')
        for name in args.models:
            model = checkpoint(name, args.model_dir)
            print(f'[2/3] Exporting INT8 TensorRT engine: {model}')
            print(
                f'[3/3] Exported: {export_engine(model, data, target, args)}',
            )
    return 0


if __name__ == '__main__':
    sys.exit(main())
