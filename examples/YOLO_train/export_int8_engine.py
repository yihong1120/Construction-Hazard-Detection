from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]


class ExportArguments(Protocol):
    """Describe CLI values required to export one TensorRT engine.

    Attributes:
        output_dir: Directory receiving an exported engine.
        fraction: Fraction of calibration data to use.
        device: Ultralytics execution device identifier.
        imgsz: Inference image size.
        batch: Calibration batch size.
        workspace: Optional TensorRT workspace size in GiB.
        dynamic: Whether to create a dynamic-shape engine.
        overwrite: Whether an existing output can be replaced.
    """

    output_dir: Path
    fraction: float
    device: str
    imgsz: int
    batch: int
    workspace: float | None
    dynamic: bool
    overwrite: bool


def checkpoint(name: str, model_dir: Path) -> Path:
    """Perform checkpoint.

    Args:
        name: Value used by this callable.
        model_dir: Value used by this callable.

    Returns:
        The callable result.
    """
    raw = Path(name).expanduser()
    raw_pt = raw if raw.suffix else raw.with_suffix('.pt')
    stem = raw.stem if raw.suffix else raw.name
    filename = stem if stem.startswith('best_') else f'best_{stem}'
    best = model_dir / f'{filename}.pt'
    if raw.is_absolute():
        candidates = [raw]
    elif raw.parent == Path('.') and not name.startswith(('.', '~')):
        candidates = [best, ROOT / raw_pt, Path.cwd() / raw_pt]
    else:
        candidates = [ROOT / raw_pt, Path.cwd() / raw_pt, best]
    for path in candidates:
        if path.exists() and path.suffix == '.pt':
            return path.resolve()
    raise FileNotFoundError(f"Cannot find .pt checkpoint for {name!r}")


def data_yaml(value: str) -> Path:
    """Perform data yaml.

    Args:
        value: Value used by this callable.

    Returns:
        The callable result.
    """
    raw = Path(value).expanduser()
    paths = [raw] if raw.is_absolute() else [ROOT / raw, Path.cwd() / raw]
    for path in paths:
        candidate = path / 'data.yaml' if path.is_dir() else path
        if candidate.exists() and candidate.suffix in {'.yaml', '.yml'}:
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot find calibration data yaml for {value!r}")


def yaml_entries(root: Path, value: object) -> list[Path]:
    """Perform yaml entries.

    Args:
        root: Value used by this callable.
        value: Value used by this callable.

    Returns:
        The callable result.
    """
    if isinstance(value, str):
        path = Path(value).expanduser()
        return [path if path.is_absolute() else root / path]
    return (
        [p for item in value for p in yaml_entries(root, item)]
        if isinstance(value, list)
        else []
    )


def calibration_yaml(source: Path, tmp: Path) -> Path:
    """Perform calibration yaml.

    Args:
        source: Value used by this callable.
        tmp: Value used by this callable.

    Returns:
        The callable result.
    """
    data = yaml.safe_load(source.read_text(encoding='utf-8')) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Calibration yaml must be a mapping: {source}")
    configured_root = Path(data.get('path') or '.').expanduser()
    if configured_root.is_absolute():
        root = configured_root
    else:
        root = source.parent / configured_root
    if not root.exists():
        root = source.parent
    root = root.resolve()
    if all(
        (paths := yaml_entries(root, data.get(k)))
        and all(p.exists() for p in paths)
        for k in ('train', 'val')
    ):
        return source
    images = root / 'images'
    if not images.is_dir():
        raise FileNotFoundError(f"Calibration images not found under {images}")
    data.update({'path': str(root), 'train': 'images', 'val': 'images'})
    data.pop('test', None)
    tmp.mkdir(parents=True, exist_ok=True)
    out = tmp / f"{source.stem}_int8_calibration.yaml"
    out.write_text(
        yaml.safe_dump(
            data,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding='utf-8',
    )
    return out


def output_path(value: Path | None, output_dir: Path) -> Path | None:
    """Perform output path.

    Args:
        value: Value used by this callable.
        output_dir: Value used by this callable.

    Returns:
        The callable result.
    """
    if value is None:
        return None
    value = value.expanduser()
    value = value if value.suffix else value.with_suffix('.engine')
    if value.is_absolute():
        return value.resolve()
    if value.parent == Path('.'):
        return (output_dir / value).resolve()
    return (ROOT / value).resolve()


def export_engine(
    model: Path,
    data: Path,
    target: Path | None,
    args: ExportArguments,
) -> Path:
    """Perform export engine.

    Args:
        model: Value used by this callable.
        data: Value used by this callable.
        target: Value used by this callable.
        args: Value used by this callable.

    Returns:
        The callable result.
    """
    exported = Path(
        YOLO(str(model)).export(
            format='engine',
            quantize=8,
            data=str(data),
            fraction=args.fraction,
            device=args.device,
            imgsz=args.imgsz,
            batch=args.batch,
            workspace=args.workspace,
            dynamic=args.dynamic,
        ),
    ).resolve()
    target = target or args.output_dir / f"{model.stem}.engine"
    target.parent.mkdir(parents=True, exist_ok=True)
    if exported != target:
        if target.exists():
            if not args.overwrite:
                raise FileExistsError(f"Output already exists: {target}")
            target.unlink()
        shutil.move(str(exported), str(target))
    return target.resolve()


def build_parser() -> argparse.ArgumentParser:
    """Perform build parser.

    Returns:
        The callable result.
    """
    parser = argparse.ArgumentParser(
        description='Export .pt checkpoints to INT8 TensorRT .engine files.',
    )
    parser.add_argument('models', nargs='+')
    parser.add_argument(
        '--model-dir',
        type=Path,
        default=ROOT / 'models' / 'pt',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=ROOT / 'models' / 'int8_engine',
    )
    parser.add_argument('-o', '--output', type=Path)
    parser.add_argument(
        '--data',
        default=str(
            ROOT / 'examples' / 'YOLO_train' / 'cv_dataset',
        ),
    )
    parser.add_argument('--device', default='0')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--workspace', type=float)
    parser.add_argument('--fraction', type=float, default=1.0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument(
        '--no-overwrite',
        dest='overwrite',
        action='store_false',
        default=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Perform main.

    Args:
        argv: Value used by this callable.

    Returns:
        The callable result.
    """
    args = build_parser().parse_args(argv)
    if args.output and len(args.models) != 1:
        build_parser().error('--output can only be used with one model')
    args.model_dir, args.output_dir = (
        args.model_dir.resolve(),
        args.output_dir.resolve(),
    )
    target = output_path(args.output, args.output_dir)
    with tempfile.TemporaryDirectory(
        prefix='yolo_int8_calibration_',
    ) as workdir:
        data = calibration_yaml(data_yaml(args.data), Path(workdir))
        print(f"[1/3] Calibration data: {data}")
        for name in args.models:
            model = checkpoint(name, args.model_dir)
            print(f"[2/3] Exporting INT8 TensorRT engine: {model}")
            print(
                f"[3/3] Exported: {export_engine(model, data, target, args)}",
            )
    return 0


if __name__ == '__main__':
    sys.exit(main())
