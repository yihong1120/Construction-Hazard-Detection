from __future__ import annotations

import argparse
import ast
import inspect
import random
import shutil
import sys
import tempfile
import textwrap
from collections import defaultdict
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any
from typing import cast
from typing import Protocol

import numpy as np
import numpy.typing as npt
import onnx
import torch
import yaml
from onnxruntime.quantization import CalibrationDataReader
from tqdm import tqdm
from ultralytics import __version__ as ultralytics_version
from ultralytics import YOLO
from ultralytics.engine.exporter import Exporter
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.export import engine as engine_export

base_dir = Path(__file__).resolve().parent
if not __package__:
    sys.path.insert(0, str(base_dir))
_export_int8_engine = import_module(
    f'{__package__}.export_int8_engine'
    if __package__
    else 'export_int8_engine',
)
build_calibration_yaml = cast(
    Callable[[Path, Path], Path],
    getattr(_export_int8_engine, 'calibration_yaml'),
)
build_data_yaml = cast(
    Callable[[str], Path],
    getattr(_export_int8_engine, 'data_yaml'),
)

default_pt_file = (base_dir / '../../models/pt/best_yolo26x.pt').resolve()
default_trt_file = (
    base_dir / '../../models/int8_engine/best_yolo26x.engine'
).resolve()
default_calibration_data_path = (base_dir / 'cv_dataset/data.yaml').resolve()
default_calibration_images = 1024
default_calibration_seed = 20260730
calibration_method = 'entropy'
image_suffixes = frozenset(
    {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'},
)
detect_head_node_pattern = r'/model\.23/.*'
node_exclusion_patterns: tuple[str, ...] = ()
required_ultralytics_version = '8.4.115'


class ProgressReporter(Protocol):
    """Minimal progress reporter used by the calibration reader."""

    def close(self) -> None:
        """Close the reporter."""

    def update(self, n: int = 1) -> object:
        """Advance the reporter."""


class AllImagesCalibrationReader(CalibrationDataReader):

    """Provide AllImagesCalibrationReader.
    """

    def __init__(self, input_name: str, dataloader: Any) -> None:
        """Perform init.

        Args:
            input_name: Value used by this callable.
            dataloader: Value used by this callable.
        """
        self.input_name = input_name
        self.dataloader = dataloader
        self.max_batches = len(dataloader)
        self.total_images = len(getattr(dataloader, 'dataset', []))
        self.batch_size = int(getattr(dataloader, 'batch_size', 1) or 1)
        self.progress: ProgressReporter | None = None
        self.rewind()

    def __len__(self) -> int:
        """Perform len.

        Returns:
            The callable result.
        """
        return self.max_batches

    def rewind(self) -> None:
        """Perform rewind.
        """
        self._close_progress()
        if hasattr(self.dataloader, 'reset'):
            self.dataloader.reset()
        self.iterator = iter(self.dataloader)
        self.seen_batches = 0
        self.seen_images = 0

    def _close_progress(self) -> None:
        """Perform close progress.
        """
        if self.progress is not None:
            self.progress.close()
            self.progress = None

    def set_range(self, start_index: int, end_index: int) -> None:
        """Perform set range.

        Args:
            start_index: Value used by this callable.
            end_index: Value used by this callable.
        """
        raise NotImplementedError(
            'Range slicing is not needed for this calibration reader.',
        )

    def _to_input(self, batch: Any) -> dict[str, npt.NDArray[np.float32]]:
        """Perform to input.

        Args:
            batch: Value used by this callable.

        Returns:
            The callable result.
        """
        images = batch['img'].to(torch.float32) / 255.0
        return {
            self.input_name: cast(
                npt.NDArray[np.float32],
                images.cpu().numpy(),
            ),
        }

    def get_first(self) -> dict[str, npt.NDArray[np.float32]]:
        """Perform get first.

        Returns:
            The callable result.
        """
        self.rewind()
        first = self._to_input(next(self.iterator))
        self.rewind()
        return first

    def get_next(self) -> dict[str, npt.NDArray[np.float32]] | None:
        """Perform get next.

        Returns:
            The callable result.
        """
        if self.seen_batches >= self.max_batches:
            self._close_progress()
            return None

        if self.progress is None:
            self.progress = tqdm(
                total=self.total_images,
                desc='ModelOpt calibration',
                unit='img',
                dynamic_ncols=True,
            )

        batch = next(self.iterator)
        self.seen_batches += 1
        batch_size = int(batch['img'].shape[0])
        self.seen_images += batch_size
        if self.progress is not None:
            self.progress.update(batch_size)

        if self.seen_batches >= self.max_batches:
            self._close_progress()

        return self._to_input(batch)


def label_classes(label_file: Path) -> set[int]:
    """Read YOLO class identifiers from one label file."""
    if not label_file.is_file():
        return set()

    classes: set[int] = set()
    for line in label_file.read_text(encoding='utf-8').splitlines():
        fields = line.split()
        if not fields:
            continue
        try:
            classes.add(int(float(fields[0])))
        except ValueError:
            continue
    return classes


def class_balanced_images(
    images_dir: Path,
    labels_dir: Path,
    limit: int,
    seed: int,
    excluded_images: set[Path] | None = None,
) -> list[Path]:
    """Choose a reproducible, roughly class-balanced calibration subset."""
    excluded_images = excluded_images or set()
    images = sorted(
        path for path in images_dir.rglob('*')
        if (
            path.is_file()
            and path.suffix.lower() in image_suffixes
            and path not in excluded_images
        )
    )
    if limit <= 0 or len(images) <= limit:
        return images

    class_images: dict[int, list[Path]] = defaultdict(list)
    for image in images:
        label = labels_dir / image.relative_to(images_dir).with_suffix('.txt')
        for class_id in label_classes(label):
            class_images[class_id].append(image)

    rng = random.Random(seed)
    selected: list[Path] = []
    selected_set: set[Path] = set()
    target_per_class = -(-limit // len(class_images)) if class_images else 0
    for class_id in sorted(class_images):
        candidates = class_images[class_id].copy()
        rng.shuffle(candidates)
        for image in candidates:
            if image in selected_set:
                continue
            selected.append(image)
            selected_set.add(image)
            if sum(
                image in selected_set for image in class_images[class_id]
            ) >= target_per_class:
                break

    remaining = [image for image in images if image not in selected_set]
    rng.shuffle(remaining)
    selected.extend(remaining[:max(0, limit - len(selected))])
    rng.shuffle(selected)
    return selected[:limit]


def prepare_calibration_data(
    source: Path,
    workdir: Path,
    image_limit: int,
    seed: int,
    split: str = 'val',
) -> tuple[Path, int]:
    """Build a temporary YOLO yaml that points at a balanced image list."""
    normalized = build_calibration_yaml(source, workdir)
    data = yaml.safe_load(normalized.read_text(encoding='utf-8')) or {}
    if not isinstance(data, dict):
        raise ValueError(f'Calibration yaml must be a mapping: {normalized}')

    configured_root = Path(data.get('path') or '.').expanduser()
    root = (
        configured_root
        if configured_root.is_absolute()
        else normalized.parent / configured_root
    ).resolve()
    split_entry = data.get(split)
    if not isinstance(split_entry, str):
        raise ValueError(
            f'Calibration split {split!r} must point to one images directory.',
        )
    images_dir = Path(split_entry).expanduser()
    images_dir = (
        images_dir
        if images_dir.is_absolute()
        else root / images_dir
    ).resolve()
    labels_dir = images_dir.parent / 'labels'
    if not images_dir.is_dir() or not labels_dir.is_dir():
        LOGGER.warning(
            f'Class-balanced sampling needs {split}/images and '
            f'{split}/labels folders; using the calibration yaml unchanged.',
        )
        return normalized, 0

    images = class_balanced_images(images_dir, labels_dir, image_limit, seed)
    if not images:
        raise FileNotFoundError(
            f'No calibration images found under {images_dir}',
        )

    workdir.mkdir(parents=True, exist_ok=True)
    image_list = workdir / f'int8_{split}_calibration_images.txt'
    image_list.write_text(
        ''.join(f'{image}\n' for image in images),
        encoding='utf-8',
    )
    data.update({
        'path': str(root),
        'train': str(image_list),
        'val': str(image_list),
    })
    data.pop('test', None)
    selected_yaml = workdir / 'int8_balanced_calibration.yaml'
    selected_yaml.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding='utf-8',
    )
    return selected_yaml, len(images)


original_modelopt_quantize_onnx = getattr(
    engine_export,
    'modelopt_quantize_onnx',
    None,
)
original_onnx2engine_source = cast(
    str | None,
    getattr(engine_export, '_hazard_original_onnx2engine_source', None),
)
if original_onnx2engine_source is None:
    original_onnx2engine_source = textwrap.dedent(
        inspect.getsource(engine_export.onnx2engine),
    )
    setattr(
        engine_export,
        '_hazard_original_onnx2engine_source',
        original_onnx2engine_source,
    )
original_get_int8_calibration_dataloader = (
    Exporter.get_int8_calibration_dataloader
)


def _require_modelopt_export_support() -> None:
    """Fail clearly when the pinned ModelOpt exporter API is unavailable."""
    if (
        ultralytics_version != required_ultralytics_version
        or original_modelopt_quantize_onnx is None
    ):
        raise RuntimeError(
            f'Mixed INT8 TensorRT export requires Ultralytics '
            f'{required_ultralytics_version} with ModelOpt support; found '
            f'{ultralytics_version}. Reinstall the pinned dependencies with '
            '`uv sync --locked`.',
        )


def modelopt_quantize_onnx_all_images(
    onnx_file: str,
    quantize: int | str | None = None,
    dataset: Any = None,
    shape: tuple[int, int, int, int] = (1, 3, 640, 640),
    dynamic: bool = False,
    prefix: str = '',
) -> str:
    """Perform modelopt quantize onnx all images.

    Args:
        onnx_file: Value used by this callable.
        quantize: Value used by this callable.
        dataset: Value used by this callable.
        shape: Value used by this callable.
        dynamic: Value used by this callable.
        prefix: Value used by this callable.

    Returns:
        The callable result.
    """
    if quantize != 8:
        if original_modelopt_quantize_onnx is None:
            raise RuntimeError(
                'This Ultralytics version does not expose '
                'modelopt_quantize_onnx.',
            )
        return original_modelopt_quantize_onnx(
            onnx_file,
            quantize=quantize,
            dataset=dataset,
            shape=shape,
            dynamic=dynamic,
            prefix=prefix,
        )

    if dataset is None:
        raise ValueError(
            'INT8 ModelOpt quantization requires a calibration dataset.',
        )

    check_requirements('nvidia-modelopt[onnx]>=0.44')
    from modelopt.onnx.quantization import quantize as modelopt_quantize

    input_name = (
        onnx.load(
            onnx_file,
            load_external_data=False,
        )
        .graph.input[0]
        .name
    )
    out_file = str(Path(onnx_file).with_suffix('.int8.onnx'))
    reader = AllImagesCalibrationReader(input_name, dataset)
    calibration_shape = (reader.batch_size, *shape[1:])
    calibration_shape_text = 'x'.join(
        str(dimension) for dimension in calibration_shape
    )
    kwargs: dict[str, str | list[str]] = (
        {
            'calibration_shapes': f"{input_name}:{calibration_shape_text}",
        }
        if dynamic
        else {}
    )
    if node_exclusion_patterns:
        kwargs['nodes_to_exclude'] = list(node_exclusion_patterns)

    LOGGER.info(
        f'{prefix} quantizing ONNX to INT8 with ModelOpt '
        f'({calibration_method}) '
        f"using {reader.total_images} calibration images...",
    )
    modelopt_quantize(
        onnx_file,
        quantize_mode='int8',
        calibration_data_reader=reader,
        calibration_method=calibration_method,
        calibration_eps=['cpu'],
        output_path=out_file,
        **kwargs,
    )
    return out_file


if original_modelopt_quantize_onnx is not None:
    engine_export.modelopt_quantize_onnx = modelopt_quantize_onnx_all_images


def _patched_onnx2engine_source(source: str) -> str:
    """Return Ultralytics onnx2engine source patched for this exporter.

    Ultralytics' ``dynamic=True`` TensorRT profile makes both batch and
    spatial dimensions dynamic. For live camera inference we want fixed
    640x640 spatial dimensions and only batch 1..N dynamic. On TensorRT 10,
    Ultralytics also defaults to implicit INT8 calibration, where ModelOpt
    node exclusions do not apply. This exporter intentionally forces explicit
    Q/DQ ModelOpt INT8 so
    ``--exclude-detect-head`` remains meaningful.
    """
    source = _replace_dynamic_profile(source)
    gate, modelopt_call = _find_modelopt_quantize_gate(source)
    return _replace_modelopt_gate(source, gate, modelopt_call)


def _dynamic_profile_source() -> str:
    """Return the upstream dynamic TensorRT profile expected by the patch."""
    return '\n'.join([
        '    if dynamic:',
        '        profile = builder.create_optimization_profile()',
        '        min_shape = (1, shape[1], 32, 32)  # minimum input shape',
        '        max_shape = (*shape[:2], *(int(max(2, workspace or 2) * '
        'd) for d in shape[2:]))  # max input shape',
        '        for inp in inputs:',
        '            profile.set_shape(inp.name, min=min_shape, opt=shape, '
        'max=max_shape)',
        '        config.add_optimization_profile(profile)',
        '',
    ])


def _batch_only_dynamic_profile_source() -> str:
    """Return the fixed-spatial profile used for dynamic live batching."""
    return '\n'.join([
        '    if dynamic:',
        '        profile = builder.create_optimization_profile()',
        '        min_shape = (1, *shape[1:])',
        '        opt_shape = shape',
        '        max_shape = shape',
        '        LOGGER.info(',
        '            f"{prefix} batch-only dynamic profile "',
        '            f"min={min_shape} opt={opt_shape} max={max_shape}"',
        '        )',
        '        for inp in inputs:',
        '            profile.set_shape(inp.name, min=min_shape, '
        'opt=opt_shape, max=max_shape)',
        '        config.add_optimization_profile(profile)',
        '',
    ])


def _replace_dynamic_profile(source: str) -> str:
    """Replace Ultralytics spatial dynamics with batch-only dynamics."""
    dynamic_profile = _dynamic_profile_source()
    if dynamic_profile not in source:
        raise RuntimeError(
            'Cannot patch Ultralytics dynamic TensorRT profile.',
        )
    return source.replace(
        dynamic_profile,
        _batch_only_dynamic_profile_source(),
    )


def _find_modelopt_quantize_gate(source: str) -> tuple[ast.If, ast.Assign]:
    """Locate the upstream condition containing the ModelOpt assignment."""
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.If):
            continue
        for statement in node.body:
            if (
                isinstance(statement, ast.Assign)
                and _is_modelopt_quantize_assignment(statement)
            ):
                return node, statement
    raise RuntimeError('Cannot patch Ultralytics ModelOpt INT8 gate.')


def _is_modelopt_quantize_assignment(statement: ast.stmt) -> bool:
    """Return whether a statement assigns ModelOpt output to ``onnx_file``."""
    if not isinstance(statement, ast.Assign):
        return False
    has_onnx_target = any(
        isinstance(target, ast.Name) and target.id == 'onnx_file'
        for target in statement.targets
    )
    call = statement.value
    return (
        has_onnx_target
        and isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == 'modelopt_quantize_onnx'
    )


def _replace_modelopt_gate(
    source: str,
    gate: ast.If,
    modelopt_call: ast.Assign,
) -> str:
    """Force an explicit Q/DQ gate around one upstream ModelOpt call."""
    lines = source.splitlines(keepends=True)
    gate_indent = lines[gate.lineno - 1][:gate.col_offset]
    call_indent = lines[modelopt_call.lineno - 1][:modelopt_call.col_offset]
    call_source = ast.get_source_segment(source, modelopt_call)
    if call_source is None:
        raise RuntimeError('Cannot read Ultralytics ModelOpt INT8 call.')
    call_end_line = modelopt_call.end_lineno
    call_end_column = modelopt_call.end_col_offset
    if call_end_line is None or call_end_column is None:
        raise RuntimeError('Cannot locate Ultralytics ModelOpt INT8 call end.')
    explicit_qdq_gate = '\n'.join([
        f'{gate_indent}force_explicit_int8 = use_int8 and FORCE_EXPLICIT_INT8',
        f'{gate_indent}if ({ast.unparse(gate.test)}) or '
        'force_explicit_int8:',
        f'{call_indent}{call_source}',
        f'{call_indent}if force_explicit_int8:',
        f'{call_indent}    use_int8 = False',
        f'{call_indent}    use_fp16 = True',
    ])
    line_offsets = _line_offsets(lines)
    gate_start = line_offsets[gate.lineno - 1] + gate.col_offset
    call_end = line_offsets[call_end_line - 1] + call_end_column
    return source[:gate_start] + explicit_qdq_gate + source[call_end:]


def _line_offsets(lines: list[str]) -> list[int]:
    """Return source-character offsets for one-based AST line locations."""
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    return offsets


def patch_tensorrt_engine_exporter() -> None:
    """Patch Ultralytics TensorRT export for batch-only dynamic mixed INT8."""
    _require_modelopt_export_support()
    namespace = dict(engine_export.__dict__)
    namespace['FORCE_EXPLICIT_INT8'] = True
    source = original_onnx2engine_source
    if source is None:
        raise RuntimeError('Ultralytics onnx2engine source is unavailable.')
    exec(_patched_onnx2engine_source(source), namespace)
    engine_export.onnx2engine = namespace['onnx2engine']


def set_calibration_batch_size(calib_batch: int) -> None:
    """Perform set calibration batch size.

    Args:
        calib_batch: Value used by this callable.
    """

    def get_int8_calibration_dataloader(self, prefix: str = '') -> Any:
        """Perform get int8 calibration dataloader.

        Args:
            prefix: Value used by this callable.

        Returns:
            The callable result.
        """
        export_batch = int(self.args.batch)
        self.args.batch = calib_batch
        try:
            LOGGER.info(
                f"{prefix} using calibration batch={calib_batch} "
                f"for engine batch profile={export_batch}",
            )
            return original_get_int8_calibration_dataloader(self, prefix)
        finally:
            self.args.batch = export_batch

    Exporter.get_int8_calibration_dataloader = get_int8_calibration_dataloader


def set_calibration_method(method: str) -> None:
    """Configure the ModelOpt method used by the exporter callback."""
    global calibration_method
    calibration_method = method


def set_node_exclusions(patterns: list[str]) -> None:
    """Configure ONNX node patterns that remain at high precision."""
    global node_exclusion_patterns
    node_exclusion_patterns = tuple(dict.fromkeys(patterns))


def parse_args() -> argparse.Namespace:
    """Perform parse args.

    Returns:
        The callable result.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Export an INT8 TensorRT engine with all calibration images.'
        ),
    )
    parser.add_argument('--model', type=Path, default=default_pt_file)
    parser.add_argument('--output', '-o', type=Path, default=default_trt_file)
    parser.add_argument(
        '--data',
        type=Path,
        default=default_calibration_data_path,
    )
    parser.add_argument('--device', default=0)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help=(
            'Max/optimization batch for dynamic TensorRT export. '
            'Default: 16.'
        ),
    )
    parser.add_argument(
        '--calib-batch',
        type=int,
        default=8,
        help=(
            'Batch size used only while reading calibration images. '
            'Default: 8.'
        ),
    )
    parser.add_argument(
        '--calib-images',
        type=int,
        default=default_calibration_images,
        help=(
            'Number of class-balanced images used for calibration. '
            'Use 0 to use every image. Default: 1024.'
        ),
    )
    parser.add_argument(
        '--calib-seed',
        type=int,
        default=default_calibration_seed,
        help='Seed for reproducible class-balanced calibration sampling.',
    )
    parser.add_argument(
        '--calib-split',
        choices=('train', 'val'),
        default='val',
        help='Dataset split used to select calibration images. Default: val.',
    )
    parser.add_argument(
        '--calibration-method',
        choices=('entropy', 'max'),
        default='entropy',
        help='ModelOpt INT8 calibration method. Default: entropy.',
    )
    parser.add_argument(
        '--exclude-node',
        action='append',
        default=[],
        metavar='REGEX',
        help='ONNX node regex to keep at high precision. May be repeated.',
    )
    parser.add_argument(
        '--exclude-detect-head',
        action='store_true',
        help='Keep the YOLO26 Detect head at FP16 while quantizing the rest.',
    )
    parser.add_argument(
        '--workspace',
        type=int,
        default=None,
        help=(
            'TensorRT workspace in GB. Default: 2 for dynamic, '
            '16 for static.'
        ),
    )
    parser.add_argument(
        '--static',
        action='store_true',
        help='Export a fixed-shape engine instead of dynamic batch/shape.',
    )
    parser.add_argument('--fraction', type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    """Perform main.
    """
    args = parse_args()
    pt_file = args.model.resolve()
    trt_file = args.output.resolve()
    calibration_data_path = args.data.resolve()
    dynamic = not args.static
    workspace = (
        args.workspace
        if args.workspace is not None
        else (2 if dynamic else 16)
    )

    if dynamic and args.batch <= 1:
        raise SystemExit(
            'dynamic TensorRT export needs --batch > 1, for example '
            '--batch 16. Use --static if you really want a fixed '
            'batch=1 engine.',
        )
    if args.calib_batch < 1:
        raise SystemExit('--calib-batch must be >= 1.')
    if args.calib_batch > args.batch:
        raise SystemExit('--calib-batch should be <= --batch.')
    if args.calib_images < 0:
        raise SystemExit('--calib-images must be >= 0.')

    node_exclusions = args.exclude_node.copy()
    if args.exclude_detect_head:
        node_exclusions.append(detect_head_node_pattern)

    print(f"Model: {pt_file}")
    print(f"Calibration data: {calibration_data_path}")
    print(f"Output: {trt_file}")
    if dynamic:
        print(f"Dynamic batch: True, batch range: 1..{args.batch}")
    else:
        print(f"Dynamic batch: False, fixed batch: {args.batch}")
    print(f"Calibration batch: {args.calib_batch}")
    print(f"Calibration method: {args.calibration_method}")
    if node_exclusions:
        print(f"FP16 ONNX node exclusions: {node_exclusions}")
    print(f"TensorRT workspace: {workspace} GB")

    set_calibration_batch_size(args.calib_batch)
    set_calibration_method(args.calibration_method)
    set_node_exclusions(node_exclusions)
    patch_tensorrt_engine_exporter()
    with tempfile.TemporaryDirectory(
        prefix='yolo_int8_calibration_',
    ) as workdir:
        data, selected_images = prepare_calibration_data(
            build_data_yaml(str(calibration_data_path)),
            Path(workdir),
            args.calib_images,
            args.calib_seed,
            args.calib_split,
        )
        if selected_images:
            print(f"Class-balanced calibration images: {selected_images}")
        exported = Path(
            YOLO(str(pt_file)).export(
                format='engine',
                device=args.device,
                dynamic=dynamic,
                batch=args.batch,
                imgsz=args.imgsz,
                workspace=workspace,
                quantize=8,
                data=str(data),
                fraction=args.fraction,
            ),
        ).resolve()

    trt_file.parent.mkdir(parents=True, exist_ok=True)
    if exported != trt_file:
        shutil.move(str(exported), trt_file)

    print(f"Exported: {trt_file}")


if __name__ == '__main__':
    main()
