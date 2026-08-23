from __future__ import annotations

import sys
import traceback
from pathlib import Path

import tensorrt as trt
from ultralytics import YOLO


def _pick_calibration_data_yaml(project_root: Path) -> str:
    """Pick a local dataset yaml for INT8 calibration.

    Prefer the repo's own dataset under examples/YOLO_train/cv_dataset.
    Use coco128.yaml when no dataset config exists.
    """

    candidates = [
        (
            project_root / 'examples' / 'YOLO_train' / 'cv_dataset'
            / 'data_fold2.yaml'
        ),
        (
            project_root / 'examples' / 'YOLO_train' / 'cv_dataset'
            / 'data_fold3.yaml'
        ),
        project_root / 'examples' / 'YOLO_train' / 'cv_dataset' / 'data.yaml',
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return 'coco128.yaml'


def _move_engine(engine_path: str | Path, target_path: Path) -> Path:
    """Move an exported engine to its final path."""
    src = Path(engine_path)
    if src == target_path:
        return src
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        target_path.unlink()
    src.rename(target_path)
    return target_path


def rebuild_engine(
    model_path: Path,
    output_dir: Path,
    model_name: str,
    calibration_data: str,
) -> str | None:
    """Re-export a TensorRT engine from a PT or ONNX model.

    Args:
        model_path: Source model path (``.pt`` or ``.onnx``).
        output_dir: Destination directory for the exported engine.
        model_name: Name used for the output engine file.
        calibration_data: Calibration dataset YAML path.
    """
    print(f"\n{'='*60}")
    print(f"正在处理: {model_path.name}")
    print(f"{'='*60}")

    try:
        model = YOLO(str(model_path))

        export_kwargs = {
            'format': 'engine',
            'device': 0,
            'dynamic': False,
            'batch': 1,
            'workspace': 4,
            'quantize': 8,
            'data': calibration_data,
        }

        print(f"校准数据: {calibration_data}")
        print(f"导出参数: {export_kwargs}")

        engine_path = model.export(**export_kwargs)

        # Use the server's standard output location and file name.
        target_path = output_dir / f'{model_name}.engine'
        final_path = _move_engine(engine_path, target_path)
        print(f"✅ 成功导出: {final_path}")
        return str(final_path)

    except Exception as e:
        print(f"❌ 导出失败: {e}", file=sys.stderr)
        traceback.print_exc()
        return None


def main() -> None:
    """Rebuild every configured TensorRT engine in a batch."""

    # Repository root directory.
    project_root = Path(__file__).parent.parent

    # Model directories.
    pt_dir = project_root / 'models' / 'pt'
    onnx_dir = project_root / 'models' / 'onnx'
    output_dir = project_root / 'models' / 'int8_engine'

    # Ensure that the output directory exists.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Back up existing engine files.
    backup_dir = project_root / 'models' / 'int8_engine_backup'
    if any(output_dir.glob('*.engine')):
        print(f"\n📦 备份旧的 engine 文件到: {backup_dir}")
        backup_dir.mkdir(parents=True, exist_ok=True)
        for engine_file in output_dir.glob('*.engine'):
            backup_path = backup_dir / engine_file.name
            engine_file.rename(backup_path)
            print(f"   已备份: {engine_file.name}")

    # Models to rebuild.
    model_names = [
        'best_yolo11n',
        'best_yolo11s',
        'best_yolo11m',
        'best_yolo11l',
        'best_yolo11x',
    ]

    calibration_data = _pick_calibration_data_yaml(project_root)

    print('\n🚀 开始重新构建 TensorRT Engine 文件')
    print(f"TensorRT 版本: {get_tensorrt_version()}")
    print(f"输出目录: {output_dir}")
    print(f"模型数量: {len(model_names)}")

    results: dict[str, str] = {}

    for model_name in model_names:
        # Prefer PT files and fall back to ONNX files.
        pt_path = pt_dir / f'{model_name}.pt'
        onnx_path = onnx_dir / f'{model_name}.onnx'

        if pt_path.exists():
            model_path = pt_path
        elif onnx_path.exists():
            model_path = onnx_path
        else:
            print(f"⚠️  跳过 {model_name}: 找不到 PT 或 ONNX 文件")
            results[model_name] = 'NOT_FOUND'
            continue

        # Re-export the engine.
        engine_path = rebuild_engine(
            model_path=model_path,
            output_dir=output_dir,
            model_name=model_name,
            calibration_data=calibration_data,
        )
        results[model_name] = 'SUCCESS' if engine_path else 'FAILED'

    # Print the summary.
    print(f"\n{'='*60}")
    print('重建总结:')
    print(f"{'='*60}")
    for model_name, status in results.items():
        status_icon = {
            'SUCCESS': '✅',
            'FAILED': '❌',
            'NOT_FOUND': '⚠️ ',
        }.get(status, '❓')
        print(f"{status_icon} {model_name}: {status}")

    success_count = sum(1 for s in results.values() if s == 'SUCCESS')
    print(f"\n总计: {success_count}/{len(model_names)} 成功")


def get_tensorrt_version() -> str:
    """Return the installed TensorRT version."""
    return trt.__version__


if __name__ == '__main__':
    main()
