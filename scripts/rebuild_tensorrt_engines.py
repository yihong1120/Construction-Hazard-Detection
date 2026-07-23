#!/usr/bin/env python3
"""scripts/rebuild_tensorrt_engines.py

批量重建 TensorRT INT8 engine。

实现说明：
- 使用 Ultralytics 的 model.export(format='engine', quantize=8, data=...) 进行导出
- dynamic=False, batch=1, workspace=4
"""
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
    """从 PT 或 ONNX 模型重新导出 TensorRT engine。

    Args:
        model_path: 原始模型路径 (.pt 或 .onnx)
        output_dir: 输出目录
        model_name: 模型名称
        calibration_data: 校准数据 yaml
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

        # 统一输出位置与命名，供 server 读取
        target_path = output_dir / f'{model_name}.engine'
        final_path = _move_engine(engine_path, target_path)
        print(f"✅ 成功导出: {final_path}")
        return str(final_path)

    except Exception as e:
        print(f"❌ 导出失败: {e}", file=sys.stderr)
        traceback.print_exc()
        return None


def main() -> None:
    """主函数：批量重新构建所有 TensorRT engine"""

    # 项目根目录
    project_root = Path(__file__).parent.parent

    # 模型目录
    pt_dir = project_root / 'models' / 'pt'
    onnx_dir = project_root / 'models' / 'onnx'
    output_dir = project_root / 'models' / 'int8_engine'

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 备份旧的 engine 文件
    backup_dir = project_root / 'models' / 'int8_engine_backup'
    if any(output_dir.glob('*.engine')):
        print(f"\n📦 备份旧的 engine 文件到: {backup_dir}")
        backup_dir.mkdir(parents=True, exist_ok=True)
        for engine_file in output_dir.glob('*.engine'):
            backup_path = backup_dir / engine_file.name
            engine_file.rename(backup_path)
            print(f"   已备份: {engine_file.name}")

    # 模型列表
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
        # 优先使用 PT 文件，如果不存在则使用 ONNX
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

        # 重新导出
        engine_path = rebuild_engine(
            model_path=model_path,
            output_dir=output_dir,
            model_name=model_name,
            calibration_data=calibration_data,
        )
        results[model_name] = 'SUCCESS' if engine_path else 'FAILED'

    # 打印总结
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
    """获取 TensorRT 版本"""
    return trt.__version__


if __name__ == '__main__':
    main()
