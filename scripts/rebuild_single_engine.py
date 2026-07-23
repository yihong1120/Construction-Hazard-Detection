#!/usr/bin/env python3
"""Rebuild one TensorRT engine file.

Usage:
    python scripts/rebuild_single_engine.py yolo11m
"""
from __future__ import annotations

import sys
from pathlib import Path

from ultralytics import YOLO


def _pick_calibration_data_yaml(project_root: Path) -> str:
    """Pick a dataset yaml for INT8 calibration."""
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


def main() -> None:
    """Rebuild the requested TensorRT engine."""
    if len(sys.argv) < 2:
        print('使用方法: python scripts/rebuild_single_engine.py <model_name>')
        print('示例: python scripts/rebuild_single_engine.py yolo11m')
        sys.exit(1)

    model_name = sys.argv[1]
    if not model_name.startswith('best_'):
        model_name = f'best_{model_name}'

    project_root = Path(__file__).parent.parent
    pt_path = project_root / 'models' / 'pt' / f'{model_name}.pt'
    output_dir = project_root / 'models' / 'int8_engine'

    if not pt_path.exists():
        print(f"❌ 找不到模型文件: {pt_path}")
        sys.exit(1)

    print(f"🚀 重新构建 {model_name}")
    print(f"   输入: {pt_path}")
    print(f"   输出: {output_dir}")
    calibration_data = _pick_calibration_data_yaml(project_root)
    print(f"   校准: {calibration_data}")

    model = YOLO(str(pt_path))

    engine_path = model.export(
        format='engine',
        device=0,
        dynamic=False,
        batch=1,
        workspace=4,
        quantize=8,
        data=calibration_data,
    )

    engine_file = Path(engine_path)
    target_path = output_dir / f'{model_name}.engine'
    output_dir.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        target_path.unlink()
    if engine_file != target_path:
        engine_file.rename(target_path)
    print(f"✅ 导出完成: {target_path}")


if __name__ == '__main__':
    main()
