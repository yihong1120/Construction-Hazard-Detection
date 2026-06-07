from __future__ import annotations

from ultralytics import YOLO

model = YOLO('../../models/pt/best_yolo11x.pt')

model.export(format='onnx', half=True, device=0)  # 必須指定 device=0 (GPU)
