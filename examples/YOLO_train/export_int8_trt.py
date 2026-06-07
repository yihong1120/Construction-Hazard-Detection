from __future__ import annotations

from ultralytics import YOLO

pt_file = '../../models/pt/best_yolo26n.pt'
trt_file = '../../models/int8_engine/best_yolo26n_int8.engine'
calibration_data_path = (
    './cv_dataset/data.yaml'
)

# export.export_engine(
#     pt_file,
#     engine_file=trt_file,
#     half=False,
#     int8=True,
#     dynamic=False,
#     workspace=4,
#     imgsz=640,
#     batch=8,
#     data=calibration_data_path
# )

model = YOLO(pt_file)
model.export(
    format='engine',
    dynamic=False,
    batch=1,
    workspace=16,
    int8=True,
    data=calibration_data_path,
)
